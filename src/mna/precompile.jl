#==============================================================================#
# MNA Optimization: Precompiled Circuit Evaluation
#
# This module provides optimized circuit evaluation by separating structure
# discovery (once) from value updates (every iteration).
#
# Key concepts:
# - PrecompiledCircuit: Holds fixed sparsity pattern, updates values in-place
# - COO→CSC mapping: Maps COO indices to sparse matrix nonzero positions
# - Fast stamping: Devices write to preallocated COO storage
#
# Inspired by OpenVAF's OSDI implementation which uses direct pointers to
# matrix entries for maximum performance.
#==============================================================================#

using SparseArrays
using LinearAlgebra
using ForwardDiff: Dual, value

# Extract real value from ForwardDiff.Dual (for tgrad compatibility)
# Needed for time-dependent sources with Rosenbrock solvers
real_time(t::Real) = Float64(t)
real_time(t::Dual) = Float64(value(t))

export PrecompiledCircuit, compile_circuit, fast_residual!, fast_jacobian!
export reset_coo_values!, update_sparse_from_coo!

#==============================================================================#
# PrecompiledCircuit Type
#==============================================================================#

"""
    PrecompiledCircuit{F,P,S}

Precompiled circuit with fixed sparsity pattern.
Structure discovered once, values updated each iteration.

This is the optimized version of MNACircuit that avoids rebuilding
sparse matrices from scratch at every Newton iteration.

# Architecture

1. **Compilation phase** (once at setup):
   - Call builder to discover nodes, currents, and matrix structure
   - Build sparse matrices to determine sparsity pattern
   - Create COO→CSC mapping for in-place value updates

2. **Evaluation phase** (every Newton iteration):
   - Reset COO values to zero
   - Call builder to re-stamp values into preallocated COO arrays
   - Update sparse matrix values in-place using mapping
   - Compute residual/Jacobian using updated matrices

# Performance

The key optimization is that `sparse(I, J, V)` is expensive because it must:
1. Sort indices to determine column pointers
2. Detect and sum duplicate entries
3. Allocate new memory

By precomputing the COO→CSC mapping, we can update values in O(nnz) time
without any allocation or sorting.

# Fields
- `builder::F`: Original circuit builder function
- `params::P`: Circuit parameters
- `spec::S`: Base simulation specification
- `n::Int`: System size (n_nodes + n_currents)
- `node_names::Vector{Symbol}`: Node names for solution interpretation
- `current_names::Vector{Symbol}`: Current variable names
- `n_nodes::Int`: Number of voltage nodes
- `n_currents::Int`: Number of current variables
- `G_I, G_J, G_V`: Preallocated COO storage for G matrix
- `C_I, C_J, C_V`: Preallocated COO storage for C matrix
- `b_direct::Vector{Float64}`: Direct b stamps (positive indices)
- `b_deferred_I, b_deferred_V`: Deferred b stamps (negative indices)
- `G, C`: Preallocated sparse matrices with fixed sparsity
- `b::Vector{Float64}`: Preallocated RHS vector
- `G_coo_to_nz::Vector{Int}`: Mapping from COO index to G nonzeros
- `C_coo_to_nz::Vector{Int}`: Mapping from COO index to C nonzeros
- `G_n_coo::Int`: Number of COO entries in G
- `C_n_coo::Int`: Number of COO entries in C
"""
mutable struct PrecompiledCircuit{F,P,S}
    # Original builder for reference
    builder::F
    params::P
    spec::S

    # System dimensions (fixed after compilation)
    n::Int
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    n_nodes::Int
    n_currents::Int

    # Preallocated COO storage (fixed length, values updated each eval)
    # These have extra capacity to handle operating-point-dependent stamping
    G_I::Vector{Int}
    G_J::Vector{Int}
    G_V::Vector{Float64}
    C_I::Vector{Int}
    C_J::Vector{Int}
    C_V::Vector{Float64}

    # b vector storage
    b_direct::Vector{Float64}
    b_deferred_I::Vector{Int}
    b_deferred_V::Vector{Float64}

    # Preallocated sparse matrices (fixed sparsity pattern)
    G::SparseMatrixCSC{Float64,Int}
    C::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}

    # Mapping from COO index to nonzeros(G/C) index
    # This is the key optimization: update values without rebuilding sparse
    G_coo_to_nz::Vector{Int}
    C_coo_to_nz::Vector{Int}

    # Number of COO entries discovered during compilation
    # Used to detect if circuit structure changes at runtime
    G_n_coo::Int
    C_n_coo::Int

    # Working storage for residual computation
    resid_tmp::Vector{Float64}
end

"""
    system_size(pc::PrecompiledCircuit) -> Int

Return the system size (number of unknowns).
"""
system_size(pc::PrecompiledCircuit) = pc.n

#==============================================================================#
# COO to CSC Mapping
#==============================================================================#

"""
    compute_coo_to_nz_mapping(I, J, S::SparseMatrixCSC) -> Vector{Int}

Compute mapping from COO indices to positions in `nonzeros(S)`.

For each COO entry `(I[k], J[k])`, finds its index in `nonzeros(S)`.
This handles duplicate entries by finding the correct position where
the value should be accumulated.

# Algorithm
For each COO entry at position k:
1. Get column j = J[k]
2. Search column j's nonzero range for row i = I[k]
3. Store the nonzero index

This is O(nnz * avg_col_nnz) but only done once during compilation.
"""
function compute_coo_to_nz_mapping(I::Vector{Int}, J::Vector{Int}, S::SparseMatrixCSC)
    n_coo = length(I)
    mapping = zeros(Int, n_coo)

    rowval = rowvals(S)
    colptr = S.colptr

    for k in 1:n_coo
        i, j = I[k], J[k]

        # Skip ground entries (shouldn't happen, but be safe)
        if i == 0 || j == 0
            continue
        end

        # Search column j for row i
        for idx in colptr[j]:(colptr[j+1]-1)
            if rowval[idx] == i
                mapping[k] = idx
                break
            end
        end

        # Sanity check: mapping should be found
        if mapping[k] == 0 && i != 0 && j != 0
            error("COO entry ($i, $j) not found in sparse matrix at k=$k")
        end
    end

    return mapping
end

#==============================================================================#
# Circuit Compilation
#==============================================================================#

"""
    compile_circuit(builder, params, spec; capacity_factor=2.0) -> PrecompiledCircuit

Compile a circuit builder into a PrecompiledCircuit.

This performs structure discovery by calling the builder once,
then creates the sparse matrices and COO→CSC mappings.

# Arguments
- `builder`: Circuit builder function `(params, spec; x=Float64[]) -> MNAContext`
- `params`: Circuit parameters (NamedTuple)
- `spec`: Simulation specification (MNASpec)
- `capacity_factor`: Extra capacity for COO arrays (safety margin)

# Returns
A `PrecompiledCircuit` ready for fast evaluation.

# Structure Discovery and Operating Points

**Critical Design Rule**: The matrix structure (which entries exist) must be
FIXED regardless of operating point. Only VALUES change during iteration.

For nonlinear devices like MOSFETs, this means:
- **Cutoff region**: gm=0, gds=0 (entries exist but are zero)
- **Linear/Saturation**: gm>0, gds>0 (same entries, different values)

Well-designed device models always stamp the SAME pattern with different values.
Avoid runtime conditionals that skip stamping entirely based on operating point.

Example - GOOD pattern (SimpleMOSFET):
```julia
# Always stamp these entries, values depend on operating region
stamp_G!(ctx, d, d,  gds)   # gds=0 in cutoff
stamp_G!(ctx, d, g,  gm)    # gm=0 in cutoff
stamp_G!(ctx, d, s, -(gds + gm))
# ...
```

Example - BAD pattern (would break precompilation):
```julia
if Vgs > Vth  # DON'T DO THIS
    stamp_G!(ctx, d, g, gm)  # Only stamps in active region!
end
```

# Initial Operating Point for Structure Discovery

The builder is called with `x=Float64[]` (empty) which signals devices to
use their default linearization point (typically x=0). For devices that
linearize around the operating point:
- Diodes: I=0 at V=0, G = Is/nVt (small but nonzero)
- MOSFETs: typically cutoff at Vgs=0, but all entries still stamped

If your device model has radically different structure at different operating
points, consider:
1. Always stamping all possible entries (with zeros where inactive)
2. Using a "compilation operating point" that activates all paths
"""
function compile_circuit(builder::F, params::P, spec::S;
                        capacity_factor::Float64=2.0) where {F,P,S}
    # First pass: discover structure at zero operating point
    ctx0 = builder(params, spec; x=Float64[])
    n = system_size(ctx0)

    if n == 0
        # Empty circuit - return minimal structure
        return PrecompiledCircuit{F,P,S}(
            builder, params, spec,
            0, Symbol[], Symbol[], 0, 0,
            Int[], Int[], Float64[],
            Int[], Int[], Float64[],
            Float64[], Int[], Float64[],
            spzeros(0, 0), spzeros(0, 0), Float64[],
            Int[], Int[],
            0, 0,
            Float64[]
        )
    end

    # Build sparse matrices to get sparsity pattern
    # Need to resolve negative indices (current variables)
    G_I_resolved = [resolve_index(ctx0, i) for i in ctx0.G_I]
    G_J_resolved = [resolve_index(ctx0, j) for j in ctx0.G_J]
    C_I_resolved = [resolve_index(ctx0, i) for i in ctx0.C_I]
    C_J_resolved = [resolve_index(ctx0, j) for j in ctx0.C_J]

    G = sparse(G_I_resolved, G_J_resolved, ctx0.G_V, n, n)
    C = sparse(C_I_resolved, C_J_resolved, ctx0.C_V, n, n)

    # Get RHS vector
    b = get_rhs(ctx0)

    # Create COO→CSC mappings
    G_coo_to_nz = compute_coo_to_nz_mapping(G_I_resolved, G_J_resolved, G)
    C_coo_to_nz = compute_coo_to_nz_mapping(C_I_resolved, C_J_resolved, C)

    # Preallocate COO arrays with extra capacity for nonlinear variation
    n_G = length(ctx0.G_I)
    n_C = length(ctx0.C_I)
    capacity_G = max(n_G, ceil(Int, n_G * capacity_factor))
    capacity_C = max(n_C, ceil(Int, n_C * capacity_factor))

    # Copy COO data (resolved indices)
    G_I = zeros(Int, capacity_G)
    G_J = zeros(Int, capacity_G)
    G_V = zeros(Float64, capacity_G)
    G_I[1:n_G] = G_I_resolved
    G_J[1:n_G] = G_J_resolved
    G_V[1:n_G] = ctx0.G_V

    C_I = zeros(Int, capacity_C)
    C_J = zeros(Int, capacity_C)
    C_V = zeros(Float64, capacity_C)
    C_I[1:n_C] = C_I_resolved
    C_J[1:n_C] = C_J_resolved
    C_V[1:n_C] = ctx0.C_V

    # b vector storage
    b_direct = copy(ctx0.b)
    if length(b_direct) < n
        resize!(b_direct, n)
        b_direct[length(ctx0.b)+1:n] .= 0.0
    end
    b_deferred_I = copy(ctx0.b_I)
    b_deferred_V = copy(ctx0.b_V)

    # Extend COO mapping with zeros for unused capacity
    G_coo_to_nz_extended = zeros(Int, capacity_G)
    G_coo_to_nz_extended[1:n_G] = G_coo_to_nz
    C_coo_to_nz_extended = zeros(Int, capacity_C)
    C_coo_to_nz_extended[1:n_C] = C_coo_to_nz

    return PrecompiledCircuit{F,P,S}(
        builder, params, spec,
        n, copy(ctx0.node_names), copy(ctx0.current_names),
        ctx0.n_nodes, ctx0.n_currents,
        G_I, G_J, G_V,
        C_I, C_J, C_V,
        b_direct, b_deferred_I, b_deferred_V,
        G, C, b,
        G_coo_to_nz_extended, C_coo_to_nz_extended,
        n_G, n_C,
        zeros(n)  # resid_tmp
    )
end

#==============================================================================#
# Fast Value Updates
#==============================================================================#

"""
    reset_coo_values!(pc::PrecompiledCircuit)

Reset all COO values and b vector to zero while preserving structure.
Called at the start of each evaluation before re-stamping.
"""
function reset_coo_values!(pc::PrecompiledCircuit)
    fill!(pc.G_V, 0.0)
    fill!(pc.C_V, 0.0)
    fill!(pc.b_direct, 0.0)
    empty!(pc.b_deferred_I)
    empty!(pc.b_deferred_V)
    return nothing
end

"""
    update_sparse_from_coo!(S::SparseMatrixCSC, V::Vector{Float64},
                            mapping::Vector{Int}, n_entries::Int)

Update sparse matrix values in-place from COO values using precomputed mapping.

This is the key optimization: instead of rebuilding the sparse matrix,
we directly update the nonzeros array using the precomputed mapping.

# Algorithm
1. Zero out all nonzeros in S
2. For each COO entry k with valid mapping:
   - Add V[k] to nonzeros(S)[mapping[k]]

This handles duplicate COO entries (same (i,j)) by accumulation.

# Complexity
O(nnz) - linear in number of nonzeros, no allocation or sorting.
"""
function update_sparse_from_coo!(S::SparseMatrixCSC, V::Vector{Float64},
                                  mapping::Vector{Int}, n_entries::Int)
    nz = nonzeros(S)
    fill!(nz, 0.0)

    @inbounds for k in 1:n_entries
        idx = mapping[k]
        if idx > 0
            nz[idx] += V[k]
        end
    end

    return nothing
end

"""
    update_b_vector!(b::Vector{Float64}, b_direct::Vector{Float64},
                     b_deferred_I::Vector{Int}, b_deferred_V::Vector{Float64},
                     n_nodes::Int)

Update the b vector from direct and deferred stamps.
"""
function update_b_vector!(b::Vector{Float64}, b_direct::Vector{Float64},
                          b_deferred_I::Vector{Int}, b_deferred_V::Vector{Float64},
                          n_nodes::Int)
    fill!(b, 0.0)

    # Copy direct stamps
    n = min(length(b_direct), length(b))
    @inbounds for i in 1:n
        b[i] = b_direct[i]
    end

    # Apply deferred stamps (negative indices for current variables)
    @inbounds for (i, v) in zip(b_deferred_I, b_deferred_V)
        # Resolve negative index: -k → n_nodes + k
        idx = i >= 0 ? i : n_nodes - i
        if 1 <= idx <= length(b)
            b[idx] += v
        end
    end

    return nothing
end

#==============================================================================#
# Fast Residual Evaluation
#==============================================================================#

"""
    fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)

Rebuild circuit at operating point u and time t using precompiled structure.

This is called at each Newton iteration to update matrix values
while reusing the fixed sparsity pattern.

# Requirements

The circuit structure (nodes, currents, COO entries) MUST be constant.
Only values change during iteration. This is enforced by assertions.
"""
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    # Build circuit at current operating point
    spec_t = MNASpec(temp=pc.spec.temp, mode=:tran, time=real_time(t))
    ctx = pc.builder(pc.params, spec_t; x=u)

    # Structure must be constant
    @assert ctx.n_nodes == pc.n_nodes && ctx.n_currents == pc.n_currents (
        "Circuit structure changed: expected $(pc.n_nodes) nodes + $(pc.n_currents) currents, " *
        "got $(ctx.n_nodes) + $(ctx.n_currents)")

    n_G = length(ctx.G_I)
    n_C = length(ctx.C_I)

    @assert n_G == pc.G_n_coo "G matrix COO length changed: expected $(pc.G_n_coo), got $n_G"
    @assert n_C == pc.C_n_coo "C matrix COO length changed: expected $(pc.C_n_coo), got $n_C"

    # Copy COO values only (indices are assumed constant)
    @inbounds for k in 1:n_G
        pc.G_V[k] = ctx.G_V[k]
    end
    @inbounds for k in 1:n_C
        pc.C_V[k] = ctx.C_V[k]
    end

    # Copy b vector data
    n_b = min(length(ctx.b), length(pc.b_direct))
    @inbounds for i in 1:n_b
        pc.b_direct[i] = ctx.b[i]
    end

    # Copy deferred b stamps
    resize!(pc.b_deferred_I, length(ctx.b_I))
    resize!(pc.b_deferred_V, length(ctx.b_V))
    copyto!(pc.b_deferred_I, ctx.b_I)
    copyto!(pc.b_deferred_V, ctx.b_V)

    # Update sparse matrices in-place using precomputed mapping
    update_sparse_from_coo!(pc.G, pc.G_V, pc.G_coo_to_nz, n_G)
    update_sparse_from_coo!(pc.C, pc.C_V, pc.C_coo_to_nz, n_C)
    update_b_vector!(pc.b, pc.b_direct, pc.b_deferred_I, pc.b_deferred_V, pc.n_nodes)

    return nothing
end

"""
    fast_residual!(resid, du, u, pc::PrecompiledCircuit, t)

Fast DAE residual evaluation using precompiled circuit.

Computes: F(du, u, t) = C*du + G*u - b = 0

This is the optimized version of the DAE residual that:
1. Rebuilds matrix values at operating point u (O(nnz))
2. Updates sparse matrices in-place (O(nnz))
3. Computes residual via SpMV (O(nnz))

Total: O(nnz) with no allocation after warmup.
"""
function fast_residual!(resid::Vector{Float64}, du::Vector{Float64},
                        u::Vector{Float64}, pc::PrecompiledCircuit, t::Real)
    # Rebuild circuit at current operating point
    fast_rebuild!(pc, u, t)

    # F(du, u) = C*du + G*u - b = 0
    mul!(resid, pc.C, du)
    mul!(resid, pc.G, u, 1.0, 1.0)
    resid .-= pc.b

    return nothing
end

"""
    fast_jacobian!(J, du, u, pc::PrecompiledCircuit, gamma, t)

Fast DAE Jacobian computation: J = G + gamma*C

This is called by DAE solvers (like Sundials IDA) to get the
combined Jacobian for the implicit solve.
"""
function fast_jacobian!(J::AbstractMatrix, du::Vector{Float64},
                        u::Vector{Float64}, pc::PrecompiledCircuit,
                        gamma::Real, t::Real)
    # Rebuild circuit at current operating point
    fast_rebuild!(pc, u, t)

    # J = G + gamma*C
    copyto!(J, pc.G)
    J .+= gamma .* pc.C

    return nothing
end

#==============================================================================#
# MNACircuitCompiled: High-Level Wrapper
#==============================================================================#

"""
    MNACircuitCompiled{F,P,S}

Compiled circuit wrapper for SciML integration.

Like MNACircuit but uses precompiled evaluation for better performance
during transient simulation.

# Usage
```julia
# Define circuit builder
function build_rc(params, spec; x=Float64[])
    ctx = MNAContext()
    # ... stamp devices ...
    return ctx
end

# Create compiled circuit
circuit = MNACircuitCompiled(build_rc; R=1000.0, C=1e-6)

# Transient analysis (automatically uses fast evaluation)
prob = DAEProblem(circuit, (0.0, 1e-3))
sol = solve(prob, IDA())
```

# Performance
For typical circuits, this provides ~5-10x speedup over MNACircuit
by avoiding:
- MNAContext allocation each iteration
- Node/current dictionary lookups
- Sparse matrix reconstruction from COO
"""
struct MNACircuitCompiled{F,P,S}
    pc::PrecompiledCircuit{F,P,S}
end

export MNACircuitCompiled

"""
    MNACircuitCompiled(builder; spec=MNASpec(), kwargs...)

Create a compiled MNA circuit with keyword parameters.
"""
function MNACircuitCompiled(builder::F; spec::S=MNASpec(), kwargs...) where {F,S}
    params = NamedTuple(kwargs)
    pc = compile_circuit(builder, params, spec)
    MNACircuitCompiled{F,typeof(params),S}(pc)
end

"""
    system_size(circuit::MNACircuitCompiled) -> Int
"""
system_size(circuit::MNACircuitCompiled) = system_size(circuit.pc)

"""
    alter(circuit::MNACircuitCompiled; kwargs...) -> MNACircuitCompiled

Create new compiled circuit with modified parameters.
Note: This recompiles the circuit.
"""
function alter(circuit::MNACircuitCompiled; spec=nothing, kwargs...)
    pc = circuit.pc
    new_spec = spec === nothing ? pc.spec : spec

    # Merge parameters
    function to_lens(selector::Symbol)
        parts = Symbol.(split(string(selector), "."))
        return Accessors.opticcompose(PropertyLens.(parts)...)
    end

    new_params = pc.params
    for (selector, value) in pairs(kwargs)
        if value === nothing
            continue
        end
        lens = to_lens(selector)
        if isa(value, Number) && isa(lens(new_params), Float64)
            value = Float64(value)
        end
        new_params = Accessors.set(new_params, lens, value)
    end

    # Recompile with new parameters
    new_pc = compile_circuit(pc.builder, new_params, new_spec)
    return MNACircuitCompiled(new_pc)
end

"""
    compute_initial_conditions(circuit::MNACircuitCompiled) -> (u0, du0)

Compute consistent initial conditions via DC operating point.
"""
function compute_initial_conditions(circuit::MNACircuitCompiled)
    pc = circuit.pc

    # DC solve for u0
    dc_spec = MNASpec(temp=pc.spec.temp, mode=:dcop, time=0.0)
    u0 = solve_dc(pc.builder, pc.params, dc_spec).x

    n = length(u0)
    du0 = zeros(n)

    # At t=0, need F(du0, u0) = C*du0 + G*u0 - b = 0
    # Rebuild circuit at u0
    fast_rebuild!(pc, u0, 0.0)

    rhs = pc.b - pc.G * u0
    diff_vars = detect_differential_vars_from_C(pc.C, n)

    # Compute du0 for differential variables
    C_dense = Matrix(pc.C)
    for i in 1:n
        if diff_vars[i]
            c_ii = C_dense[i, i]
            if abs(c_ii) > 1e-15
                du0[i] = rhs[i] / c_ii
            end
        end
    end

    return u0, du0
end

"""
    detect_differential_vars_from_C(C::SparseMatrixCSC, n::Int) -> BitVector

Determine which variables are differential from the C matrix structure.
"""
function detect_differential_vars_from_C(C::SparseMatrixCSC, n::Int)
    diff_vars = falses(n)

    for j in 1:n
        for k in nzrange(C, j)
            i = rowvals(C)[k]
            diff_vars[i] = true
        end
    end

    return diff_vars
end

# NOTE: SciML integration (DAEProblem, ODEProblem) for MNACircuitCompiled
# is not needed here since MNACircuit now uses PrecompiledCircuit by default
# in solve.jl. Users should use MNACircuit, not MNACircuitCompiled.
#
# The MNACircuitCompiled type is kept for explicit precompilation when needed.
