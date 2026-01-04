#==============================================================================#
# MNA Optimization: Precompiled Circuit Evaluation
#
# This module provides optimized circuit evaluation by separating structure
# discovery (once) from value updates (every iteration).
#
# Key concepts:
# - CompiledStructure: Immutable circuit structure (sparsity pattern, mappings)
# - EvalWorkspace: Mutable workspace for per-iteration values
# - COO→CSC mapping: Maps COO indices to sparse matrix nonzero positions
# - Fast stamping: Devices write to preallocated COO storage
#
# Inspired by OpenVAF's OSDI implementation which uses direct pointers to
# matrix entries for maximum performance.
#
# Architecture (zero-allocation iteration):
#   CompiledStructure (immutable, shared)
#       ↓
#   EvalWorkspace (mutable, per-thread)
#       ↓
#   fast_rebuild!(ws, u, t) → no allocation!
#==============================================================================#

using SparseArrays
using LinearAlgebra
using ForwardDiff: Dual, value
using StaticArrays: SVector, MVector
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# Extract real value from ForwardDiff.Dual (for tgrad compatibility)
# Needed for time-dependent sources with Rosenbrock solvers
real_time(t::Real) = Float64(t)
real_time(t::Dual) = Float64(value(t))

export CompiledStructure, EvalWorkspace, compile_structure, create_workspace
export PrecompiledCircuit, compile_circuit, fast_residual!, fast_jacobian!
export reset_coo_values!, update_sparse_from_coo!

#==============================================================================#
# CompiledStructure: Immutable Circuit Structure
#==============================================================================#

"""
    CompiledStructure{F,P,S}

Immutable compiled circuit structure containing everything that doesn't change
during Newton iterations.

This separation enables:
1. **Compiler optimization**: Fields can be constant-propagated
2. **Thread safety**: Can be shared across threads (each has own EvalWorkspace)
3. **Clear semantics**: Immutable parts shared, mutable parts per-evaluation

# Fields
- `builder::F`: Circuit builder function
- `params::P`: Circuit parameters (NamedTuple)
- `spec::S`: Base simulation spec (temp, mode, tolerances - NOT time)
- `n::Int`: System size (n_nodes + n_currents)
- `n_nodes::Int`: Number of voltage nodes
- `n_currents::Int`: Number of current variables
- `node_names::Vector{Symbol}`: Node names for solution interpretation
- `current_names::Vector{Symbol}`: Current variable names
- `G_coo_to_nz::Vector{Int}`: Mapping from COO index to G nonzeros
- `C_coo_to_nz::Vector{Int}`: Mapping from COO index to C nonzeros
- `G_n_coo::Int`: Number of COO entries in G
- `C_n_coo::Int`: Number of COO entries in C
- `G::SparseMatrixCSC`: Sparse G matrix (structure fixed, values in nzval)
- `C::SparseMatrixCSC`: Sparse C matrix (structure fixed, values in nzval)
"""
struct CompiledStructure{F,P,S}
    # Builder and parameters
    builder::F
    params::P
    spec::S

    # System dimensions (fixed)
    n::Int
    n_nodes::Int
    n_currents::Int

    # Node/current names for solution interpretation
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}

    # Fixed sparsity pattern mappings
    G_coo_to_nz::Vector{Int}
    C_coo_to_nz::Vector{Int}
    G_n_coo::Int
    C_n_coo::Int

    # Sparse matrices (structure fixed, values updated via nonzeros())
    G::SparseMatrixCSC{Float64,Int}
    C::SparseMatrixCSC{Float64,Int}

    # Resolved indices for deferred b stamps (CurrentIndex/ChargeIndex → actual positions)
    # Pre-computed during compilation for zero-allocation value-only mode
    b_deferred_resolved::Vector{Int}
    n_b_deferred::Int
end

"""
    system_size(cs::CompiledStructure) -> Int

Return the system size (number of unknowns).
"""
system_size(cs::CompiledStructure) = cs.n

#==============================================================================#
# EvalWorkspace: Mutable Per-Iteration Workspace
#==============================================================================#

"""
    EvalWorkspace{T,CS}

Immutable evaluation workspace containing preallocated storage for Newton iteration.
Passed as the `p` parameter to DAE solvers.

This struct is IMMUTABLE but contains MUTABLE vectors - the vector contents
can change while the struct itself cannot be reassigned. This gives us both
safety (no accidental field reassignment) and performance (mutable storage).

Time is passed explicitly to fast_rebuild! rather than stored, avoiding the
need for a mutable struct.

Each thread should have its own EvalWorkspace, while sharing the same
CompiledStructure.

# Fields
- `structure::CompiledStructure`: Reference to compiled circuit (immutable)
- `b::Vector{T}`: Preallocated RHS vector (mutable contents)
- `resid_tmp::Vector{T}`: Working storage for residual computation
- `ctx::MNAContext`: Preallocated context for fallback path
- `vctx::ValueOnlyContext{T}`: Zero-allocation stamping context
- `supports_ctx_reuse::Bool`: Whether builder supports ctx kwarg
- `supports_value_only_mode::Bool`: Whether to use ValueOnlyContext path

Note: G_V and C_V arrays are no longer stored here - in value-only mode
we stamp directly from vctx to sparse matrices (eliminating intermediate copy).
"""
struct EvalWorkspace{T,CS<:CompiledStructure}
    # Reference to immutable structure
    structure::CS

    # RHS vector (mutable contents, immutable reference)
    b::Vector{T}

    # Working storage
    resid_tmp::Vector{T}

    # Preallocated MNAContext for fallback Newton iteration path
    # Each EvalWorkspace has its own ctx for thread safety
    ctx::MNAContext

    # ValueOnlyContext for true zero-allocation stamping (no push!)
    # Used when supports_value_only_mode is true
    vctx::ValueOnlyContext{T}

    # Whether the builder supports ctx keyword argument for context reuse
    supports_ctx_reuse::Bool

    # Whether to use ValueOnlyContext for zero-allocation stamping
    # True when builder supports ctx and we can use vctx for value-only mode
    supports_value_only_mode::Bool
end

"""
    create_workspace(cs::CompiledStructure{F,P,S}) -> EvalWorkspace

Create an immutable evaluation workspace for the given compiled structure.

The workspace contains preallocated vectors whose contents are mutable,
but the struct itself is immutable (no field reassignment).
"""
function create_workspace(cs::CompiledStructure{F,P,S}) where {F,P,S}
    # Create a preallocated MNAContext for this workspace
    # We call the builder once to get a fully initialized context with correct structure
    ctx = cs.builder(cs.params, cs.spec, 0.0; x=ZERO_VECTOR)

    # Check if builder supports ctx reuse
    ctx_reuse = supports_ctx_kwarg(cs.builder, cs.params, cs.spec)

    # Create ValueOnlyContext for true zero-allocation stamping
    vctx = create_value_only_context(ctx)

    # Check if we can use value-only mode (ctx reuse is required)
    # We also verify the builder works with ValueOnlyContext
    value_only_mode = ctx_reuse && supports_value_only_ctx(cs.builder, cs.params, cs.spec, vctx)

    EvalWorkspace{Float64,typeof(cs)}(
        cs,
        zeros(Float64, cs.n),    # b vector
        zeros(Float64, cs.n),    # resid_tmp
        ctx,                     # Preallocated MNAContext (fallback)
        vctx,                    # ValueOnlyContext for zero-allocation mode
        ctx_reuse,               # Whether builder supports ctx reuse
        value_only_mode          # Whether to use ValueOnlyContext
    )
end

"""
    supports_value_only_ctx(builder, params, spec, vctx) -> Bool

Check if the builder works correctly with ValueOnlyContext.

This verifies that passing a ValueOnlyContext produces the same number of
stamps as the original MNAContext build.
"""
function supports_value_only_ctx(builder::F, params::P, spec::S, vctx::ValueOnlyContext) where {F,P,S}
    try
        # Reset and try a build with ValueOnlyContext
        reset_value_only!(vctx)
        builder(params, spec, 0.0; x=Float64[], ctx=vctx)

        # Check that positions match expected counts
        # G_pos should be n_G + 1 after stamping all G entries
        # C_pos should be n_C + 1 after stamping all C entries
        return (vctx.G_pos == vctx.n_G + 1) && (vctx.C_pos == vctx.n_C + 1)
    catch e
        # If it fails, fall back to MNAContext path
        return false
    end
end

"""
    system_size(ws::EvalWorkspace) -> Int

Return the system size from the workspace's structure.
"""
system_size(ws::EvalWorkspace) = system_size(ws.structure)

#==============================================================================#
# PrecompiledCircuit Type (Backwards Compatibility)
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
- `b_deferred_I, b_deferred_V`: Deferred b stamps (typed MNAIndex values)
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
    b_deferred_I::Vector{MNAIndex}
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

    # Preallocated MNAContext for zero-allocation Newton iteration
    # Reused by fast_rebuild! to avoid allocating a new context each iteration
    ctx::MNAContext

    # Whether the builder supports ctx keyword argument for context reuse
    # Builders from SPICE/VA codegen support this; hand-written builders may not
    supports_ctx_reuse::Bool
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
    compile_structure(builder, params, spec) -> CompiledStructure

Compile a circuit builder into an immutable CompiledStructure.

This performs structure discovery by calling the builder once,
then creates the sparse matrices and COO→CSC mappings.

# Arguments
- `builder`: Circuit builder function with signature:
    `(params, spec, t::Real=0.0; x=ZERO_VECTOR) -> MNAContext`
  Time is passed explicitly for zero-allocation iteration.
- `params`: Circuit parameters (NamedTuple)
- `spec`: Simulation specification (MNASpec)

# Returns
An immutable `CompiledStructure` that can be shared across threads.
Use `create_workspace(cs)` to create a mutable workspace for evaluation.
"""
function compile_structure(builder::F, params::P, spec::S) where {F,P,S}
    # First pass: discover structure at zero operating point (t=0.0)
    ctx0 = builder(params, spec, 0.0; x=ZERO_VECTOR)
    n = system_size(ctx0)

    if n == 0
        # Empty circuit - return minimal structure
        return CompiledStructure{F,P,S}(
            builder, params, spec,
            0, 0, 0,
            Symbol[], Symbol[],
            Int[], Int[],
            0, 0,
            spzeros(0, 0), spzeros(0, 0),
            Int[], 0
        )
    end

    # Build sparse matrices to get sparsity pattern
    # Resolve typed indices to actual matrix positions
    G_I_resolved = Int[resolve_index(ctx0, i) for i in ctx0.G_I]
    G_J_resolved = Int[resolve_index(ctx0, j) for j in ctx0.G_J]
    C_I_resolved = Int[resolve_index(ctx0, i) for i in ctx0.C_I]
    C_J_resolved = Int[resolve_index(ctx0, j) for j in ctx0.C_J]

    G = sparse(G_I_resolved, G_J_resolved, ctx0.G_V, n, n)
    C = sparse(C_I_resolved, C_J_resolved, ctx0.C_V, n, n)

    # Create COO→CSC mappings
    G_coo_to_nz = compute_coo_to_nz_mapping(G_I_resolved, G_J_resolved, G)
    C_coo_to_nz = compute_coo_to_nz_mapping(C_I_resolved, C_J_resolved, C)

    n_G = length(ctx0.G_I)
    n_C = length(ctx0.C_I)

    # Resolve deferred b stamp indices (CurrentIndex/ChargeIndex → actual b vector positions)
    # These are fixed after the first build and can be reused in value-only mode
    n_b_deferred = length(ctx0.b_I)
    b_deferred_resolved = Vector{Int}(undef, n_b_deferred)
    @inbounds for k in 1:n_b_deferred
        idx_typed = ctx0.b_I[k]
        b_deferred_resolved[k] = if idx_typed isa NodeIndex
            idx_typed.idx
        elseif idx_typed isa CurrentIndex
            ctx0.n_nodes + idx_typed.k
        elseif idx_typed isa ChargeIndex
            ctx0.n_nodes + ctx0.n_currents + idx_typed.k
        else
            0  # GroundIndex - skip
        end
    end

    return CompiledStructure{F,P,S}(
        builder, params, spec,
        n, ctx0.n_nodes, ctx0.n_currents,
        copy(ctx0.node_names), copy(ctx0.current_names),
        G_coo_to_nz, C_coo_to_nz,
        n_G, n_C,
        G, C,
        b_deferred_resolved, n_b_deferred
    )
end

"""
    supports_ctx_kwarg(builder, params, spec) -> Bool

Check if the builder function supports the `ctx` keyword argument for context reuse.

Builders generated by SPICE/VA codegen support this for zero-allocation Newton iteration.
Hand-written builders may not support it.
"""
function supports_ctx_kwarg(builder::F, params::P, spec::S) where {F,P,S}
    # Try calling with ctx kwarg - if it fails with MethodError, it's not supported
    # This is done ONCE at compile_circuit time, not every iteration
    try
        ctx_test = MNAContext()
        builder(params, spec, 0.0; x=Float64[], ctx=ctx_test)
        return true
    catch e
        if e isa MethodError
            return false
        end
        rethrow(e)  # Re-throw other errors
    end
end

"""
    compile_circuit(builder, params, spec; capacity_factor=2.0) -> PrecompiledCircuit

Compile a circuit builder into a PrecompiledCircuit.

This performs structure discovery by calling the builder once,
then creates the sparse matrices and COO→CSC mappings.

# Arguments
- `builder`: Circuit builder function with signature:
    `(params, spec, t::Real=0.0; x=ZERO_VECTOR) -> MNAContext`
  Time is passed explicitly for zero-allocation iteration.
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
    # First pass: discover structure at zero operating point (t=0.0)
    ctx0 = builder(params, spec, 0.0; x=ZERO_VECTOR)
    n = system_size(ctx0)

    # Check if builder supports ctx reuse (for zero-allocation path)
    ctx_reuse = supports_ctx_kwarg(builder, params, spec)

    if n == 0
        # Empty circuit - return minimal structure
        return PrecompiledCircuit{F,P,S}(
            builder, params, spec,
            0, Symbol[], Symbol[], 0, 0,
            Int[], Int[], Float64[],
            Int[], Int[], Float64[],
            Float64[], MNAIndex[], Float64[],
            spzeros(0, 0), spzeros(0, 0), Float64[],
            Int[], Int[],
            0, 0,
            Float64[],
            ctx0,       # Empty but preallocated context
            ctx_reuse   # Whether builder supports ctx reuse
        )
    end

    # Build sparse matrices to get sparsity pattern
    # Resolve typed indices to actual matrix positions
    G_I_resolved = Int[resolve_index(ctx0, i) for i in ctx0.G_I]
    G_J_resolved = Int[resolve_index(ctx0, j) for j in ctx0.G_J]
    C_I_resolved = Int[resolve_index(ctx0, i) for i in ctx0.C_I]
    C_J_resolved = Int[resolve_index(ctx0, j) for j in ctx0.C_J]

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
        zeros(n),   # resid_tmp
        ctx0,       # Preallocated context for zero-allocation iteration
        ctx_reuse   # Whether builder supports ctx reuse
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
                     b_deferred_I::Vector{MNAIndex}, b_deferred_V::Vector{Float64},
                     n_nodes::Int, n_currents::Int)

Update the b vector from direct and deferred stamps.
Deferred stamps use typed MNAIndex values resolved using n_nodes and n_currents.
"""
function update_b_vector!(b::Vector{Float64}, b_direct::Vector{Float64},
                          b_deferred_I::Vector{MNAIndex}, b_deferred_V::Vector{Float64},
                          n_nodes::Int, n_currents::Int)
    fill!(b, 0.0)

    # Copy direct stamps
    n = min(length(b_direct), length(b))
    @inbounds for i in 1:n
        b[i] = b_direct[i]
    end

    # Apply deferred stamps (typed indices for current/charge variables)
    @inbounds for (idx_typed, v) in zip(b_deferred_I, b_deferred_V)
        # Resolve typed index to actual position
        idx = if idx_typed isa NodeIndex
            idx_typed.idx
        elseif idx_typed isa CurrentIndex
            n_nodes + idx_typed.k
        elseif idx_typed isa ChargeIndex
            n_nodes + n_currents + idx_typed.k
        else
            0  # GroundIndex - skip
        end
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
    # Build circuit at current operating point with explicit time
    ctx = if pc.supports_ctx_reuse
        # ZERO-ALLOCATION PATH: Pass stored context to avoid allocating a new one
        # The builder will call reset_for_restamping!(ctx) and restamp into it
        pc.builder(pc.params, pc.spec, real_time(t); x=u, ctx=pc.ctx)
    else
        # Fallback for hand-written builders without ctx support
        pc.builder(pc.params, pc.spec, real_time(t); x=u)
    end

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
    update_b_vector!(pc.b, pc.b_direct, pc.b_deferred_I, pc.b_deferred_V, pc.n_nodes, pc.n_currents)

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

#==============================================================================#
# EvalWorkspace Fast Evaluation (Zero-Allocation Path)
#==============================================================================#

"""
    reset_workspace!(ws::EvalWorkspace)

Reset workspace b vector to zero.
Called at the start of each evaluation before re-stamping.

Note: In value-only mode, reset_value_only!(vctx) handles its own reset.
Sparse matrix values are zeroed directly in fast_rebuild!.
"""
function reset_workspace!(ws::EvalWorkspace{T}) where T
    fill!(ws.b, zero(T))
    return nothing
end

"""
    fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)

Zero-allocation rebuild of circuit at operating point u and time t.

This version uses the EvalWorkspace which preallocates all storage.
Time is passed explicitly as a parameter (not stored in workspace).

# Builder API
Builders must accept time as explicit parameter:
    builder(params, spec, t::Real; x=ZERO_VECTOR) -> MNAContext

The spec contains temperature, mode, and tolerances (immutable during simulation).
Time is passed separately since it changes every iteration.

# Zero-Allocation Path
When `supports_value_only_mode` is true, uses ValueOnlyContext which writes
directly to pre-sized arrays without any push! operations. Values are then
stamped directly from vctx to sparse matrices (no intermediate copy).
"""
function fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)
    cs = ws.structure
    time = real_time(t)  # Local variable, not stored in struct

    if ws.supports_value_only_mode
        # TRUE ZERO-ALLOCATION PATH: Use ValueOnlyContext
        # ValueOnlyContext writes directly to pre-sized arrays without push!
        # Call positional version to avoid keyword argument allocation overhead
        vctx = ws.vctx
        reset_value_only!(vctx)
        cs.builder(cs.params, cs.spec, time, u, vctx)

        # Stamp directly from vctx to sparse matrices (no intermediate ws.G_V copy)
        n_G = cs.G_n_coo
        n_C = cs.C_n_coo

        # Update sparse matrices directly from vctx values
        # This eliminates the intermediate copy: vctx.G_V → ws.G_V → sparse
        _update_sparse_direct!(cs.G, vctx.G_V, cs.G_coo_to_nz, n_G)
        _update_sparse_direct!(cs.C, vctx.C_V, cs.C_coo_to_nz, n_C)

        # b vector: zero first, then copy direct stamps and apply deferred
        fill!(ws.b, zero(eltype(ws.b)))

        # Copy direct stamps from vctx.b (node stamps only, length = n_nodes)
        n_b = length(vctx.b)
        @inbounds @simd for i in 1:n_b
            ws.b[i] = vctx.b[i]
        end

        # Apply deferred b stamps using pre-resolved indices from CompiledStructure
        n_deferred = cs.n_b_deferred
        @inbounds for k in 1:n_deferred
            idx = cs.b_deferred_resolved[k]
            if idx > 0
                ws.b[idx] += vctx.b_V[k]
            end
        end

    elseif ws.supports_ctx_reuse
        # REDUCED-ALLOCATION PATH: Reuse MNAContext but still uses push!
        ctx = cs.builder(cs.params, cs.spec, time; x=u, ctx=ws.ctx)
        _copy_ctx_to_workspace!(ws, ctx, cs)
    else
        # FALLBACK PATH: Allocates new MNAContext each iteration
        ctx = cs.builder(cs.params, cs.spec, time; x=u)
        _copy_ctx_to_workspace!(ws, ctx, cs)
    end

    return nothing
end

"""
    _update_sparse_direct!(S, V, mapping, n_entries)

Update sparse matrix values directly from source array.
Zeros the matrix first, then accumulates values using precomputed mapping.

This is the direct path: source → sparse (no intermediate array).
"""
@inline function _update_sparse_direct!(S::SparseMatrixCSC, V::AbstractVector,
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
    _copy_ctx_to_workspace!(ws::EvalWorkspace, ctx::MNAContext, cs::CompiledStructure)

Copy values from MNAContext to workspace and update sparse matrices.
Internal helper for the non-value-only rebuild paths (fallback).
"""
function _copy_ctx_to_workspace!(ws::EvalWorkspace, ctx::MNAContext, cs::CompiledStructure)
    # Structure must be constant
    @assert ctx.n_nodes == cs.n_nodes && ctx.n_currents == cs.n_currents (
        "Circuit structure changed: expected $(cs.n_nodes) nodes + $(cs.n_currents) currents, " *
        "got $(ctx.n_nodes) + $(ctx.n_currents)")

    n_G = length(ctx.G_I)
    n_C = length(ctx.C_I)

    @assert n_G == cs.G_n_coo "G matrix COO length changed: expected $(cs.G_n_coo), got $n_G"
    @assert n_C == cs.C_n_coo "C matrix COO length changed: expected $(cs.C_n_coo), got $n_C"

    # Update sparse matrices directly from ctx values (no intermediate copy)
    _update_sparse_direct!(cs.G, ctx.G_V, cs.G_coo_to_nz, n_G)
    _update_sparse_direct!(cs.C, ctx.C_V, cs.C_coo_to_nz, n_C)

    # Update b vector: zero, copy direct stamps, then apply deferred stamps
    fill!(ws.b, zero(eltype(ws.b)))

    # Copy direct stamps from ctx.b
    n_b = min(length(ctx.b), length(ws.b))
    @inbounds for i in 1:n_b
        ws.b[i] = ctx.b[i]
    end

    # Apply deferred b stamps (typed indices for current/charge variables)
    @inbounds for (idx_typed, v) in zip(ctx.b_I, ctx.b_V)
        idx = if idx_typed isa NodeIndex
            idx_typed.idx
        elseif idx_typed isa CurrentIndex
            cs.n_nodes + idx_typed.k
        elseif idx_typed isa ChargeIndex
            cs.n_nodes + cs.n_currents + idx_typed.k
        else
            0  # GroundIndex - skip
        end
        if 1 <= idx <= length(ws.b)
            ws.b[idx] += v
        end
    end

    return nothing
end

"""
    fast_residual!(resid, du, u, ws::EvalWorkspace, t)

Fast DAE residual evaluation using EvalWorkspace.

Computes: F(du, u, t) = C*du + G*u - b = 0
"""
function fast_residual!(resid::AbstractVector, du::AbstractVector,
                        u::AbstractVector, ws::EvalWorkspace, t::Real)
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # F(du, u) = C*du + G*u - b = 0
    mul!(resid, cs.C, du)
    mul!(resid, cs.G, u, 1.0, 1.0)
    resid .-= ws.b

    return nothing
end

"""
    fast_jacobian!(J, du, u, ws::EvalWorkspace, gamma, t)

Fast DAE Jacobian computation using EvalWorkspace: J = G + gamma*C
"""
function fast_jacobian!(J::AbstractMatrix, du::AbstractVector,
                        u::AbstractVector, ws::EvalWorkspace,
                        gamma::Real, t::Real)
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # J = G + gamma*C
    copyto!(J, cs.G)
    J .+= gamma .* cs.C

    return nothing
end

#==============================================================================#
# PrecompiledCircuit Fast Evaluation (Backwards Compatibility)
#==============================================================================#

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
function build_rc(params, spec; x=ZERO_VECTOR)
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

Correctly handles explicit zeros in the sparse matrix by checking actual values.
"""
function detect_differential_vars_from_C(C::SparseMatrixCSC, n::Int)
    diff_vars = falses(n)
    nzvals = nonzeros(C)

    for j in 1:n
        for k in nzrange(C, j)
            # Only mark as differential if the value is actually nonzero
            if abs(nzvals[k]) > 1e-30
                i = rowvals(C)[k]
                diff_vars[i] = true
            end
        end
    end

    return diff_vars
end

# NOTE: SciML integration (DAEProblem, ODEProblem) for MNACircuitCompiled
# is not needed here since MNACircuit now uses PrecompiledCircuit by default
# in solve.jl. Users should use MNACircuit, not MNACircuitCompiled.
#
# The MNACircuitCompiled type is kept for explicit precompilation when needed.

#==============================================================================#
# Specialized Function Generation for SROA Optimization
#
# For small circuits, we generate a specialized fast_rebuild! function where
# the COO-to-CSC mappings are baked in as SVector literals. This enables:
# - Full SROA (Scalar Replacement of Aggregates) by the compiler
# - The mapping indices become constants in the generated code
# - Loop unrolling for small circuits
#
# The threshold (MAX_SVECTOR_SIZE) balances compile time vs. runtime performance.
#==============================================================================#

export make_specialized_rebuild, SpecializedWorkspace, create_specialized_workspace

"""
Maximum number of COO entries for SVector-based optimization.
Beyond this, compilation time becomes excessive and we fall back to Vector.
"""
const MAX_SVECTOR_SIZE = 64

"""
    SpecializedWorkspace{T,CS,RF}

Workspace with a specialized rebuild function for maximum performance.

The rebuild function has structure mappings baked in as constants,
enabling full SROA and loop unrolling for small circuits.

# Fields
- `structure::CS`: Compiled circuit structure
- `b::Vector{T}`: RHS vector (mutable contents)
- `resid_tmp::Vector{T}`: Working storage
- `vctx::ValueOnlyContext{T}`: Value-only stamping context
- `rebuild!::RF`: Specialized rebuild function
"""
struct SpecializedWorkspace{T,CS<:CompiledStructure,RF}
    structure::CS
    b::Vector{T}
    resid_tmp::Vector{T}
    vctx::ValueOnlyContext{T}
    rebuild!::RF  # Specialized function: (vctx, G_nzval, C_nzval, b, u, t) -> nothing
end

"""
    make_specialized_rebuild(cs::CompiledStructure) -> Function

Generate a specialized rebuild function with structure captured in a closure.

The closure captures:
- builder, params, spec: Circuit definition
- n_G, n_C, n_b, n_deferred: Size constants
- G_mapping, C_mapping, b_resolved: COO-to-CSC mappings as SVectors (for small circuits)

For small circuits, the SVector mappings can be SROA'd by the compiler.
"""
function make_specialized_rebuild(cs::CompiledStructure{F,P,S}) where {F,P,S}
    # Capture circuit definition
    builder = cs.builder
    params = cs.params
    spec = cs.spec

    # Capture size constants
    n_G = cs.G_n_coo
    n_C = cs.C_n_coo
    n_b = cs.n_nodes
    n_deferred = cs.n_b_deferred

    # For small circuits, use SVectors which can be SROA'd
    # For large circuits, use regular vectors
    G_mapping = if n_G <= MAX_SVECTOR_SIZE && n_G > 0
        SVector{n_G,Int}(cs.G_coo_to_nz[1:n_G]...)
    else
        cs.G_coo_to_nz[1:n_G]
    end

    C_mapping = if n_C <= MAX_SVECTOR_SIZE && n_C > 0
        SVector{n_C,Int}(cs.C_coo_to_nz[1:n_C]...)
    else
        cs.C_coo_to_nz[1:n_C]
    end

    b_resolved = if n_deferred <= MAX_SVECTOR_SIZE && n_deferred > 0
        SVector{n_deferred,Int}(cs.b_deferred_resolved[1:n_deferred]...)
    else
        n_deferred > 0 ? cs.b_deferred_resolved[1:n_deferred] : Int[]
    end

    # Create the specialized rebuild closure
    # All structure is captured, enabling compiler optimization
    function specialized_rebuild!(vctx::ValueOnlyContext, G_nzval::Vector{Float64},
                                  C_nzval::Vector{Float64}, b::Vector{Float64},
                                  u::AbstractVector, t::Real)
        # Reset and run builder
        reset_value_only!(vctx)
        # Use keyword args for compatibility, positional for generated builders
        builder(params, spec, real_time(t); x=u, ctx=vctx)

        # Zero sparse matrices
        fill!(G_nzval, 0.0)
        fill!(C_nzval, 0.0)
        fill!(b, 0.0)

        # Stamp G matrix directly from vctx to sparse
        @inbounds for k in 1:n_G
            idx = G_mapping[k]
            if idx > 0
                G_nzval[idx] += vctx.G_V[k]
            end
        end

        # Stamp C matrix directly from vctx to sparse
        @inbounds for k in 1:n_C
            idx = C_mapping[k]
            if idx > 0
                C_nzval[idx] += vctx.C_V[k]
            end
        end

        # Copy direct b stamps
        @inbounds for i in 1:min(length(vctx.b), n_b)
            b[i] = vctx.b[i]
        end

        # Apply deferred b stamps
        @inbounds for k in 1:n_deferred
            idx = b_resolved[k]
            if idx > 0
                b[idx] += vctx.b_V[k]
            end
        end

        return nothing
    end

    return specialized_rebuild!
end

"""
    create_specialized_workspace(cs::CompiledStructure) -> SpecializedWorkspace

Create a workspace with a specialized rebuild function for maximum performance.

This is the most optimized path: the rebuild function has all structure
information baked in as constants, enabling SROA and loop unrolling.
"""
function create_specialized_workspace(cs::CompiledStructure{F,P,S}) where {F,P,S}
    # Create a preallocated MNAContext for structure discovery
    ctx = cs.builder(cs.params, cs.spec, 0.0; x=ZERO_VECTOR)

    # Create ValueOnlyContext for stamping
    vctx = create_value_only_context(ctx)

    # Generate specialized rebuild function
    rebuild! = make_specialized_rebuild(cs)

    SpecializedWorkspace{Float64,typeof(cs),typeof(rebuild!)}(
        cs,
        zeros(Float64, cs.n),    # b vector
        zeros(Float64, cs.n),    # resid_tmp
        vctx,
        rebuild!
    )
end

"""
    fast_rebuild!(ws::SpecializedWorkspace, u::AbstractVector, t::Real)

Ultra-fast rebuild using specialized function with baked-in structure.
"""
function fast_rebuild!(ws::SpecializedWorkspace, u::AbstractVector, t::Real)
    cs = ws.structure
    G_nzval = nonzeros(cs.G)
    C_nzval = nonzeros(cs.C)
    ws.rebuild!(ws.vctx, G_nzval, C_nzval, ws.b, u, t)
    return nothing
end

"""
    fast_residual!(resid, du, u, ws::SpecializedWorkspace, t)

Fast DAE residual evaluation using specialized rebuild.
"""
function fast_residual!(resid::AbstractVector, du::AbstractVector,
                        u::AbstractVector, ws::SpecializedWorkspace, t::Real)
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # F(du, u) = C*du + G*u - b = 0
    mul!(resid, cs.C, du)
    mul!(resid, cs.G, u, 1.0, 1.0)
    resid .-= ws.b

    return nothing
end

"""
    fast_jacobian!(J, du, u, ws::SpecializedWorkspace, gamma, t)

Fast DAE Jacobian computation using specialized rebuild: J = G + gamma*C
"""
function fast_jacobian!(J::AbstractMatrix, du::AbstractVector,
                        u::AbstractVector, ws::SpecializedWorkspace,
                        gamma::Real, t::Real)
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # J = G + gamma*C
    copyto!(J, cs.G)
    J .+= gamma .* cs.C

    return nothing
end

"""
    system_size(ws::SpecializedWorkspace) -> Int
"""
system_size(ws::SpecializedWorkspace) = system_size(ws.structure)

#==============================================================================#
# DirectWorkspace: Zero-Copy Stamping for Large Circuits
#
# Uses DirectStampContext which stamps directly to sparse matrix nzval,
# eliminating ALL intermediate arrays. This is optimal for large circuits
# where SVector optimization doesn't apply.
#
# Data flow:
#   builder stamps → DirectStampContext → G.nzval/C.nzval (single write)
#
# No vctx.G_V, no ws.G_V, no copying between arrays.
#==============================================================================#

export DirectWorkspace, create_direct_workspace

"""
    DirectWorkspace{T,CS}

Workspace for large circuits using zero-copy DirectStampContext.

Stamps go directly to sparse matrix nzval arrays - no intermediate storage.
This is the fastest path for large circuits where SVector optimization
would cause excessive compile time.

# Fields
- `structure::CS`: Compiled circuit structure
- `dctx::DirectStampContext`: Direct stamping context with sparse refs
- `resid_tmp::Vector{T}`: Working storage for residual computation
"""
struct DirectWorkspace{T,CS<:CompiledStructure}
    structure::CS
    dctx::DirectStampContext
    resid_tmp::Vector{T}
end

"""
    create_direct_workspace(cs::CompiledStructure) -> DirectWorkspace

Create a workspace that stamps directly to sparse matrices.

This is optimal for large circuits:
- No intermediate G_V, C_V arrays
- Stamps go straight to sparse nzval
- Deferred b stamps resolved using precomputed mapping
"""
function create_direct_workspace(cs::CompiledStructure{F,P,S}) where {F,P,S}
    # Get initial context for structure info
    ctx = cs.builder(cs.params, cs.spec, 0.0; x=ZERO_VECTOR)

    # Create b vector
    b = zeros(Float64, cs.n)

    # Create DirectStampContext with references to sparse nzval
    dctx = create_direct_stamp_context(
        ctx,
        nonzeros(cs.G),
        nonzeros(cs.C),
        b,
        cs.G_coo_to_nz,
        cs.C_coo_to_nz,
        cs.b_deferred_resolved
    )

    DirectWorkspace{Float64,typeof(cs)}(
        cs,
        dctx,
        zeros(Float64, cs.n)
    )
end

"""
    fast_rebuild!(ws::DirectWorkspace, u::AbstractVector, t::Real)

Zero-copy rebuild using DirectStampContext.

Stamps go directly to sparse matrix nzval - no intermediate arrays.
"""
function fast_rebuild!(ws::DirectWorkspace, u::AbstractVector, t::Real)
    cs = ws.structure
    dctx = ws.dctx

    # Reset counters and zero matrices
    reset_direct_stamp!(dctx)

    # Builder stamps directly to sparse via DirectStampContext
    cs.builder(cs.params, cs.spec, real_time(t); x=u, ctx=dctx)

    # Apply deferred b stamps
    n_deferred = cs.n_b_deferred
    @inbounds for k in 1:n_deferred
        idx = dctx.b_resolved[k]
        if idx > 0
            dctx.b[idx] += dctx.b_V[k]
        end
    end

    return nothing
end

"""
    fast_residual!(resid, du, u, ws::DirectWorkspace, t)

Fast DAE residual evaluation using direct stamping.
"""
function fast_residual!(resid::AbstractVector, du::AbstractVector,
                        u::AbstractVector, ws::DirectWorkspace, t::Real)
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # F(du, u) = C*du + G*u - b = 0
    mul!(resid, cs.C, du)
    mul!(resid, cs.G, u, 1.0, 1.0)
    resid .-= ws.dctx.b

    return nothing
end

"""
    fast_jacobian!(J, du, u, ws::DirectWorkspace, gamma, t)

Fast DAE Jacobian computation using direct stamping: J = G + gamma*C
"""
function fast_jacobian!(J::AbstractMatrix, du::AbstractVector,
                        u::AbstractVector, ws::DirectWorkspace,
                        gamma::Real, t::Real)
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # J = G + gamma*C
    copyto!(J, cs.G)
    J .+= gamma .* cs.C

    return nothing
end

"""
    system_size(ws::DirectWorkspace) -> Int
"""
system_size(ws::DirectWorkspace) = system_size(ws.structure)

"""
    b_vector(ws::DirectWorkspace) -> Vector{Float64}

Get the b vector from a DirectWorkspace.
"""
b_vector(ws::DirectWorkspace) = ws.dctx.b
