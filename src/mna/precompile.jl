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

# Extract real value from ForwardDiff.Dual (for tgrad compatibility)
# Needed for time-dependent sources with Rosenbrock solvers
real_time(t::Real) = Float64(t)
real_time(t::Dual) = Float64(value(t))

export CompiledStructure, EvalWorkspace, compile_structure, create_workspace
export fast_residual!, fast_jacobian!

#==============================================================================#
# Sparse Matrix Utilities for Zero-Allocation Jacobian
#==============================================================================#

"""
    _pad_to_pattern(M::SparseMatrixCSC, pattern::SparseMatrixCSC) -> SparseMatrixCSC

Expand sparse matrix M to have the sparsity pattern of `pattern`.
Entries in `pattern` not in `M` are zero.
"""
function _pad_to_pattern(M::SparseMatrixCSC{Tv}, pattern::SparseMatrixCSC) where Tv
    result = similar(pattern, Tv)
    fill!(nonzeros(result), zero(Tv))
    I, J, V = findnz(M)
    result[CartesianIndex.(I, J)] .= V
    return result
end

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

    # Sparse matrices with UNIFIED sparsity pattern for zero-allocation Jacobian computation
    # Both G and C are stored with the same sparsity pattern (|G| + |C|), padded with zeros
    # where necessary. This enables J = G + gamma*C via direct nzval operations:
    #   nonzeros(J) .= nonzeros(G) .+ gamma .* nonzeros(C)
    # without any intermediate sparse matrix allocation.
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
# EvalWorkspace: Zero-Copy Stamping Workspace
#
# Uses DirectStampContext which stamps directly to sparse matrix nzval,
# eliminating ALL intermediate arrays. Optimal for all circuit sizes.
#
# Data flow:
#   builder stamps → DirectStampContext → G.nzval/C.nzval (single write)
#
# No vctx.G_V, no intermediate copying between arrays.
#==============================================================================#

"""
    EvalWorkspace{T,CS}

Immutable evaluation workspace with zero-copy DirectStampContext.

Stamps go directly to sparse matrix nzval arrays - no intermediate storage.
This is the fastest path for all circuits.

# Fields
- `structure::CS`: Compiled circuit structure
- `dctx::DirectStampContext`: Direct stamping context with sparse refs
- `resid_tmp::Vector{T}`: Working storage for residual computation
"""
struct EvalWorkspace{T,CS<:CompiledStructure}
    structure::CS
    dctx::DirectStampContext
    resid_tmp::Vector{T}
end

"""
    create_workspace(cs::CompiledStructure{F,P,S}; ctx=nothing) -> EvalWorkspace

Create a workspace that stamps directly to sparse matrices.

This is the single recommended API - it works optimally for all circuit sizes:
- No intermediate G_V, C_V arrays
- Stamps go straight to sparse nzval
- Deferred b stamps resolved using precomputed mapping

If `ctx` is provided, it will be used for the DirectStampContext (including its
detection cache). This is important for voltage-dependent capacitor detection:
if ZERO_VECTOR is used to build the context, reactive branches like ddt(Q(V))
may return scalars instead of Duals, causing incorrect detection cache.
"""
function create_workspace(cs::CompiledStructure{F,P,S}; ctx::Union{MNAContext, Nothing}=nothing) where {F,P,S}
    # Use provided context or rebuild (fallback for backward compatibility)
    if ctx === nothing
        ctx = cs.builder(cs.params, cs.spec, 0.0; x=ZERO_VECTOR)
    end

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

    EvalWorkspace{Float64,typeof(cs)}(
        cs,
        dctx,
        zeros(Float64, cs.n)
    )
end

"""
    system_size(ws::EvalWorkspace) -> Int

Return the system size from the workspace's structure.
"""
system_size(ws::EvalWorkspace) = system_size(ws.structure)


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
- `ctx`: Optional pre-built context with detection cache. If provided,
  this context is used instead of building fresh. This ensures consistent
  detection results across code paths.

# Returns
An immutable `CompiledStructure` that can be shared across threads.
Use `create_workspace(cs)` to create a mutable workspace for evaluation.
"""
function compile_structure(builder::F, params::P, spec::S; ctx::Union{MNAContext, Nothing}=nothing) where {F,P,S}
    # Use provided context or build fresh
    if ctx === nothing
        ctx0 = builder(params, spec, 0.0; x=ZERO_VECTOR)
    else
        ctx0 = ctx
    end
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

    G_raw = sparse(G_I_resolved, G_J_resolved, ctx0.G_V, n, n)
    C_raw = sparse(C_I_resolved, C_J_resolved, ctx0.C_V, n, n)

    # Create UNIFIED sparsity pattern from COO indices directly.
    # Using ones() ensures no positions are dropped due to value cancellation.
    # This ensures J = G + gamma*C can be computed without allocation.
    jac_pattern = sparse(
        vcat(G_I_resolved, C_I_resolved),
        vcat(G_J_resolved, C_J_resolved),
        ones(length(G_I_resolved) + length(C_I_resolved)), n, n)

    # Pad G and C to match the unified pattern (same colptr, rowvals)
    G = _pad_to_pattern(G_raw, jac_pattern)
    C = _pad_to_pattern(C_raw, jac_pattern)

    # Create COO→CSC mappings (now mapping to the padded matrices)
    G_coo_to_nz = compute_coo_to_nz_mapping(G_I_resolved, G_J_resolved, G)
    C_coo_to_nz = compute_coo_to_nz_mapping(C_I_resolved, C_J_resolved, C)

    n_G = length(ctx0.G_I)
    n_C = length(ctx0.C_I)

    # Resolve deferred b stamp indices (CurrentIndex/ChargeIndex → actual b vector positions)
    # These are fixed after the first build and can be reused in value-only mode
    n_b_deferred = length(ctx0.b_I)
    b_deferred_resolved = Vector{Int}(undef, n_b_deferred)
    for k in 1:n_b_deferred
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


#==============================================================================#
# EvalWorkspace Fast Evaluation (Zero-Allocation Path)
#==============================================================================#

"""
    fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)

Zero-copy rebuild using DirectStampContext.

Stamps go directly to sparse matrix nzval - no intermediate arrays.
"""
function fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)
    fast_rebuild!(ws, ws.structure, u, t)
end

"""
    fast_rebuild!(ws::EvalWorkspace, cs::CompiledStructure, u::AbstractVector, t::Real)

Zero-copy rebuild using DirectStampContext with an explicit CompiledStructure.

This variant allows passing a different CompiledStructure (e.g., with modified spec
for dcop mode) while still using the same workspace for stamping.
"""
function fast_rebuild!(ws::EvalWorkspace, cs::CompiledStructure, u::AbstractVector, t::Real)
    dctx = ws.dctx

    # Reset counters and zero matrices
    reset_direct_stamp!(dctx)

    # Builder stamps directly to sparse via DirectStampContext
    cs.builder(cs.params, cs.spec, real_time(t); x=u, ctx=dctx)

    # Apply deferred b stamps
    n_deferred = cs.n_b_deferred
    for k in 1:n_deferred
        idx = dctx.b_resolved[k]
        if idx > 0
            dctx.b[idx] += dctx.b_V[k]
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
    resid .-= ws.dctx.b

    return nothing
end

"""
    fast_jacobian!(J, du, u, ws::EvalWorkspace, gamma, t)

Fast DAE Jacobian computation using EvalWorkspace: J = G + gamma*C

For sparse J: Uses zero-allocation nzval operations since G and C have unified sparsity.
For dense J: Falls back to broadcast operations (IDA uses dense Jacobian by default).
"""
function fast_jacobian!(J::SparseMatrixCSC, du::AbstractVector,
                        u::AbstractVector, ws::EvalWorkspace,
                        gamma::Real, t::Real)
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # J = G + gamma*C via direct nzval operations (zero allocation)
    # This works because G and C have been padded to the same sparsity pattern
    J_nz = nonzeros(J)
    G_nz = nonzeros(cs.G)
    C_nz = nonzeros(cs.C)
    @inbounds for i in eachindex(J_nz, G_nz, C_nz)
        J_nz[i] = G_nz[i] + gamma * C_nz[i]
    end

    return nothing
end

# Fallback for dense matrices (Sundials IDA uses dense Jacobian by default)
function fast_jacobian!(J::AbstractMatrix, du::AbstractVector,
                        u::AbstractVector, ws::EvalWorkspace,
                        gamma::Real, t::Real)
    fast_rebuild!(ws, u, t)
    cs = ws.structure

    # J = G + gamma*C via dense matrix operations
    # This path allocates but is used for dense solvers
    copyto!(J, Matrix(cs.G))
    J .+= gamma .* Matrix(cs.C)

    return nothing
end


"""
    b_vector(ws::EvalWorkspace) -> Vector{Float64}

Get the b vector from an EvalWorkspace.
"""
b_vector(ws::EvalWorkspace) = ws.dctx.b
