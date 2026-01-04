#==============================================================================#
# DirectStampContext: Zero-Copy Stamping to Sparse Matrices
#
# This module provides optimized circuit evaluation by stamping directly
# to sparse matrix nzval arrays using precomputed COO-to-nzval mapping.
#
# Key insight: COO indices (G_I, G_J, C_I, C_J) are CONSTANT after first build.
# Only values (G_V, C_V, b) change. DirectStampContext exploits this by:
#   - Storing node_to_idx reference for fast lookups (no new allocations)
#   - Tracking write positions (counter, not push!)
#   - Writing DIRECTLY to sparse matrix nzval via precomputed mapping
#
# Data flow (single write, no intermediate copy):
#   stamp → sparse.nzval[precomputed_idx]
#==============================================================================#

export DirectStampContext, create_direct_stamp_context, reset_direct_stamp!

"""
    DirectStampContext

Zero-copy context that stamps directly to sparse matrix nzval arrays.

No intermediate G_V, C_V arrays - values go straight to sparse storage.
Uses precomputed COO-to-nzval mapping for O(1) position lookup.

# Fields
- `node_to_idx::Dict{Symbol,Int}`: Node name to matrix index
- `n_nodes::Int`, `n_currents::Int`: Circuit dimensions
- `G_nzval::Vector{Float64}`: Direct reference to G sparse matrix nonzeros
- `C_nzval::Vector{Float64}`: Direct reference to C sparse matrix nonzeros
- `G_mapping::Vector{Int}`: COO position → G.nzval index
- `C_mapping::Vector{Int}`: COO position → C.nzval index
- `b::Vector{Float64}`: RHS vector (direct reference)
- `b_V::Vector{Float64}`: Deferred b stamp values
- `b_resolved::Vector{Int}`: Deferred stamp positions in b
- `G_pos::Int`, `C_pos::Int`, etc.: Mutable counters

This eliminates all intermediate copies:
  stamp → sparse.nzval[idx] directly
"""
mutable struct DirectStampContext
    # Node lookups (immutable reference)
    node_to_idx::Dict{Symbol,Int}
    n_nodes::Int
    n_currents::Int

    # Direct references to sparse matrix storage (these are the actual nzval arrays)
    G_nzval::Vector{Float64}
    C_nzval::Vector{Float64}

    # Precomputed COO-to-nzval mapping
    G_mapping::Vector{Int}
    C_mapping::Vector{Int}

    # RHS vector and deferred stamps
    b::Vector{Float64}
    b_V::Vector{Float64}
    b_resolved::Vector{Int}

    # Mutable counters (reset each iteration)
    G_pos::Int
    C_pos::Int
    b_deferred_pos::Int
    current_pos::Int
    charge_pos::Int

    # Expected sizes for bounds checking
    n_G::Int
    n_C::Int
    n_b_deferred::Int

    # Charge detection cache
    charge_is_vdep::Vector{Bool}
    charge_detection_pos::Int
end

"""
    create_direct_stamp_context(ctx::MNAContext, ...) -> DirectStampContext

Create a DirectStampContext from a completed MNAContext.

The context holds direct references to sparse matrix nzval arrays,
enabling zero-copy stamping.
"""
function create_direct_stamp_context(ctx::MNAContext, G_nzval::Vector{Float64},
                                     C_nzval::Vector{Float64}, b::Vector{Float64},
                                     G_mapping::Vector{Int}, C_mapping::Vector{Int},
                                     b_resolved::Vector{Int})
    n_G = length(ctx.G_V)
    n_C = length(ctx.C_V)
    n_b_deferred = length(ctx.b_V)

    DirectStampContext(
        ctx.node_to_idx,
        ctx.n_nodes,
        ctx.n_currents,
        G_nzval,
        C_nzval,
        G_mapping,
        C_mapping,
        b,
        Vector{Float64}(undef, n_b_deferred),
        b_resolved,
        1, 1, 1, 1, 1,  # positions
        n_G, n_C, n_b_deferred,
        copy(ctx.charge_is_vdep),
        1
    )
end

"""
    reset_direct_stamp!(dctx::DirectStampContext)

Reset counters and zero sparse matrix values for a new iteration.
"""
@inline function reset_direct_stamp!(dctx::DirectStampContext)
    # Reset counters
    dctx.G_pos = 1
    dctx.C_pos = 1
    dctx.b_deferred_pos = 1
    dctx.current_pos = 1
    dctx.charge_pos = 1
    dctx.charge_detection_pos = 1

    # Zero sparse matrices and b vector
    fill!(dctx.G_nzval, 0.0)
    fill!(dctx.C_nzval, 0.0)
    fill!(dctx.b, 0.0)

    return nothing
end

#==============================================================================#
# DirectStampContext Stamp Methods - Direct to Sparse
#==============================================================================#

@inline function get_node!(dctx::DirectStampContext, name::Symbol)::Int
    (name === :gnd || name === Symbol("0") || name === Symbol("gnd!")) && return 0
    @inbounds return dctx.node_to_idx[name]
end

@inline get_node!(dctx::DirectStampContext, name::String) = get_node!(dctx, Symbol(name))
@inline get_node!(dctx::DirectStampContext, idx::Int) = idx

@inline function alloc_internal_node!(dctx::DirectStampContext, name::Symbol)::Int
    @inbounds return dctx.node_to_idx[name]
end

@inline alloc_internal_node!(dctx::DirectStampContext, name::String) = alloc_internal_node!(dctx, Symbol(name))

@inline function alloc_current!(dctx::DirectStampContext, name::Symbol)::CurrentIndex
    pos = dctx.current_pos
    dctx.current_pos = pos + 1
    return CurrentIndex(pos)
end

@inline alloc_current!(dctx::DirectStampContext, name::String) = alloc_current!(dctx, Symbol(name))

@inline function alloc_charge!(dctx::DirectStampContext, name::Symbol, p::Int, n::Int)::ChargeIndex
    pos = dctx.charge_pos
    dctx.charge_pos = pos + 1
    return ChargeIndex(pos)
end

@inline alloc_charge!(dctx::DirectStampContext, name::String, p::Int, n::Int) = alloc_charge!(dctx, Symbol(name), p, n)

"""
    stamp_G!(dctx::DirectStampContext, i, j, val)

Stamp G matrix value DIRECTLY to sparse nzval using precomputed mapping.
No intermediate array - single memory write.
"""
@inline function stamp_G!(dctx::DirectStampContext, i, j, val)
    iszero(i) && return nothing
    iszero(j) && return nothing

    pos = dctx.G_pos
    dctx.G_pos = pos + 1

    # Direct write to sparse matrix nzval
    nz_idx = @inbounds dctx.G_mapping[pos]
    if nz_idx > 0
        v = extract_value(val)
        @inbounds dctx.G_nzval[nz_idx] += v
    end

    return nothing
end

"""
    stamp_C!(dctx::DirectStampContext, i, j, val)

Stamp C matrix value DIRECTLY to sparse nzval using precomputed mapping.
"""
@inline function stamp_C!(dctx::DirectStampContext, i, j, val)
    iszero(i) && return nothing
    iszero(j) && return nothing

    pos = dctx.C_pos
    dctx.C_pos = pos + 1

    nz_idx = @inbounds dctx.C_mapping[pos]
    if nz_idx > 0
        v = extract_value(val)
        @inbounds dctx.C_nzval[nz_idx] += v
    end

    return nothing
end

"""
    stamp_b!(dctx::DirectStampContext, i, val)

Stamp b vector value. Direct for nodes, deferred for currents/charges.
"""
@inline function stamp_b!(dctx::DirectStampContext, i, val)
    iszero(i) && return nothing
    v = extract_value(val)
    typed = _to_typed(i)

    if typed isa CurrentIndex || typed isa ChargeIndex
        # Deferred stamp: store value, apply later with pre-resolved index
        pos = dctx.b_deferred_pos
        @inbounds dctx.b_V[pos] = v
        dctx.b_deferred_pos = pos + 1
    else
        # NodeIndex: direct stamp
        idx = typed.idx
        @inbounds dctx.b[idx] += v
    end
    return nothing
end

"""
    stamp_conductance!(dctx::DirectStampContext, p, n, G)

Stamp conductance pattern for 2-terminal element.
"""
@inline function stamp_conductance!(dctx::DirectStampContext, p::Int, n::Int, G)
    stamp_G!(dctx, p, p,  G)
    stamp_G!(dctx, p, n, -G)
    stamp_G!(dctx, n, p, -G)
    stamp_G!(dctx, n, n,  G)
    return nothing
end

"""
    stamp_capacitance!(dctx::DirectStampContext, p, n, C)

Stamp capacitance pattern for 2-terminal element.
"""
@inline function stamp_capacitance!(dctx::DirectStampContext, p::Int, n::Int, C)
    stamp_C!(dctx, p, p,  C)
    stamp_C!(dctx, p, n, -C)
    stamp_C!(dctx, n, p, -C)
    stamp_C!(dctx, n, n,  C)
    return nothing
end

@inline reset_for_restamping!(dctx::DirectStampContext) = reset_direct_stamp!(dctx)

#==============================================================================#
# AnyMNAContext: Union Type for Context Dispatch
#
# This allows device stamp! methods to work with either MNAContext
# (structure discovery) or DirectStampContext (fast restamping).
#==============================================================================#

"""
    AnyMNAContext

Union type alias for MNAContext or DirectStampContext.

Device stamp! methods use this to accept either context type, allowing
the same code to work for both initial structure discovery (MNAContext)
and zero-copy restamping (DirectStampContext).
"""
const AnyMNAContext = Union{MNAContext, DirectStampContext}
export AnyMNAContext

# Alias for backward compatibility
const AnyStampContext = AnyMNAContext
export AnyStampContext
