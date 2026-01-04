#==============================================================================#
# Value-Only Evaluation Mode
#
# This module provides zero-allocation circuit evaluation by separating
# structure discovery (first build) from value updates (subsequent builds).
#
# Inspired by OpenVAF's OSDI implementation which stores direct pointers to
# matrix entries. In Julia, we use a type-based approach where ValueOnlyContext
# provides specialized stamp! methods that write directly to pre-sized arrays.
#
# Architecture:
#   1. First build: Normal MNAContext stamping discovers structure
#   2. Create ValueOnlyContext: Wraps ctx with pre-sized arrays and write positions
#   3. Subsequent builds: ValueOnlyContext stamp methods write directly (zero alloc)
#
# Key insight: COO indices (G_I, G_J, C_I, C_J) are CONSTANT after first build.
# Only values (G_V, C_V, b) change. ValueOnlyContext exploits this by:
#   - Storing node_to_idx reference for fast lookups (no new allocations)
#   - Tracking write positions (counter, not push!)
#   - Writing directly to pre-sized arrays via @inbounds setindex!
#==============================================================================#

export ValueOnlyContext, create_value_only_context, reset_value_only!

"""
    ValueOnlyContext{T}

Zero-allocation context for value-only circuit rebuilding.

This wraps an MNAContext and provides specialized stamp! methods that
write directly to pre-sized value arrays instead of using push!.

# Type Parameter
- `T`: Element type (typically Float64)

# Usage
```julia
# First build (normal)
ctx = builder(params, spec, 0.0; x=Float64[])

# Create value-only context
vctx = create_value_only_context(ctx)

# Subsequent builds (zero allocation)
reset_value_only!(vctx)
builder(params, spec, t; x=u, ctx=vctx)  # Uses specialized stamp! methods
```
"""
mutable struct ValueOnlyContext{T}
    # Reference to original MNAContext for node lookups
    node_to_idx::Dict{Symbol,Int}
    n_nodes::Int
    n_currents::Int

    # Pre-sized value arrays
    G_V::Vector{T}
    C_V::Vector{T}
    b::Vector{T}
    b_V::Vector{T}  # deferred b stamps

    # Write positions (reset to 1 each iteration)
    G_pos::Int
    C_pos::Int
    b_deferred_pos::Int

    # Current index counter (for alloc_current!)
    current_pos::Int

    # Charge index counter (for alloc_charge!)
    charge_pos::Int

    # Expected sizes for assertions
    n_G::Int
    n_C::Int
    n_b_deferred::Int

    # Charge detection cache (for VA devices with voltage-dependent capacitance)
    # Baked from MNAContext.charge_is_vdep Dict at create_value_only_context time
    # Uses counter-based access to maintain execution order consistency
    charge_is_vdep::Vector{Bool}
    charge_detection_pos::Int
end

"""
    create_value_only_context(ctx::MNAContext) -> ValueOnlyContext

Create a ValueOnlyContext from a completed MNAContext.

The original context's structure is captured, and value arrays are
sized appropriately for subsequent value-only rebuilds.

Charge detection cache is directly copied from MNAContext (both use Vector{Bool}).
Counter-based access during value-only rebuilds matches the original detection order.
"""
function create_value_only_context(ctx::MNAContext)
    n_G = length(ctx.G_V)
    n_C = length(ctx.C_V)
    n_b = length(ctx.b)
    n_b_deferred = length(ctx.b_V)

    # Copy charge detection cache (already a Vector in MNAContext)
    charge_is_vdep_vec = copy(ctx.charge_is_vdep)

    ValueOnlyContext{Float64}(
        ctx.node_to_idx,
        ctx.n_nodes,
        ctx.n_currents,
        Vector{Float64}(undef, n_G),     # G_V
        Vector{Float64}(undef, n_C),     # C_V
        zeros(Float64, n_b),             # b (needs zeros for accumulation)
        Vector{Float64}(undef, n_b_deferred), # b_V
        1, 1, 1,                         # positions (G_pos, C_pos, b_deferred_pos)
        1,                               # current_pos
        1,                               # charge_pos
        n_G, n_C, n_b_deferred,
        charge_is_vdep_vec,              # charge detection cache
        1                                # charge_detection_pos
    )
end

"""
    reset_value_only!(vctx::ValueOnlyContext)

Reset a ValueOnlyContext for a new iteration.

This zeros the b vector and resets all write positions to 1.
Zero allocation operation.
"""
@inline function reset_value_only!(vctx::ValueOnlyContext{T}) where T
    # Reset write positions
    vctx.G_pos = 1
    vctx.C_pos = 1
    vctx.b_deferred_pos = 1
    vctx.current_pos = 1
    vctx.charge_pos = 1
    vctx.charge_detection_pos = 1

    # Zero b vector for accumulation
    fill!(vctx.b, zero(T))

    return nothing
end

#==============================================================================#
# Specialized Stamp Methods for ValueOnlyContext
#
# These mirror the MNAContext stamp methods but write directly to pre-sized
# arrays using tracked positions instead of push!.
#==============================================================================#

"""
    get_node!(vctx::ValueOnlyContext, name::Symbol) -> Int

Fast node lookup in value-only mode.
Nodes must already exist from the first build.
"""
@inline function get_node!(vctx::ValueOnlyContext, name::Symbol)::Int
    # Ground check
    (name === :gnd || name === Symbol("0") || name === Symbol("gnd!")) && return 0
    # Direct lookup - node must exist
    @inbounds return vctx.node_to_idx[name]
end

@inline get_node!(vctx::ValueOnlyContext, name::String) = get_node!(vctx, Symbol(name))
@inline get_node!(vctx::ValueOnlyContext, idx::Int) = idx

"""
    alloc_internal_node!(vctx::ValueOnlyContext, name::Symbol) -> Int

Return the index of an internal node in value-only mode.
The node must already exist from the first build.
"""
@inline function alloc_internal_node!(vctx::ValueOnlyContext, name::Symbol)::Int
    # Internal nodes should already be in node_to_idx from first build
    @inbounds return vctx.node_to_idx[name]
end

@inline alloc_internal_node!(vctx::ValueOnlyContext, name::String) = alloc_internal_node!(vctx, Symbol(name))

"""
    alloc_current!(vctx::ValueOnlyContext, name::Symbol) -> CurrentIndex

Return the next current index in value-only mode.
No actual allocation - just returns sequential indices.
"""
@inline function alloc_current!(vctx::ValueOnlyContext, name::Symbol)::CurrentIndex
    pos = vctx.current_pos
    vctx.current_pos = pos + 1
    return CurrentIndex(pos)
end

@inline alloc_current!(vctx::ValueOnlyContext, name::String) = alloc_current!(vctx, Symbol(name))

"""
    alloc_charge!(vctx::ValueOnlyContext, name::Symbol, p::Int, n::Int) -> ChargeIndex

Return the next charge index in value-only mode.
No actual allocation - just returns sequential indices.
The charge structure is fixed from the first build.
"""
@inline function alloc_charge!(vctx::ValueOnlyContext, name::Symbol, p::Int, n::Int)::ChargeIndex
    pos = vctx.charge_pos
    vctx.charge_pos = pos + 1
    return ChargeIndex(pos)
end

@inline alloc_charge!(vctx::ValueOnlyContext, name::String, p::Int, n::Int) = alloc_charge!(vctx, Symbol(name), p, n)

"""
    stamp_G!(vctx::ValueOnlyContext, i, j, val)

Write G matrix value at current position and advance.
Ground stamps (iszero(i) or iszero(j)) are skipped.
"""
@inline function stamp_G!(vctx::ValueOnlyContext{T}, i, j, val) where T
    iszero(i) && return nothing
    iszero(j) && return nothing
    v = extract_value(val)
    pos = vctx.G_pos
    @inbounds vctx.G_V[pos] = v
    vctx.G_pos = pos + 1
    return nothing
end

"""
    stamp_C!(vctx::ValueOnlyContext, i, j, val)

Write C matrix value at current position and advance.
"""
@inline function stamp_C!(vctx::ValueOnlyContext{T}, i, j, val) where T
    iszero(i) && return nothing
    iszero(j) && return nothing
    v = extract_value(val)
    pos = vctx.C_pos
    @inbounds vctx.C_V[pos] = v
    vctx.C_pos = pos + 1
    return nothing
end

"""
    stamp_b!(vctx::ValueOnlyContext, i, val)

Accumulate b vector value or write deferred stamp.
"""
@inline function stamp_b!(vctx::ValueOnlyContext{T}, i, val) where T
    iszero(i) && return nothing
    v = extract_value(val)
    typed = _to_typed(i)

    if typed isa CurrentIndex || typed isa ChargeIndex
        # Deferred stamp: write to b_V at tracked position
        pos = vctx.b_deferred_pos
        @inbounds vctx.b_V[pos] = v
        vctx.b_deferred_pos = pos + 1
    else
        # NodeIndex: accumulate directly
        idx = typed.idx
        @inbounds vctx.b[idx] += v
    end
    return nothing
end

#==============================================================================#
# Helper Stamp Methods (2-terminal patterns)
#==============================================================================#

"""
    stamp_conductance!(vctx::ValueOnlyContext, p, n, G)

Stamp conductance pattern for 2-terminal element.
"""
@inline function stamp_conductance!(vctx::ValueOnlyContext, p::Int, n::Int, G)
    stamp_G!(vctx, p, p,  G)
    stamp_G!(vctx, p, n, -G)
    stamp_G!(vctx, n, p, -G)
    stamp_G!(vctx, n, n,  G)
    return nothing
end

"""
    stamp_capacitance!(vctx::ValueOnlyContext, p, n, C)

Stamp capacitance pattern for 2-terminal element.
"""
@inline function stamp_capacitance!(vctx::ValueOnlyContext, p::Int, n::Int, C)
    stamp_C!(vctx, p, p,  C)
    stamp_C!(vctx, p, n, -C)
    stamp_C!(vctx, n, p, -C)
    stamp_C!(vctx, n, n,  C)
    return nothing
end

# extract_value and _to_typed are defined in context.jl which is included before this file

"""
    reset_for_restamping!(vctx::ValueOnlyContext)

Alias for reset_value_only! - allows generated builder code to work unchanged.
"""
@inline reset_for_restamping!(vctx::ValueOnlyContext) = reset_value_only!(vctx)

#==============================================================================#
# Type Alias for Both Context Types
#
# This allows device stamp! methods to work with either MNAContext or
# ValueOnlyContext using a single method definition.
#==============================================================================#

"""
    AnyMNAContext

Union type alias for MNAContext or ValueOnlyContext.

Device stamp! methods use this to accept either context type, allowing
the same code to work for both initial structure discovery (MNAContext)
and value-only rebuilds (ValueOnlyContext).
"""
const AnyMNAContext = Union{MNAContext, ValueOnlyContext}

export AnyMNAContext

#==============================================================================#
# DirectStampContext: Zero-Copy Stamping to Sparse Matrices
#
# For large circuits, eliminates ALL intermediate arrays by stamping directly
# to sparse matrix nzval using precomputed COO-to-nzval mapping.
#
# Data flow comparison:
#   ValueOnlyContext: stamp → G_V[pos] → copy → G.nzval[nz_idx]  (2 writes)
#   DirectStampContext: stamp → G.nzval[nz_idx]                   (1 write)
#
# This halves memory bandwidth for large circuits where SVector optimization
# doesn't apply.
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

This eliminates the intermediate copy:
  OLD: vctx.G_V[k] → sparse.nzval[idx]
  NEW: sparse.nzval[idx] directly
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
    create_direct_stamp_context(ctx::MNAContext, cs::CompiledStructure) -> DirectStampContext

Create a DirectStampContext from a completed MNAContext and CompiledStructure.

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

# Update AnyMNAContext to include DirectStampContext
const AnyStampContext = Union{MNAContext, ValueOnlyContext, DirectStampContext}
export AnyStampContext
