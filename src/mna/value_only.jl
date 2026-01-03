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
    n_charges::Int  # Number of charge state variables

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

    # Charge detection results (baked from MNAContext during compilation)
    # This Dict is populated once and never modified, enabling zero-allocation
    # lookups during value-only stamping. Uses the same Symbol keys as MNAContext.
    charge_is_vdep::Dict{Symbol,Bool}
end

"""
    create_value_only_context(ctx::MNAContext) -> ValueOnlyContext

Create a ValueOnlyContext from a completed MNAContext.

The original context's structure is captured, and value arrays are
sized appropriately for subsequent value-only rebuilds.
"""
function create_value_only_context(ctx::MNAContext)
    n_G = length(ctx.G_V)
    n_C = length(ctx.C_V)
    n_b = length(ctx.b)
    n_b_deferred = length(ctx.b_V)

    # Copy the charge detection results from MNAContext
    # This Dict is populated during the first build and never modified,
    # so subsequent lookups don't allocate.
    charge_is_vdep_copy = copy(ctx.charge_is_vdep)

    ValueOnlyContext{Float64}(
        ctx.node_to_idx,
        ctx.n_nodes,
        ctx.n_currents,
        ctx.n_charges,                   # n_charges
        Vector{Float64}(undef, n_G),     # G_V
        Vector{Float64}(undef, n_C),     # C_V
        zeros(Float64, n_b),             # b (needs zeros for accumulation)
        Vector{Float64}(undef, n_b_deferred), # b_V
        1, 1, 1,                         # positions
        1,                               # current_pos
        1,                               # charge_pos
        n_G, n_C, n_b_deferred,
        charge_is_vdep_copy              # baked charge detection results
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
    alloc_internal_node!(vctx::ValueOnlyContext, name::Symbol) -> Int

Fast internal node lookup in value-only mode.
Internal nodes must already exist from the first build.
"""
@inline function alloc_internal_node!(vctx::ValueOnlyContext, name::Symbol)::Int
    # Ground check
    (name === :gnd || name === Symbol("0") || name === Symbol("gnd!")) && return 0
    # Direct lookup - node must exist from first build
    @inbounds return vctx.node_to_idx[name]
end

@inline alloc_internal_node!(vctx::ValueOnlyContext, name::String) = alloc_internal_node!(vctx, Symbol(name))

"""
    alloc_charge!(vctx::ValueOnlyContext, name::Symbol, p::Int, n::Int) -> ChargeIndex

Return the next charge index in value-only mode.
No actual allocation - just returns sequential indices.
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
