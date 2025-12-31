#==============================================================================#
# MNA Phase 1: Context and Stamping Primitives
#
# This module provides the core MNA (Modified Nodal Analysis) infrastructure:
# - MNAContext: Tracks nodes, currents, and matrix stamps
# - Stamping primitives: stamp_G!, stamp_C!, stamp_b!
# - Node/current allocation: get_node!, alloc_current!
#==============================================================================#

export MNAContext, MNASystem
export get_node!, alloc_current!, get_current_idx, has_current, resolve_index
export alloc_internal_node!, is_internal_node, n_internal_nodes
export stamp_G!, stamp_C!, stamp_b!
export stamp_conductance!, stamp_capacitance!
export system_size

"""
    MNAContext

Accumulates MNA (Modified Nodal Analysis) matrix stamps during circuit traversal.

The MNA formulation solves:
    G*x + C*dx/dt = b

where:
    x = [V₁, V₂, ..., Vₙ, I₁, I₂, ..., Iₘ]ᵀ
    - Vᵢ are node voltages
    - Iⱼ are current variables (for V-sources, inductors)

Matrix entries are accumulated in COO (coordinate) format for efficient
sparse matrix construction.

# Fields
- `node_names::Vector{Symbol}`: Names of nodes (for debugging/output)
- `node_to_idx::Dict{Symbol,Int}`: Map node names to indices
- `n_nodes::Int`: Number of nodes (excluding ground)
- `internal_node_flags::BitVector`: Flags indicating which nodes are internal (not terminals)
- `current_names::Vector{Symbol}`: Names of current variables
- `n_currents::Int`: Number of current variables
- `G_I, G_J, G_V`: COO format for conductance matrix G
- `C_I, C_J, C_V`: COO format for capacitance matrix C
- `b::Vector{Float64}`: Right-hand side vector (source terms)

# Ground Convention
Ground (node 0) is implicit and not included in the system.
Stamping functions skip entries involving node 0.

# Example
```julia
ctx = MNAContext()
vcc = get_node!(ctx, :vcc)
out = get_node!(ctx, :out)

# Resistor from vcc to out: G = 1/R
G = 1.0 / 1000.0
stamp_G!(ctx, vcc, vcc,  G)
stamp_G!(ctx, vcc, out, -G)
stamp_G!(ctx, out, vcc, -G)
stamp_G!(ctx, out, out,  G)
```
"""
mutable struct MNAContext
    # Node tracking
    node_names::Vector{Symbol}
    node_to_idx::Dict{Symbol,Int}
    n_nodes::Int

    # Internal node tracking (nodes not accessible as terminals)
    internal_node_flags::BitVector

    # Current variables (for V-sources, inductors, VCVS, etc.)
    current_names::Vector{Symbol}
    n_currents::Int

    # COO format for G matrix (conductance/resistive)
    G_I::Vector{Int}
    G_J::Vector{Int}
    G_V::Vector{Float64}

    # COO format for C matrix (capacitance/reactive)
    C_I::Vector{Int}
    C_J::Vector{Int}
    C_V::Vector{Float64}

    # RHS vector (pre-allocated for known size, or extended dynamically)
    b::Vector{Float64}

    # Deferred b vector stamps for negative indices (current variables)
    # These are resolved at assembly time when n_nodes is final
    b_I::Vector{Int}
    b_V::Vector{Float64}

    # Track if system has been finalized
    finalized::Bool
end

"""
    MNAContext()

Create an empty MNA context ready for circuit stamping.
"""
function MNAContext()
    MNAContext(
        Symbol[],           # node_names
        Dict{Symbol,Int}(), # node_to_idx
        0,                  # n_nodes
        BitVector(),        # internal_node_flags
        Symbol[],           # current_names
        0,                  # n_currents
        Int[],              # G_I
        Int[],              # G_J
        Float64[],          # G_V
        Int[],              # C_I
        Int[],              # C_J
        Float64[],          # C_V
        Float64[],          # b
        Int[],              # b_I (deferred b stamps)
        Float64[],          # b_V (deferred b stamps)
        false               # finalized
    )
end

"""
    system_size(ctx::MNAContext) -> Int

Return the total system size (number of unknowns).
"""
@inline system_size(ctx::MNAContext) = ctx.n_nodes + ctx.n_currents

#==============================================================================#
# Node and Current Variable Allocation
#==============================================================================#

"""
    get_node!(ctx::MNAContext, name::Symbol) -> Int

Get or create a node index for the given name.
Returns 0 for :gnd or :0 (ground node).

Ground is implicit in MNA and not included in the system.
All other nodes are numbered 1, 2, 3, ...
"""
function get_node!(ctx::MNAContext, name::Symbol)::Int
    # Ground node is always 0
    (name === :gnd || name === Symbol("0") || name === Symbol("gnd!")) && return 0

    # Check if node already exists
    idx = get(ctx.node_to_idx, name, 0)
    if idx != 0
        return idx
    end

    # Create new node
    ctx.n_nodes += 1
    push!(ctx.node_names, name)
    ctx.node_to_idx[name] = ctx.n_nodes

    # Extend internal_node_flags (default: not internal)
    push!(ctx.internal_node_flags, false)

    # Extend b vector if needed
    if length(ctx.b) < system_size(ctx)
        old_len = length(ctx.b)
        new_len = system_size(ctx)
        resize!(ctx.b, new_len)
        # Zero-initialize all new elements (resize! leaves them uninitialized)
        @inbounds for j in (old_len+1):new_len
            ctx.b[j] = 0.0
        end
    end

    return ctx.n_nodes
end

"""
    get_node!(ctx::MNAContext, name::String) -> Int

Convenience method for string node names.
"""
get_node!(ctx::MNAContext, name::String) = get_node!(ctx, Symbol(name))

"""
    get_node!(ctx::MNAContext, idx::Int) -> Int

Pass through integer indices (for programmatic use).
Ground (0) is passed through unchanged.
"""
get_node!(ctx::MNAContext, idx::Int) = idx

"""
    alloc_current!(ctx::MNAContext, name::Symbol) -> Int

Allocate a new current variable (for voltage sources, inductors, etc.).
Returns the system index (n_nodes + current_number).

Current variables are numbered after all voltage nodes in the system vector:
x = [V₁, V₂, ..., Vₙ, I₁, I₂, ..., Iₘ]

# Example
```julia
# Voltage source needs a current variable
i_idx = alloc_current!(ctx, :I_V1)
# i_idx is negative (deferred index), resolved to n_nodes + k at assembly time
```

Note: Returns a NEGATIVE index representing current variable k.
This allows n_nodes to change (e.g., from internal node allocation)
without invalidating previously stamped indices.
The negative index is translated to the actual index (n_nodes + k)
at assembly time by `resolve_index`.
"""
function alloc_current!(ctx::MNAContext, name::Symbol)::Int
    ctx.n_currents += 1
    push!(ctx.current_names, name)

    # Return negative index representing current variable k
    # This is resolved to n_nodes + k at assembly time
    return -ctx.n_currents
end

alloc_current!(ctx::MNAContext, name::String) = alloc_current!(ctx, Symbol(name))

"""
    resolve_index(ctx::MNAContext, idx::Int) -> Int

Resolve an index that may be negative (current variable) to an actual system index.
- Positive indices (node indices) are returned unchanged
- Zero (ground) is returned unchanged
- Negative indices (-k) are translated to n_nodes + k (current variable k)

This allows current variable indices to be stamped before all nodes are known,
and resolved at assembly time when n_nodes is final.
"""
@inline function resolve_index(ctx::MNAContext, idx::Int)::Int
    idx >= 0 && return idx  # Node index or ground
    return ctx.n_nodes - idx  # -k → n_nodes + k
end

"""
    get_current_idx(ctx::MNAContext, name::Symbol) -> Int

Look up a current variable index by name.
Returns a negative index (-k) representing current variable k.
Use `resolve_index` to get the actual system index.
Throws an error if the current variable doesn't exist.

# Example
```julia
# After stamping a voltage source V1
i_idx = get_current_idx(ctx, :I_V1)  # Returns -1
actual_idx = resolve_index(ctx, i_idx)  # Returns n_nodes + 1
```
"""
function get_current_idx(ctx::MNAContext, name::Symbol)::Int
    idx = findfirst(==(name), ctx.current_names)
    idx === nothing && error("Current variable $name not found in MNA context")
    return -idx  # Return negative index
end

get_current_idx(ctx::MNAContext, name::String) = get_current_idx(ctx, Symbol(name))

"""
    has_current(ctx::MNAContext, name::Symbol) -> Bool

Check if a current variable with the given name exists.
"""
function has_current(ctx::MNAContext, name::Symbol)::Bool
    return name in ctx.current_names
end

#==============================================================================#
# Internal Node Allocation
#==============================================================================#

"""
    alloc_internal_node!(ctx::MNAContext, name::Symbol) -> Int

Allocate an internal node (not accessible as a terminal).

Internal nodes are used within device models for nodes that are not
externally accessible. For example, a diode with series resistance has
an internal node between the resistor and the junction.

Internal nodes participate in KCL equations but are not visible outside
the device. They are marked in the context for debugging and analysis.

# Arguments
- `ctx`: MNA context to allocate in
- `name`: Unique name for the internal node (typically instance-qualified)

# Returns
The node index for use in stamping operations.

# Example
```julia
# In a diode with series resistance:
function stamp!(dev::DiodeRs, ctx::MNAContext, a::Int, c::Int; instance_name=:D1, ...)
    # Allocate internal node between Rs and junction
    a_int = alloc_internal_node!(ctx, Symbol(instance_name, ".a_int"))

    # Stamp series resistance: a to a_int
    stamp_conductance!(ctx, a, a_int, 1/dev.Rs)

    # Stamp junction: a_int to c
    # ...
end
```

# Notes
- Internal nodes are algebraic in transient analysis (no capacitance to ground)
- Each device instance should use unique internal node names
- Use `is_internal_node(ctx, idx)` to query if a node is internal
"""
function alloc_internal_node!(ctx::MNAContext, name::Symbol)::Int
    # Check if node already exists (shouldn't happen with proper naming)
    idx = get(ctx.node_to_idx, name, 0)
    if idx != 0
        # Node exists - mark as internal if not already
        ctx.internal_node_flags[idx] = true
        return idx
    end

    # Create new node (via get_node!)
    idx = get_node!(ctx, name)

    # Mark as internal
    ctx.internal_node_flags[idx] = true

    return idx
end

alloc_internal_node!(ctx::MNAContext, name::String) = alloc_internal_node!(ctx, Symbol(name))

"""
    is_internal_node(ctx::MNAContext, idx::Int) -> Bool

Check if a node at the given index is an internal node.

Returns `false` for ground (idx=0) or invalid indices.
"""
function is_internal_node(ctx::MNAContext, idx::Int)::Bool
    idx <= 0 && return false
    idx > length(ctx.internal_node_flags) && return false
    return ctx.internal_node_flags[idx]
end

"""
    n_internal_nodes(ctx::MNAContext) -> Int

Return the number of internal nodes in the context.
"""
function n_internal_nodes(ctx::MNAContext)::Int
    return count(ctx.internal_node_flags)
end

#==============================================================================#
# Stamping Primitives
#==============================================================================#

"""
    extract_value(x)

Extract the underlying Float64 value from a potentially nested Dual.
This is needed when ForwardDiff Duals are nested (e.g., ContributionTag wrapping port duals).
"""
@inline extract_value(x::Float64) = x
@inline extract_value(x::Real) = Float64(x)
@inline function extract_value(x::ForwardDiff.Dual)
    extract_value(ForwardDiff.value(x))
end

"""
    stamp_G!(ctx::MNAContext, i::Int, j::Int, val)

Stamp a value into the G (conductance) matrix at position (i, j).
Entries involving ground (i=0 or j=0) are skipped.

The G matrix represents resistive (algebraic) relationships:
- Resistor: stamps ±1/R pattern
- Voltage source constraint equations
- Current source KCL contributions

# Sign Convention
MNA uses "current leaving node is positive" convention.
For a resistor between nodes p and n:
```julia
G = 1/R
stamp_G!(ctx, p, p,  G)  # Current leaving p due to Vp
stamp_G!(ctx, p, n, -G)  # Current leaving p due to Vn
stamp_G!(ctx, n, p, -G)  # Current leaving n due to Vp
stamp_G!(ctx, n, n,  G)  # Current leaving n due to Vn
```
"""
@inline function stamp_G!(ctx::MNAContext, i::Int, j::Int, val)
    (i == 0 || j == 0) && return nothing
    v = extract_value(val)
    # Note: We do NOT skip zero entries here. For precompilation to work,
    # the COO structure must be consistent regardless of operating point.
    # Zero values are handled correctly by sparse matrix assembly.
    push!(ctx.G_I, i)
    push!(ctx.G_J, j)
    push!(ctx.G_V, v)
    return nothing
end

"""
    stamp_C!(ctx::MNAContext, i::Int, j::Int, val)

Stamp a value into the C (capacitance) matrix at position (i, j).
Entries involving ground (i=0 or j=0) are skipped.

The C matrix represents reactive (differential) relationships:
- Capacitor: stamps ±C pattern
- Inductor: stamps -L in current equation

For the ODE formulation: C * dx/dt + G * x = b
"""
@inline function stamp_C!(ctx::MNAContext, i::Int, j::Int, val)
    (i == 0 || j == 0) && return nothing
    v = extract_value(val)
    # Note: For voltage-dependent capacitors, we don't skip zeros because
    # the COO structure must be consistent regardless of operating point.
    # Zero values at some operating points are fine - they'll be in the matrix.
    # However, devices WITHOUT ddt() should not call stamp_C! at all; this is
    # handled by stamp_contribution! which skips C stamps for pure resistive devices.
    push!(ctx.C_I, i)
    push!(ctx.C_J, j)
    push!(ctx.C_V, v)
    return nothing
end

"""
    stamp_b!(ctx::MNAContext, i::Int, val)

Add a value to the RHS vector b at position i.
Entries at ground (i=0) are skipped.

The b vector contains source terms:
- Independent current sources: stamp into KCL equations
- Independent voltage sources: stamp voltage value into constraint equation
"""
@inline function stamp_b!(ctx::MNAContext, i::Int, val)
    i == 0 && return nothing
    v = extract_value(val)
    # Note: We do NOT skip zero entries here for consistency with stamp_G!/stamp_C!.
    # This ensures the structure of deferred stamps is consistent.

    # Resolve negative index (current variable) to actual index
    # Note: For b vector, we need to resolve at assembly time since n_nodes may change.
    # Instead of storing directly, we'll store the stamp and resolve later.
    # For now, handle negative indices by deferring to assembly time.
    if i < 0
        # Store in deferred b stamps (need to add this)
        push!(ctx.b_I, i)
        push!(ctx.b_V, v)
        return nothing
    end

    # Extend b if needed
    if i > length(ctx.b)
        old_len = length(ctx.b)
        resize!(ctx.b, i)
        # Zero-initialize all new elements (resize! leaves them uninitialized)
        @inbounds for j in (old_len+1):i
            ctx.b[j] = 0.0
        end
    end

    ctx.b[i] += v
    return nothing
end

#==============================================================================#
# Conductance Stamping Helpers (2-terminal pattern)
#==============================================================================#

"""
    stamp_conductance!(ctx::MNAContext, p::Int, n::Int, G)

Stamp a conductance G between nodes p and n.
This is the standard 2-terminal resistive element pattern.

Equivalent to manually stamping:
```julia
stamp_G!(ctx, p, p,  G)
stamp_G!(ctx, p, n, -G)
stamp_G!(ctx, n, p, -G)
stamp_G!(ctx, n, n,  G)
```
"""
@inline function stamp_conductance!(ctx::MNAContext, p::Int, n::Int, G)
    stamp_G!(ctx, p, p,  G)
    stamp_G!(ctx, p, n, -G)
    stamp_G!(ctx, n, p, -G)
    stamp_G!(ctx, n, n,  G)
    return nothing
end

"""
    stamp_capacitance!(ctx::MNAContext, p::Int, n::Int, C)

Stamp a capacitance C between nodes p and n.
This is the standard 2-terminal reactive element pattern.
"""
@inline function stamp_capacitance!(ctx::MNAContext, p::Int, n::Int, C)
    stamp_C!(ctx, p, p,  C)
    stamp_C!(ctx, p, n, -C)
    stamp_C!(ctx, n, p, -C)
    stamp_C!(ctx, n, n,  C)
    return nothing
end

#==============================================================================#
# Context Utilities
#==============================================================================#

"""
    reset_values!(ctx::MNAContext)

Reset all matrix values and RHS to zero while preserving structure.
Useful for re-stamping at a new operating point (Newton iteration).
"""
function reset_values!(ctx::MNAContext)
    fill!(ctx.G_V, 0.0)
    fill!(ctx.C_V, 0.0)
    fill!(ctx.b, 0.0)
    return nothing
end

"""
    clear!(ctx::MNAContext)

Completely clear the context, removing all nodes and stamps.
"""
function clear!(ctx::MNAContext)
    empty!(ctx.node_names)
    empty!(ctx.node_to_idx)
    ctx.n_nodes = 0
    empty!(ctx.internal_node_flags)
    empty!(ctx.current_names)
    ctx.n_currents = 0
    empty!(ctx.G_I)
    empty!(ctx.G_J)
    empty!(ctx.G_V)
    empty!(ctx.C_I)
    empty!(ctx.C_J)
    empty!(ctx.C_V)
    empty!(ctx.b)
    empty!(ctx.b_I)
    empty!(ctx.b_V)
    ctx.finalized = false
    return nothing
end

function Base.show(io::IO, ctx::MNAContext)
    n_int = n_internal_nodes(ctx)
    print(io, "MNAContext(")
    print(io, "nodes=$(ctx.n_nodes)")
    if n_int > 0
        print(io, " ($(n_int) internal)")
    end
    print(io, ", currents=$(ctx.n_currents), ")
    print(io, "G_nnz=$(length(ctx.G_V)), ")
    print(io, "C_nnz=$(length(ctx.C_V))")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", ctx::MNAContext)
    n_int = n_internal_nodes(ctx)
    println(io, "MNAContext:")
    if n_int > 0
        println(io, "  Nodes ($(ctx.n_nodes), $(n_int) internal):")
    else
        println(io, "  Nodes ($(ctx.n_nodes)):")
    end
    for (i, name) in enumerate(ctx.node_names)
        if is_internal_node(ctx, i)
            println(io, "    [$i] $name (internal)")
        else
            println(io, "    [$i] $name")
        end
    end
    if ctx.n_currents > 0
        println(io, "  Current variables ($(ctx.n_currents)):")
        for (i, name) in enumerate(ctx.current_names)
            println(io, "    [$(ctx.n_nodes + i)] $name")
        end
    end
    println(io, "  G matrix: $(length(ctx.G_V)) entries")
    println(io, "  C matrix: $(length(ctx.C_V)) entries")
    println(io, "  System size: $(system_size(ctx))")
end
