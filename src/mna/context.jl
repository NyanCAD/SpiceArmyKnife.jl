#==============================================================================#
# MNA Phase 1: Context and Stamping Primitives
#
# This module provides the core MNA (Modified Nodal Analysis) infrastructure:
# - MNAContext: Tracks nodes, currents, and matrix stamps
# - Stamping primitives: stamp_G!, stamp_C!, stamp_b!
# - Node/current allocation: get_node!, alloc_current!
#==============================================================================#

export MNAContext, MNASystem
export MNAIndex, NodeIndex, CurrentIndex, ChargeIndex, GroundIndex
export get_node!, alloc_current!, get_current_idx, has_current, resolve_index
export alloc_internal_node!, is_internal_node, n_internal_nodes
export alloc_charge!, get_charge_idx, has_charge, n_charges
export stamp_G!, stamp_C!, stamp_b!
export stamp_conductance!, stamp_capacitance!
export system_size

#==============================================================================#
# Typed Index References
#
# MNA indices are typed to distinguish between different variable types:
# - NodeIndex: Voltage nodes (resolved immediately, index 1..n_nodes)
# - CurrentIndex: Current variables (deferred, resolved to n_nodes + k)
# - ChargeIndex: Charge variables (deferred, resolved to n_nodes + n_currents + k)
# - GroundIndex: Ground node (always resolves to 0, stamps are skipped)
#
# This prevents index corruption when n_nodes changes during stamping.
#==============================================================================#

"""
    MNAIndex

Abstract type for MNA system indices. Subtypes represent different variable types
that are resolved to actual matrix indices at assembly time.
"""
abstract type MNAIndex end

"""
    GroundIndex

Represents the ground node (index 0). Stamps involving ground are skipped.
"""
struct GroundIndex <: MNAIndex end
const GROUND = GroundIndex()

"""
    NodeIndex

Index for a voltage node. Contains the actual matrix row/column index.
Node indices are stable and don't need deferred resolution.
"""
struct NodeIndex <: MNAIndex
    idx::Int
end

"""
    CurrentIndex

Index for a current variable (e.g., from voltage sources, inductors).
Contains the current variable number k; resolved to n_nodes + k at assembly.
"""
struct CurrentIndex <: MNAIndex
    k::Int  # Current variable number (1-based)
end

"""
    ChargeIndex

Index for a charge state variable (voltage-dependent capacitors).
Contains the charge variable number k; resolved to n_nodes + n_currents + k at assembly.
"""
struct ChargeIndex <: MNAIndex
    k::Int  # Charge variable number (1-based)
end

# Ground check for stamps
Base.iszero(::GroundIndex) = true
Base.iszero(::NodeIndex) = false
Base.iszero(::CurrentIndex) = false
Base.iszero(::ChargeIndex) = false

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
    G_I::Vector{MNAIndex}
    G_J::Vector{MNAIndex}
    G_V::Vector{Float64}

    # COO format for C matrix (capacitance/reactive)
    C_I::Vector{MNAIndex}
    C_J::Vector{MNAIndex}
    C_V::Vector{Float64}

    # RHS vector (pre-allocated for known size, or extended dynamically)
    b::Vector{Float64}

    # Deferred b vector stamps (resolved at assembly time when n_nodes is final)
    b_I::Vector{MNAIndex}
    b_V::Vector{Float64}

    # Charge state variables (for voltage-dependent capacitors)
    # See doc/voltage_dependent_capacitors.md
    # These are differential variables with dq/dt = I and constraint q = Q(V)
    charge_names::Vector{Symbol}
    n_charges::Int
    # Branch nodes (p, n) for each charge - links charge to KCL
    charge_branches::Vector{Tuple{Int,Int}}

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
        MNAIndex[],         # G_I
        MNAIndex[],         # G_J
        Float64[],          # G_V
        MNAIndex[],         # C_I
        MNAIndex[],         # C_J
        Float64[],          # C_V
        Float64[],          # b
        MNAIndex[],         # b_I (deferred b stamps)
        Float64[],          # b_V (deferred b stamps)
        Symbol[],           # charge_names
        0,                  # n_charges
        Tuple{Int,Int}[],   # charge_branches
        false               # finalized
    )
end

"""
    system_size(ctx::MNAContext) -> Int

Return the total system size (number of unknowns).
Includes nodes, current variables, and charge state variables.

System vector layout: [V₁...Vₙ, I₁...Iₘ, q₁...qₖ]
"""
@inline system_size(ctx::MNAContext) = ctx.n_nodes + ctx.n_currents + ctx.n_charges

"""
    n_charges(ctx::MNAContext) -> Int

Return the number of charge state variables.
"""
@inline n_charges(ctx::MNAContext) = ctx.n_charges

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
# i_idx is a CurrentIndex, resolved to n_nodes + k at assembly time
```

Returns a `CurrentIndex` representing current variable k.
This allows n_nodes to change (e.g., from internal node allocation)
without invalidating previously stamped indices.
"""
function alloc_current!(ctx::MNAContext, name::Symbol)::CurrentIndex
    ctx.n_currents += 1
    push!(ctx.current_names, name)
    return CurrentIndex(ctx.n_currents)
end

alloc_current!(ctx::MNAContext, name::String) = alloc_current!(ctx, Symbol(name))

"""
    resolve_index(ctx::MNAContext, idx::MNAIndex) -> Int

Resolve an MNA index to an actual system row/column index.

- GroundIndex: Returns 0 (stamps involving ground are skipped)
- NodeIndex: Returns the node index unchanged
- CurrentIndex(k): Returns n_nodes + k
- ChargeIndex(k): Returns n_nodes + n_currents + k

This allows current and charge variable indices to be stamped before all nodes are known,
and resolved at assembly time when n_nodes is final.
"""
@inline resolve_index(ctx::MNAContext, ::GroundIndex)::Int = 0
@inline resolve_index(ctx::MNAContext, idx::NodeIndex)::Int = idx.idx
@inline resolve_index(ctx::MNAContext, idx::CurrentIndex)::Int = ctx.n_nodes + idx.k
@inline resolve_index(ctx::MNAContext, idx::ChargeIndex)::Int = ctx.n_nodes + ctx.n_currents + idx.k

"""
    get_current_idx(ctx::MNAContext, name::Symbol) -> CurrentIndex

Look up a current variable index by name.
Returns a `CurrentIndex` representing current variable k.
Use `resolve_index` to get the actual system index.
Throws an error if the current variable doesn't exist.
"""
function get_current_idx(ctx::MNAContext, name::Symbol)::CurrentIndex
    idx = findfirst(==(name), ctx.current_names)
    idx === nothing && error("Current variable $name not found in MNA context")
    return CurrentIndex(idx)
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
# Charge State Variable Allocation (Voltage-Dependent Capacitors)
#==============================================================================#

"""
    alloc_charge!(ctx::MNAContext, name::Symbol, p::Int, n::Int) -> ChargeIndex

Allocate a charge state variable for a voltage-dependent capacitor.

Charge variables are used to reformulate voltage-dependent capacitors with
a constant mass matrix. Instead of `C(V) * dV/dt`, we use:
- `dq/dt = I` (differential equation with constant coefficient)
- `q = Q(V)` (algebraic constraint)

This yields a constant mass matrix suitable for SciML Rosenbrock solvers.

# Arguments
- `ctx`: MNA context
- `name`: Unique name for the charge variable
- `p`, `n`: Branch nodes - current flows from p to n

# Returns
A `ChargeIndex` representing charge variable k, resolved to
n_nodes + n_currents + k at assembly time by `resolve_index`.

# Example
```julia
# For voltage-dependent junction capacitor on branch (a, c):
q_idx = alloc_charge!(ctx, :Q_Cj_D1, a, c)

# Now stamp:
# - C[q_idx, q_idx] = 1.0 (dq/dt term)
# - C[a, q_idx] = 1.0, C[c, q_idx] = -1.0 (current I = dq/dt in KCL)
# - G[q_idx, ...] and b[q_idx] for constraint q = Q(V)
```

See doc/voltage_dependent_capacitors.md for full formulation.
"""
function alloc_charge!(ctx::MNAContext, name::Symbol, p::Int, n::Int)::ChargeIndex
    ctx.n_charges += 1
    push!(ctx.charge_names, name)
    push!(ctx.charge_branches, (p, n))
    return ChargeIndex(ctx.n_charges)
end

alloc_charge!(ctx::MNAContext, name::String, p::Int, n::Int) = alloc_charge!(ctx, Symbol(name), p, n)

"""
    get_charge_idx(ctx::MNAContext, name::Symbol) -> ChargeIndex

Look up a charge variable index by name.
Returns a `ChargeIndex` representing charge variable k.
Use `resolve_index` to get the actual system index.
Throws an error if the charge variable doesn't exist.
"""
function get_charge_idx(ctx::MNAContext, name::Symbol)::ChargeIndex
    idx = findfirst(==(name), ctx.charge_names)
    idx === nothing && error("Charge variable $name not found in MNA context")
    return ChargeIndex(idx)
end

get_charge_idx(ctx::MNAContext, name::String) = get_charge_idx(ctx, Symbol(name))

"""
    has_charge(ctx::MNAContext, name::Symbol) -> Bool

Check if a charge variable with the given name exists.
"""
function has_charge(ctx::MNAContext, name::Symbol)::Bool
    return name in ctx.charge_names
end

"""
    get_charge_branch(ctx::MNAContext, name::Symbol) -> Tuple{Int, Int}

Get the branch nodes (p, n) for a charge variable.
The current I = dq/dt flows from p to n.
"""
function get_charge_branch(ctx::MNAContext, name::Symbol)::Tuple{Int, Int}
    idx = findfirst(==(name), ctx.charge_names)
    idx === nothing && error("Charge variable $name not found in MNA context")
    return ctx.charge_branches[idx]
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

# Convert to typed index for storage
@inline _to_typed(idx::Int) = NodeIndex(idx)
@inline _to_typed(idx::NodeIndex) = idx
@inline _to_typed(idx::CurrentIndex) = idx
@inline _to_typed(idx::ChargeIndex) = idx

"""
    stamp_G!(ctx::MNAContext, i, j, val)

Stamp a value into the G (conductance) matrix at position (i, j).
Entries involving ground are skipped.

Accepts typed indices (NodeIndex, CurrentIndex, ChargeIndex) or integers.

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
@inline function stamp_G!(ctx::MNAContext, i, j, val)
    iszero(i) && return nothing
    iszero(j) && return nothing
    v = extract_value(val)
    push!(ctx.G_I, _to_typed(i))
    push!(ctx.G_J, _to_typed(j))
    push!(ctx.G_V, v)
    return nothing
end

"""
    stamp_C!(ctx::MNAContext, i, j, val)

Stamp a value into the C (capacitance) matrix at position (i, j).
Entries involving ground are skipped.

Accepts typed indices (NodeIndex, CurrentIndex, ChargeIndex) or integers.

The C matrix represents reactive (differential) relationships:
- Capacitor: stamps ±C pattern
- Inductor: stamps -L in current equation

For the ODE formulation: C * dx/dt + G * x = b
"""
@inline function stamp_C!(ctx::MNAContext, i, j, val)
    iszero(i) && return nothing
    iszero(j) && return nothing
    v = extract_value(val)
    push!(ctx.C_I, _to_typed(i))
    push!(ctx.C_J, _to_typed(j))
    push!(ctx.C_V, v)
    return nothing
end

"""
    stamp_b!(ctx::MNAContext, i, val)

Add a value to the RHS vector b at position i.
Entries at ground are skipped.

Accepts typed indices (NodeIndex, CurrentIndex, ChargeIndex) or integers.

The b vector contains source terms:
- Independent current sources: stamp into KCL equations
- Independent voltage sources: stamp voltage value into constraint equation
"""
@inline function stamp_b!(ctx::MNAContext, i, val)
    iszero(i) && return nothing
    v = extract_value(val)
    typed = _to_typed(i)

    # CurrentIndex/ChargeIndex are deferred until n_nodes is final
    if typed isa CurrentIndex || typed isa ChargeIndex
        push!(ctx.b_I, typed)
        push!(ctx.b_V, v)
        return nothing
    end

    # NodeIndex can be applied immediately
    idx = typed.idx
    if idx > length(ctx.b)
        old_len = length(ctx.b)
        resize!(ctx.b, idx)
        @inbounds for j in (old_len+1):idx
            ctx.b[j] = 0.0
        end
    end
    ctx.b[idx] += v
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
    reset_for_restamping!(ctx::MNAContext)

Reset the context for restamping at a new operating point.

This is used for reduced-allocation Newton iteration. The context is fully reset
but its internal arrays retain their allocated capacity, avoiding reallocation.

The builder will recreate the node/current structure (which is fixed) and
restamp all devices. Since the internal vectors have capacity from the previous
build, push! operations won't allocate new memory.

# Usage
```julia
# In PrecompiledCircuit
ctx = pc.ctx  # stored context from initial build
reset_for_restamping!(ctx)
pc.builder(pc.params, pc.spec, t; x=u, ctx=ctx)  # restamps into same ctx
```
"""
function reset_for_restamping!(ctx::MNAContext)
    # Empty all arrays but preserve capacity (empty! keeps allocated memory)

    # Node structure
    empty!(ctx.node_names)
    empty!(ctx.node_to_idx)
    ctx.n_nodes = 0
    empty!(ctx.internal_node_flags)

    # Current variables
    empty!(ctx.current_names)
    ctx.n_currents = 0

    # COO arrays
    empty!(ctx.G_I)
    empty!(ctx.G_J)
    empty!(ctx.G_V)
    empty!(ctx.C_I)
    empty!(ctx.C_J)
    empty!(ctx.C_V)

    # Reset b vector to zeros (keep the allocation but resize to 0 and regrow)
    # Note: We empty! here because the size depends on n_nodes + n_currents + n_charges
    # which will be rebuilt
    empty!(ctx.b)

    # Deferred b stamps
    empty!(ctx.b_I)
    empty!(ctx.b_V)

    # Charge state variables
    empty!(ctx.charge_names)
    ctx.n_charges = 0
    empty!(ctx.charge_branches)

    ctx.finalized = false
    return nothing
end

export reset_for_restamping!

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
    empty!(ctx.charge_names)
    ctx.n_charges = 0
    empty!(ctx.charge_branches)
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
    print(io, ", currents=$(ctx.n_currents)")
    if ctx.n_charges > 0
        print(io, ", charges=$(ctx.n_charges)")
    end
    print(io, ", G_nnz=$(length(ctx.G_V)), ")
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
    if ctx.n_charges > 0
        println(io, "  Charge variables ($(ctx.n_charges)):")
        base_idx = ctx.n_nodes + ctx.n_currents
        for (i, name) in enumerate(ctx.charge_names)
            (p, n) = ctx.charge_branches[i]
            println(io, "    [$(base_idx + i)] $name (branch $p → $n)")
        end
    end
    println(io, "  G matrix: $(length(ctx.G_V)) entries")
    println(io, "  C matrix: $(length(ctx.C_V)) entries")
    println(io, "  System size: $(system_size(ctx))")
end
