#==============================================================================#
# MNA Phase 1: Context and Stamping Primitives
#
# This module provides the core MNA (Modified Nodal Analysis) infrastructure:
# - MNAContext: Tracks nodes, currents, and matrix stamps
# - Stamping primitives: stamp_G!, stamp_C!, stamp_b!
# - Node/current allocation: get_node!, alloc_current!
#==============================================================================#

export MNAContext, MNASystem
export get_node!, alloc_current!, get_current_idx, has_current
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
        Symbol[],           # current_names
        0,                  # n_currents
        Int[],              # G_I
        Int[],              # G_J
        Float64[],          # G_V
        Int[],              # C_I
        Int[],              # C_J
        Float64[],          # C_V
        Float64[],          # b
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

    # Extend b vector if needed
    if length(ctx.b) < system_size(ctx)
        resize!(ctx.b, system_size(ctx))
        ctx.b[end] = 0.0
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
# i_idx = n_nodes + 1 (for first current variable)
```
"""
function alloc_current!(ctx::MNAContext, name::Symbol)::Int
    ctx.n_currents += 1
    push!(ctx.current_names, name)

    # Current variable index in system
    idx = ctx.n_nodes + ctx.n_currents

    # Extend b vector if needed
    if length(ctx.b) < idx
        resize!(ctx.b, idx)
        ctx.b[end] = 0.0
    end

    return idx
end

alloc_current!(ctx::MNAContext, name::String) = alloc_current!(ctx, Symbol(name))

"""
    get_current_idx(ctx::MNAContext, name::Symbol) -> Int

Look up a current variable index by name.
Returns the system index (n_nodes + position in current_names).
Throws an error if the current variable doesn't exist.

# Example
```julia
# After stamping a voltage source V1
i_idx = get_current_idx(ctx, :I_V1)
```
"""
function get_current_idx(ctx::MNAContext, name::Symbol)::Int
    idx = findfirst(==(name), ctx.current_names)
    idx === nothing && error("Current variable $name not found in MNA context")
    return ctx.n_nodes + idx
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
    v == 0 && return nothing  # Skip zero entries
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
    v == 0 && return nothing  # Skip zero entries
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
    v == 0 && return nothing

    # Extend b if needed
    if i > length(ctx.b)
        resize!(ctx.b, i)
        ctx.b[i] = 0.0
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
    empty!(ctx.current_names)
    ctx.n_currents = 0
    empty!(ctx.G_I)
    empty!(ctx.G_J)
    empty!(ctx.G_V)
    empty!(ctx.C_I)
    empty!(ctx.C_J)
    empty!(ctx.C_V)
    empty!(ctx.b)
    ctx.finalized = false
    return nothing
end

function Base.show(io::IO, ctx::MNAContext)
    print(io, "MNAContext(")
    print(io, "nodes=$(ctx.n_nodes), ")
    print(io, "currents=$(ctx.n_currents), ")
    print(io, "G_nnz=$(length(ctx.G_V)), ")
    print(io, "C_nnz=$(length(ctx.C_V))")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", ctx::MNAContext)
    println(io, "MNAContext:")
    println(io, "  Nodes ($(ctx.n_nodes)):")
    for (i, name) in enumerate(ctx.node_names)
        println(io, "    [$i] $name")
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
