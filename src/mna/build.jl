#==============================================================================#
# MNA Phase 1: Sparse Matrix Assembly
#
# This module builds sparse matrices from COO format stamps in MNAContext.
# The assembled system can be used for DC, AC, and transient analyses.
#==============================================================================#

using SparseArrays
using LinearAlgebra

export MNASystem, assemble!, assemble_G, assemble_C, get_rhs

"""
    MNASystem{T}

Assembled MNA system ready for analysis.

Contains the sparse G and C matrices and RHS vector b representing:
    G*x + C*dx/dt = b

# Fields
- `G::SparseMatrixCSC{T,Int}`: Conductance matrix (resistive/algebraic part)
- `C::SparseMatrixCSC{T,Int}`: Capacitance matrix (reactive/differential part)
- `b::Vector{T}`: Right-hand side vector (source terms)
- `node_names::Vector{Symbol}`: Node names for solution interpretation
- `current_names::Vector{Symbol}`: Current variable names
- `n_nodes::Int`: Number of voltage nodes
- `n_currents::Int`: Number of current variables

# Solution Interpretation
The solution vector x is ordered as:
    x = [V₁, V₂, ..., Vₙ, I₁, I₂, ..., Iₘ]

# Analysis Modes
- **DC**: Solve G*x = b (set dx/dt = 0)
- **AC**: Solve (G + jωC)*x = b for each frequency ω
- **Transient**: Form ODEProblem with mass matrix C
"""
struct MNASystem{T<:Real}
    G::SparseMatrixCSC{T,Int}
    C::SparseMatrixCSC{T,Int}
    b::Vector{T}
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    charge_names::Vector{Symbol}
    n_nodes::Int
    n_currents::Int
    n_charges::Int
end

# Backwards-compatible constructor (no charges)
function MNASystem{T}(G, C, b, node_names, current_names, n_nodes, n_currents) where {T<:Real}
    MNASystem{T}(G, C, b, node_names, current_names, Symbol[], n_nodes, n_currents, 0)
end

"""
    system_size(sys::MNASystem) -> Int

Return the total system size (number of unknowns).
"""
system_size(sys::MNASystem) = sys.n_nodes + sys.n_currents + sys.n_charges

#==============================================================================#
# Matrix Assembly
#==============================================================================#

"""
    assemble_G(ctx::MNAContext; gmin::Float64=0.0) -> SparseMatrixCSC{Float64,Int}

Assemble the G (conductance) matrix from COO format stamps.
Duplicate entries are summed (standard sparse matrix behavior).

If gmin > 0, GMIN (minimum conductance) is added from each voltage node
to ground to prevent singular matrices from floating nodes. This is
standard SPICE practice for circuit simulation stability with complex
device models.

Note: Negative indices (representing current variables) are resolved
to actual indices using resolve_index before assembly.
"""
function assemble_G(ctx::MNAContext; gmin::Float64=0.0)
    n = system_size(ctx)
    if n == 0
        return spzeros(Float64, 0, 0)
    end

    # Resolve any negative indices (current variables)
    if isempty(ctx.G_I)
        if gmin > 0
            # Only GMIN stamps
            gmin_I = collect(1:ctx.n_nodes)
            gmin_J = collect(1:ctx.n_nodes)
            gmin_V = fill(gmin, ctx.n_nodes)
            return sparse(gmin_I, gmin_J, gmin_V, n, n)
        else
            return spzeros(Float64, n, n)
        end
    end

    I_resolved = Int[resolve_index(ctx, i) for i in ctx.G_I]
    J_resolved = Int[resolve_index(ctx, j) for j in ctx.G_J]

    if gmin > 0
        # Add GMIN from each voltage node to ground
        # This is standard SPICE practice to prevent singular matrices
        # GMIN is only added to voltage nodes (1:n_nodes), not current variables
        gmin_I = collect(1:ctx.n_nodes)
        gmin_J = collect(1:ctx.n_nodes)
        gmin_V = fill(gmin, ctx.n_nodes)

        # Combine stamps with GMIN
        all_I = vcat(I_resolved, gmin_I)
        all_J = vcat(J_resolved, gmin_J)
        all_V = vcat(ctx.G_V, gmin_V)

        return sparse(all_I, all_J, all_V, n, n)
    else
        return sparse(I_resolved, J_resolved, ctx.G_V, n, n)
    end
end

"""
    assemble_C(ctx::MNAContext) -> SparseMatrixCSC{Float64,Int}

Assemble the C (capacitance) matrix from COO format stamps.
Duplicate entries are summed (standard sparse matrix behavior).

Note: Negative indices (representing current variables) are resolved
to actual indices using resolve_index before assembly.
"""
function assemble_C(ctx::MNAContext)
    n = system_size(ctx)
    if n == 0
        return spzeros(Float64, 0, 0)
    end
    if isempty(ctx.C_I)
        return spzeros(Float64, n, n)
    end
    # Resolve any negative indices (current variables)
    I_resolved = Int[resolve_index(ctx, i) for i in ctx.C_I]
    J_resolved = Int[resolve_index(ctx, j) for j in ctx.C_J]
    return sparse(I_resolved, J_resolved, ctx.C_V, n, n)
end

"""
    get_rhs(ctx::MNAContext) -> Vector{Float64}

Get the RHS vector b, properly sized.

Note: Deferred b stamps (for current variables with negative indices)
are resolved and applied here.
"""
function get_rhs(ctx::MNAContext)
    n = system_size(ctx)
    if n == 0
        return Float64[]
    end

    # Create result vector
    result = zeros(Float64, n)

    # Copy existing direct stamps
    for i in 1:min(length(ctx.b), n)
        result[i] = ctx.b[i]
    end

    # Apply deferred stamps (negative indices for current variables)
    for (i, v) in zip(ctx.b_I, ctx.b_V)
        idx = resolve_index(ctx, i)
        if 1 <= idx <= n
            result[idx] += v
        end
    end

    return result
end

"""
    assemble!(ctx::MNAContext) -> MNASystem

Assemble the complete MNA system from the context.
Returns an MNASystem ready for analysis.

# Note on C Matrix Stamping
C matrix stamping is determined by TYPE, not VALUE. Devices with ddt() terms
(detected via `Dual{ContributionTag}` type) always stamp into C to maintain
consistent sparse matrix structure. Devices without ddt() never stamp into C.
This ensures the COO structure is consistent between precompilation and runtime,
even when capacitance values happen to be zero at certain operating points.

# Example
```julia
ctx = MNAContext()
# ... stamp devices ...
sys = assemble!(ctx)
x = sys.G \\ sys.b  # DC solution
```
"""
function assemble!(ctx::MNAContext)
    G = assemble_G(ctx)
    C = assemble_C(ctx)
    b = get_rhs(ctx)

    ctx.finalized = true

    return MNASystem{Float64}(
        G, C, b,
        copy(ctx.node_names),
        copy(ctx.current_names),
        copy(ctx.charge_names),
        ctx.n_nodes,
        ctx.n_currents,
        ctx.n_charges
    )
end

#==============================================================================#
# System Accessors
#==============================================================================#

"""
    node_voltage_indices(sys::MNASystem) -> UnitRange{Int}

Return the indices in the solution vector corresponding to node voltages.
"""
node_voltage_indices(sys::MNASystem) = 1:sys.n_nodes

"""
    current_variable_indices(sys::MNASystem) -> UnitRange{Int}

Return the indices in the solution vector corresponding to current variables.
"""
current_variable_indices(sys::MNASystem) = (sys.n_nodes + 1):(sys.n_nodes + sys.n_currents)

"""
    charge_variable_indices(sys::MNASystem) -> UnitRange{Int}

Return the indices in the solution vector corresponding to charge variables.
"""
charge_variable_indices(sys::MNASystem) = (sys.n_nodes + sys.n_currents + 1):(sys.n_nodes + sys.n_currents + sys.n_charges)

"""
    get_node_index(sys::MNASystem, name::Symbol) -> Int

Get the solution vector index for a node by name.
Returns 0 if the node is ground.
"""
function get_node_index(sys::MNASystem, name::Symbol)
    (name === :gnd || name === Symbol("0")) && return 0
    idx = findfirst(==(name), sys.node_names)
    return idx === nothing ? error("Unknown node: $name") : idx
end

"""
    get_current_index(sys::MNASystem, name::Symbol) -> Int

Get the solution vector index for a current variable by name.
"""
function get_current_index(sys::MNASystem, name::Symbol)
    idx = findfirst(==(name), sys.current_names)
    return idx === nothing ? error("Unknown current: $name") : sys.n_nodes + idx
end

"""
    get_charge_index(sys::MNASystem, name::Symbol) -> Int

Get the solution vector index for a charge variable by name.
"""
function get_charge_index(sys::MNASystem, name::Symbol)
    idx = findfirst(==(name), sys.charge_names)
    return idx === nothing ? error("Unknown charge: $name") : sys.n_nodes + sys.n_currents + idx
end

#==============================================================================#
# Pretty Printing
#==============================================================================#

function Base.show(io::IO, sys::MNASystem)
    print(io, "MNASystem(")
    print(io, "size=$(system_size(sys)), ")
    print(io, "G_nnz=$(nnz(sys.G)), ")
    print(io, "C_nnz=$(nnz(sys.C))")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", sys::MNASystem{T}) where T
    n = system_size(sys)
    println(io, "MNASystem{$T}:")
    println(io, "  System size: $n")
    println(io, "  Voltage nodes: $(sys.n_nodes)")
    println(io, "  Current variables: $(sys.n_currents)")
    println(io, "  Charge variables: $(sys.n_charges)")
    println(io, "  G matrix: $(nnz(sys.G)) nonzeros")
    println(io, "  C matrix: $(nnz(sys.C)) nonzeros")
    if !isempty(sys.node_names)
        println(io, "  Nodes: $(join(sys.node_names, ", "))")
    end
    if !isempty(sys.current_names)
        println(io, "  Currents: $(join(sys.current_names, ", "))")
    end
    if !isempty(sys.charge_names)
        println(io, "  Charges: $(join(sys.charge_names, ", "))")
    end
end

#==============================================================================#
# Matrix Visualization (for debugging)
#==============================================================================#

"""
    show_matrix(io::IO, M::SparseMatrixCSC, names::Vector{Symbol}=Symbol[])

Print a sparse matrix in a readable format (for small matrices).
"""
function show_matrix(io::IO, M::SparseMatrixCSC, names::Vector{Symbol}=Symbol[])
    m, n = size(M)
    if m > 20 || n > 20
        println(io, "Matrix too large to display ($(m)x$(n))")
        return
    end

    # Convert to dense for display
    D = Matrix(M)

    # Header
    print(io, "     ")
    for j in 1:n
        name = j <= length(names) ? string(names[j]) : string(j)
        @printf(io, "%8s", name[1:min(8, length(name))])
    end
    println(io)

    # Rows
    for i in 1:m
        name = i <= length(names) ? string(names[i]) : string(i)
        @printf(io, "%4s ", name[1:min(4, length(name))])
        for j in 1:n
            if D[i, j] == 0
                print(io, "       .")
            else
                @printf(io, "%8.4g", D[i, j])
            end
        end
        println(io)
    end
end

"""
    show_G(sys::MNASystem)

Display the G matrix (for debugging small circuits).
"""
function show_G(sys::MNASystem)
    all_names = vcat(sys.node_names, sys.current_names, sys.charge_names)
    show_matrix(stdout, sys.G, all_names)
end

"""
    show_C(sys::MNASystem)

Display the C matrix (for debugging small circuits).
"""
function show_C(sys::MNASystem)
    all_names = vcat(sys.node_names, sys.current_names, sys.charge_names)
    show_matrix(stdout, sys.C, all_names)
end
