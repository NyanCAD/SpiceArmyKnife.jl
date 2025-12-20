# Phase 1: MNA Core and Primitives

## Overview

This phase establishes the foundational MNA infrastructure: data structures, matrix stamping primitives, and a basic DC solver. Everything is tested at the Julia level before integration with the parser/codegen.

## Deliverables

1. `src/mna/types.jl` - Core data structures
2. `src/mna/stamping.jl` - Matrix stamping primitives
3. `src/mna/dc.jl` - DC operating point solver
4. `src/mna/sparse.jl` - Sparse matrix construction
5. `test/mna/test_core.jl` - Unit tests

## Data Structures

### Node Representation

```julia
# src/mna/types.jl

"""
Ground node singleton - index 0, never stamped
"""
struct GroundNode end
const GND = GroundNode()

"""
Circuit node with MNA index.
Index 1..n for internal nodes.
"""
struct Node
    index::Int
    name::Symbol
end

"""
Branch for voltage sources/inductors.
Adds an extra unknown (branch current) and equation.
"""
struct Branch
    index::Int          # Index in branch current section of x vector
    pos::Union{Node, GroundNode}
    neg::Union{Node, GroundNode}
    name::Symbol
end

# State vector layout:
# x = [V_1, V_2, ..., V_n, I_br1, I_br2, ..., I_brm]
#     |---- n nodes ----|  |---- m branches ------|
```

### MNA Context

```julia
"""
Accumulates device contributions during circuit elaboration and evaluation.
"""
mutable struct MNAContext
    # Topology
    nodes::Dict{Symbol, Node}
    branches::Dict{Symbol, Branch}
    next_node_index::Int
    next_branch_index::Int

    # Matrix entries (COO format for construction)
    # Each entry: (row, col, value)
    G_coo::Vector{Tuple{Int, Int, Float64}}  # Resistive Jacobian
    C_coo::Vector{Tuple{Int, Int, Float64}}  # Reactive Jacobian

    # RHS vector
    b::Vector{Float64}

    # Residual vector (for nonlinear iteration)
    residual::Vector{Float64}

    # Current solution
    x::Vector{Float64}

    # Problem dimensions
    n_nodes::Int
    n_branches::Int
end

function MNAContext()
    MNAContext(
        Dict{Symbol, Node}(),
        Dict{Symbol, Branch}(),
        1, 1,
        Tuple{Int, Int, Float64}[],
        Tuple{Int, Int, Float64}[],
        Float64[],
        Float64[],
        Float64[],
        0, 0
    )
end
```

### Finalized Circuit

```julia
"""
Immutable circuit after elaboration. Ready for solving.
"""
struct MNACircuit
    n_nodes::Int
    n_branches::Int
    n_vars::Int  # n_nodes + n_branches

    # Sparse matrices (CSC format)
    G::SparseMatrixCSC{Float64, Int}
    C::SparseMatrixCSC{Float64, Int}

    # Node/branch name mappings
    node_names::Vector{Symbol}
    branch_names::Vector{Symbol}

    # For nonlinear circuits: device evaluation functions
    devices::Vector{Any}

    # Differential variables mask (for DAEProblem)
    differential_vars::BitVector
end
```

## Matrix Stamping Primitives

### Core Stamping Functions

```julia
# src/mna/stamping.jl

"""
Get MNA index for a node. Ground returns 0 (skip stamping).
"""
node_index(n::Node) = n.index
node_index(::GroundNode) = 0

"""
Get MNA index for a branch current (offset by n_nodes).
"""
branch_index(ctx::MNAContext, b::Branch) = ctx.n_nodes + b.index

"""
Stamp a value into a matrix (COO format).
Skips if either index is 0 (ground).
"""
function stamp!(coo::Vector{Tuple{Int,Int,Float64}}, row::Int, col::Int, val::Float64)
    row == 0 && return
    col == 0 && return
    push!(coo, (row, col, val))
end

"""
Stamp conductance between two nodes.
MNA pattern for conductance G between nodes p and n:
  G[p,p] += G    G[p,n] -= G
  G[n,p] -= G    G[n,n] += G
"""
function stamp_conductance!(ctx::MNAContext, p::Union{Node,GroundNode},
                            n::Union{Node,GroundNode}, G::Float64)
    pi, ni = node_index(p), node_index(n)
    stamp!(ctx.G_coo, pi, pi, +G)
    stamp!(ctx.G_coo, pi, ni, -G)
    stamp!(ctx.G_coo, ni, pi, -G)
    stamp!(ctx.G_coo, ni, ni, +G)
end

"""
Stamp capacitance between two nodes into C matrix.
Same pattern as conductance.
"""
function stamp_capacitance!(ctx::MNAContext, p::Union{Node,GroundNode},
                            n::Union{Node,GroundNode}, C::Float64)
    pi, ni = node_index(p), node_index(n)
    stamp!(ctx.C_coo, pi, pi, +C)
    stamp!(ctx.C_coo, pi, ni, -C)
    stamp!(ctx.C_coo, ni, pi, -C)
    stamp!(ctx.C_coo, ni, ni, +C)
end

"""
Stamp current source from node n to node p (current flows p → n).
Adds to RHS vector: b[p] += I, b[n] -= I
"""
function stamp_current_source!(ctx::MNAContext, p::Union{Node,GroundNode},
                               n::Union{Node,GroundNode}, I::Float64)
    pi, ni = node_index(p), node_index(n)
    if pi != 0
        ctx.b[pi] += I
    end
    if ni != 0
        ctx.b[ni] -= I
    end
end

"""
Stamp voltage source. Requires a branch variable.
Adds KCL contributions and voltage constraint equation.

For V(p,n) = V:
  KCL at p: I_branch flows out
  KCL at n: I_branch flows in
  Constraint: V_p - V_n = V
"""
function stamp_voltage_source!(ctx::MNAContext, branch::Branch, V::Float64)
    pi = node_index(branch.pos)
    ni = node_index(branch.neg)
    bi = branch_index(ctx, branch)

    # KCL stamps: branch current contribution
    # G[p, branch] = +1  (current leaves p)
    # G[n, branch] = -1  (current enters n)
    stamp!(ctx.G_coo, pi, bi, +1.0)
    stamp!(ctx.G_coo, ni, bi, -1.0)

    # Voltage constraint equation (row = branch index)
    # G[branch, p] = +1
    # G[branch, n] = -1
    # b[branch] = V
    stamp!(ctx.G_coo, bi, pi, +1.0)
    stamp!(ctx.G_coo, bi, ni, -1.0)
    ctx.b[bi] = V
end

"""
Stamp VCCS (Voltage-Controlled Current Source).
I(p,n) = gm * V(cp, cn)

Stamps transconductance gm:
  G[p, cp] += gm
  G[p, cn] -= gm
  G[n, cp] -= gm
  G[n, cn] += gm
"""
function stamp_vccs!(ctx::MNAContext,
                     p::Union{Node,GroundNode}, n::Union{Node,GroundNode},
                     cp::Union{Node,GroundNode}, cn::Union{Node,GroundNode},
                     gm::Float64)
    pi, ni = node_index(p), node_index(n)
    cpi, cni = node_index(cp), node_index(cn)

    stamp!(ctx.G_coo, pi, cpi, +gm)
    stamp!(ctx.G_coo, pi, cni, -gm)
    stamp!(ctx.G_coo, ni, cpi, -gm)
    stamp!(ctx.G_coo, ni, cni, +gm)
end

"""
Stamp VCVS (Voltage-Controlled Voltage Source).
V(p,n) = gain * V(cp,cn)

Uses branch variable for the output voltage.
"""
function stamp_vcvs!(ctx::MNAContext, branch::Branch,
                     cp::Union{Node,GroundNode}, cn::Union{Node,GroundNode},
                     gain::Float64)
    pi = node_index(branch.pos)
    ni = node_index(branch.neg)
    bi = branch_index(ctx, branch)
    cpi, cni = node_index(cp), node_index(cn)

    # KCL: branch current
    stamp!(ctx.G_coo, pi, bi, +1.0)
    stamp!(ctx.G_coo, ni, bi, -1.0)

    # Constraint: V_p - V_n - gain*(V_cp - V_cn) = 0
    stamp!(ctx.G_coo, bi, pi, +1.0)
    stamp!(ctx.G_coo, bi, ni, -1.0)
    stamp!(ctx.G_coo, bi, cpi, -gain)
    stamp!(ctx.G_coo, bi, cni, +gain)
end

"""
Stamp inductor. Uses branch variable for current.
V(p,n) = L * dI/dt

In MNA:
  - KCL: branch current flows through inductor
  - KVL in C matrix: L coefficient for dI/dt
  - Marks branch current as differential variable
"""
function stamp_inductor!(ctx::MNAContext, branch::Branch, L::Float64)
    pi = node_index(branch.pos)
    ni = node_index(branch.neg)
    bi = branch_index(ctx, branch)

    # KCL stamps
    stamp!(ctx.G_coo, pi, bi, +1.0)
    stamp!(ctx.G_coo, ni, bi, -1.0)

    # KVL: V_p - V_n = L * dI/dt
    # In G: V_p - V_n term
    stamp!(ctx.G_coo, bi, pi, +1.0)
    stamp!(ctx.G_coo, bi, ni, -1.0)

    # In C: L * dI/dt term (negative because moved to LHS)
    stamp!(ctx.C_coo, bi, bi, -L)
end
```

### Matrix Finalization

```julia
# src/mna/sparse.jl

using SparseArrays

"""
Convert COO entries to CSC sparse matrix.
Combines duplicate entries by summing.
"""
function coo_to_sparse(coo::Vector{Tuple{Int,Int,Float64}}, n::Int)
    if isempty(coo)
        return spzeros(n, n)
    end
    rows = [e[1] for e in coo]
    cols = [e[2] for e in coo]
    vals = [e[3] for e in coo]
    return sparse(rows, cols, vals, n, n)
end

"""
Finalize MNA context into immutable circuit.
"""
function finalize!(ctx::MNAContext)
    n = ctx.n_nodes + ctx.n_branches

    G = coo_to_sparse(ctx.G_coo, n)
    C = coo_to_sparse(ctx.C_coo, n)

    # Resize vectors
    resize!(ctx.b, n)
    resize!(ctx.residual, n)
    resize!(ctx.x, n)

    # Determine differential variables:
    # - Nodes with capacitors: V is differential
    # - Inductor branches: I is differential
    differential_vars = falses(n)

    # Check C matrix diagonal for nonzeros
    for i in 1:n
        if C[i,i] != 0.0
            differential_vars[i] = true
        end
    end

    node_names = [k for (k,v) in sort(collect(ctx.nodes), by=x->x[2].index)]
    branch_names = [k for (k,v) in sort(collect(ctx.branches), by=x->x[2].index)]

    MNACircuit(
        ctx.n_nodes,
        ctx.n_branches,
        n,
        G, C,
        node_names,
        branch_names,
        Any[],
        differential_vars
    )
end
```

## DC Solver

### Linear DC Solve

```julia
# src/mna/dc.jl

using LinearAlgebra
using SparseArrays

"""
Solve DC operating point for linear circuit.
Solves: G * x = b
Returns solution vector x.
"""
function solve_dc_linear(circuit::MNACircuit)
    # For DC: reactive terms are zero, solve G*x = b
    x = circuit.G \ circuit.b
    return x
end

"""
DC solution result with named access.
"""
struct DCSolution
    x::Vector{Float64}
    node_voltages::Dict{Symbol, Float64}
    branch_currents::Dict{Symbol, Float64}
end

function DCSolution(circuit::MNACircuit, x::Vector{Float64})
    node_voltages = Dict{Symbol, Float64}()
    for (i, name) in enumerate(circuit.node_names)
        node_voltages[name] = x[i]
    end

    branch_currents = Dict{Symbol, Float64}()
    for (i, name) in enumerate(circuit.branch_names)
        branch_currents[name] = x[circuit.n_nodes + i]
    end

    DCSolution(x, node_voltages, branch_currents)
end

"""
Solve DC operating point, return named solution.
"""
function solve_dc(circuit::MNACircuit)
    x = solve_dc_linear(circuit)
    DCSolution(circuit, x)
end
```

### Nonlinear DC Solve (Newton-Raphson)

```julia
"""
Nonlinear DC solver using Newton-Raphson iteration.
For circuits with diodes, transistors, etc.
"""
function solve_dc_nonlinear(circuit::MNACircuit;
                            maxiter::Int = 100,
                            abstol::Float64 = 1e-12,
                            reltol::Float64 = 1e-6)
    n = circuit.n_vars
    x = zeros(n)  # Initial guess
    residual = zeros(n)
    J = copy(circuit.G)  # Jacobian starts as G

    for iter in 1:maxiter
        # Evaluate all devices, get residual and Jacobian
        fill!(residual, 0.0)

        # Start with linear part
        residual .= circuit.G * x .- circuit.b

        # Add nonlinear device contributions
        # (Placeholder - devices update residual and J)
        for device in circuit.devices
            evaluate!(device, residual, J, x)
        end

        # Check convergence (residual-based, VACASK style)
        converged = true
        for i in 1:n
            tol = abs(x[i]) * reltol + abstol
            if abs(residual[i]) > tol
                converged = false
                break
            end
        end

        if converged
            return DCSolution(circuit, x), iter
        end

        # Newton step: J * dx = -residual
        dx = J \ (-residual)

        # Update with damping if needed
        x .+= dx
    end

    error("DC Newton-Raphson failed to converge in $maxiter iterations")
end
```

## Node/Branch Creation API

```julia
"""
Get or create a node by name.
"""
function get_node!(ctx::MNAContext, name::Symbol)
    if haskey(ctx.nodes, name)
        return ctx.nodes[name]
    end
    node = Node(ctx.next_node_index, name)
    ctx.nodes[name] = node
    ctx.next_node_index += 1
    ctx.n_nodes = max(ctx.n_nodes, node.index)
    return node
end

"""
Create a branch for voltage sources/inductors.
"""
function create_branch!(ctx::MNAContext, name::Symbol,
                        pos::Union{Node,GroundNode},
                        neg::Union{Node,GroundNode})
    branch = Branch(ctx.next_branch_index, pos, neg, name)
    ctx.branches[name] = branch
    ctx.next_branch_index += 1
    ctx.n_branches = max(ctx.n_branches, branch.index)
    return branch
end
```

## Unit Tests

### Test Cases

```julia
# test/mna/test_core.jl

using Test
using SpiceArmyKnife.MNA

@testset "MNA Core" begin

    @testset "Voltage Divider" begin
        # V1 = 10V, R1 = 1k, R2 = 1k
        # Expected: V_mid = 5V, I = 5mA
        ctx = MNAContext()

        n1 = get_node!(ctx, :n1)  # Top (V1)
        n2 = get_node!(ctx, :n2)  # Middle
        # Ground is implicit

        # Voltage source V1 = 10V from n1 to GND
        br_v1 = create_branch!(ctx, :V1, n1, GND)
        stamp_voltage_source!(ctx, br_v1, 10.0)

        # R1 = 1k between n1 and n2
        stamp_conductance!(ctx, n1, n2, 1e-3)  # G = 1/R = 1mS

        # R2 = 1k between n2 and GND
        stamp_conductance!(ctx, n2, GND, 1e-3)

        circuit = finalize!(ctx)
        sol = solve_dc(circuit)

        @test sol.node_voltages[:n1] ≈ 10.0 atol=1e-10
        @test sol.node_voltages[:n2] ≈ 5.0 atol=1e-10
        @test sol.branch_currents[:V1] ≈ -5e-3 atol=1e-12  # Current into source
    end

    @testset "Current Source" begin
        # I1 = 1mA into node, R1 = 1k to ground
        # Expected: V = I*R = 1V
        ctx = MNAContext()

        n1 = get_node!(ctx, :n1)

        stamp_current_source!(ctx, n1, GND, 1e-3)  # 1mA into n1
        stamp_conductance!(ctx, n1, GND, 1e-3)     # 1k to ground

        circuit = finalize!(ctx)
        sol = solve_dc(circuit)

        @test sol.node_voltages[:n1] ≈ 1.0 atol=1e-10
    end

    @testset "VCCS" begin
        # V1 = 2V, gm = 1mS, R_load = 1k
        # I_out = gm * V1 = 2mA
        # V_out = I_out * R = 2V
        ctx = MNAContext()

        n_in = get_node!(ctx, :n_in)
        n_out = get_node!(ctx, :n_out)

        # Input voltage source
        br_v1 = create_branch!(ctx, :V1, n_in, GND)
        stamp_voltage_source!(ctx, br_v1, 2.0)

        # VCCS: I(n_out, GND) = 1mS * V(n_in, GND)
        stamp_vccs!(ctx, n_out, GND, n_in, GND, 1e-3)

        # Load resistor
        stamp_conductance!(ctx, n_out, GND, 1e-3)

        circuit = finalize!(ctx)
        sol = solve_dc(circuit)

        @test sol.node_voltages[:n_in] ≈ 2.0 atol=1e-10
        @test sol.node_voltages[:n_out] ≈ 2.0 atol=1e-10
    end

    @testset "VCVS" begin
        # V1 = 1V, gain = 10
        # V_out = 10V
        ctx = MNAContext()

        n_in = get_node!(ctx, :n_in)
        n_out = get_node!(ctx, :n_out)

        # Input voltage source
        br_v1 = create_branch!(ctx, :V1, n_in, GND)
        stamp_voltage_source!(ctx, br_v1, 1.0)

        # VCVS: V(n_out, GND) = 10 * V(n_in, GND)
        br_vcvs = create_branch!(ctx, :E1, n_out, GND)
        stamp_vcvs!(ctx, br_vcvs, n_in, GND, 10.0)

        # Load resistor (needed to have a path to ground)
        stamp_conductance!(ctx, n_out, GND, 1e-3)

        circuit = finalize!(ctx)
        sol = solve_dc(circuit)

        @test sol.node_voltages[:n_out] ≈ 10.0 atol=1e-10
    end

    @testset "Wheatstone Bridge" begin
        #     V1=10V
        #       |
        #      n1
        #     /  \
        #   R1    R2
        #   /      \
        #  n2      n3
        #   \      /
        #   R3    R4
        #    \    /
        #     GND
        #
        # R1=R4=1k, R2=R3=2k → balanced, V(n2)=V(n3)
        ctx = MNAContext()

        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)
        n3 = get_node!(ctx, :n3)

        br_v1 = create_branch!(ctx, :V1, n1, GND)
        stamp_voltage_source!(ctx, br_v1, 10.0)

        stamp_conductance!(ctx, n1, n2, 1e-3)  # R1 = 1k
        stamp_conductance!(ctx, n1, n3, 0.5e-3)  # R2 = 2k
        stamp_conductance!(ctx, n2, GND, 0.5e-3)  # R3 = 2k
        stamp_conductance!(ctx, n3, GND, 1e-3)  # R4 = 1k

        circuit = finalize!(ctx)
        sol = solve_dc(circuit)

        # Balanced bridge: V(n2) = V(n3) = 10 * R3/(R1+R3) = 10 * 2/3 ≈ 6.667V
        @test sol.node_voltages[:n2] ≈ sol.node_voltages[:n3] atol=1e-10
        @test sol.node_voltages[:n2] ≈ 20/3 atol=1e-10
    end

end

@testset "Capacitor Structure" begin
    # Just test that capacitors stamp into C matrix
    ctx = MNAContext()

    n1 = get_node!(ctx, :n1)
    stamp_capacitance!(ctx, n1, GND, 1e-12)  # 1pF

    circuit = finalize!(ctx)

    @test circuit.C[1,1] ≈ 1e-12
    @test circuit.differential_vars[1] == true
end

@testset "Inductor Structure" begin
    ctx = MNAContext()

    n1 = get_node!(ctx, :n1)
    n2 = get_node!(ctx, :n2)

    br_l = create_branch!(ctx, :L1, n1, n2)
    stamp_inductor!(ctx, br_l, 1e-6)  # 1µH

    circuit = finalize!(ctx)

    # Inductor branch should be differential
    bi = circuit.n_nodes + 1
    @test circuit.C[bi, bi] ≈ -1e-6
    @test circuit.differential_vars[bi] == true
end
```

## Integration with Existing Code

### Module Structure

```julia
# src/mna/MNA.jl

module MNA

using SparseArrays
using LinearAlgebra

include("types.jl")
include("stamping.jl")
include("sparse.jl")
include("dc.jl")

export MNAContext, MNACircuit, DCSolution
export Node, Branch, GND, GroundNode
export get_node!, create_branch!, finalize!
export stamp_conductance!, stamp_capacitance!
export stamp_current_source!, stamp_voltage_source!
export stamp_vccs!, stamp_vcvs!, stamp_inductor!
export solve_dc, solve_dc_linear, solve_dc_nonlinear

end
```

### Add to Main Package

```julia
# In src/CedarSim.jl, add:
include("mna/MNA.jl")
using .MNA
```

## Success Criteria

Phase 1 is complete when:

1. All unit tests pass
2. DC solve matches analytical solutions to 1e-10 precision
3. Sparse matrices correctly accumulate duplicate stamps
4. Ground node (index 0) correctly skipped in all stamping
5. Differential variables correctly identified from C matrix
6. Code is documented and type-stable

## Next Steps

After Phase 1, proceed to Phase 2:
- Create device wrappers using these primitives
- Update `spectre.jl` codegen to emit MNA stamps
- Run existing test suite with new backend
