# Modified Nodal Analysis (MNA) Implementation
# Replaces DAECompiler with direct matrix-based circuit analysis

using LinearAlgebra
using SparseArrays

export MNACircuit, MNANet, stamp!, compile_circuit, solve_dc!, transient_problem

#=
MNA Matrix Structure:
    [G  B] [v]   [i]
    [C  D] [j] = [e]

Where:
- G: n×n conductance matrix (from resistors)
- B: n×m voltage source connection matrix
- C: m×n (transpose of B for independent sources)
- D: m×m (zero for independent sources)
- v: node voltages (unknowns)
- j: voltage source currents (unknowns)
- i: current source contributions
- e: voltage source values

For dynamic elements (capacitors/inductors):
    C_dyn * dx/dt + G * x = b(t)

Where C_dyn is the dynamic matrix (capacitances on voltage nodes,
inductances require additional current variables).
=#

"""
    MNANet

Represents a circuit node in the MNA formulation.
Each net has an index into the solution vector.
"""
mutable struct MNANet
    name::Symbol
    index::Int  # 0 = ground, >0 = internal node
end

MNANet(name::Symbol) = MNANet(name, -1)  # -1 means unassigned
MNANet(name::String) = MNANet(Symbol(name))

"""
    BranchVar

Represents a branch current variable (for voltage sources, inductors).
"""
mutable struct BranchVar
    name::Symbol
    index::Int  # index into the branch current section of solution vector
end

BranchVar(name::Symbol) = BranchVar(name, -1)

"""
    MNACircuit

Container for circuit topology and MNA matrices.
"""
mutable struct MNACircuit
    # Node management
    nets::Dict{Symbol, MNANet}
    ground::MNANet
    num_nodes::Int  # excludes ground

    # Branch current management (for voltage sources, inductors)
    branches::Dict{Symbol, BranchVar}
    num_branches::Int

    # Linear part of MNA matrix (constant coefficients)
    # Stored in COO format for assembly, converted to sparse later
    G_i::Vector{Int}
    G_j::Vector{Int}
    G_v::Vector{Float64}

    # Dynamic matrix (capacitances)
    C_i::Vector{Int}
    C_j::Vector{Int}
    C_v::Vector{Float64}

    # Source vector contributions (linear)
    b_i::Vector{Int}
    b_v::Vector{Float64}

    # Nonlinear element callbacks
    # Each callback: (x, dx, t, params) -> (residual_contributions, jacobian_contributions)
    nonlinear_elements::Vector{Any}

    # Time-dependent source callbacks
    # Each callback: (t, params) -> source_contributions
    time_sources::Vector{Any}

    # Simulation parameters
    params::Dict{Symbol, Any}

    # Mode: :dcop, :tran, :ac
    mode::Symbol

    # Temperature (Celsius)
    temp::Float64

    # gmin (minimum conductance for convergence)
    gmin::Float64
end

function MNACircuit(;temp=27.0, gmin=1e-12)
    ground = MNANet(:ground, 0)
    MNACircuit(
        Dict{Symbol, MNANet}(:ground => ground, Symbol("0") => ground),
        ground,
        0,
        Dict{Symbol, BranchVar}(),
        0,
        Int[], Int[], Float64[],  # G matrix COO
        Int[], Int[], Float64[],  # C matrix COO
        Int[], Float64[],          # b vector
        Any[],                     # nonlinear elements
        Any[],                     # time sources
        Dict{Symbol, Any}(),       # params
        :dcop,                     # mode
        temp,                      # temperature
        gmin                       # gmin
    )
end

"""
    get_net!(circuit::MNACircuit, name::Symbol) -> MNANet

Get or create a net with the given name.
"""
function get_net!(circuit::MNACircuit, name::Symbol)
    if haskey(circuit.nets, name)
        return circuit.nets[name]
    end
    circuit.num_nodes += 1
    net = MNANet(name, circuit.num_nodes)
    circuit.nets[name] = net
    return net
end

get_net!(circuit::MNACircuit, name::String) = get_net!(circuit, Symbol(name))
get_net!(circuit::MNACircuit, net::MNANet) = net

"""
    get_branch!(circuit::MNACircuit, name::Symbol) -> BranchVar

Get or create a branch current variable.
"""
function get_branch!(circuit::MNACircuit, name::Symbol)
    if haskey(circuit.branches, name)
        return circuit.branches[name]
    end
    circuit.num_branches += 1
    branch = BranchVar(name, circuit.num_branches)
    circuit.branches[name] = branch
    return branch
end

"""
Node index in the solution vector (0 for ground).
"""
node_index(net::MNANet) = net.index

"""
Branch index placeholder (negative to mark as branch during stamping).
The actual index is resolved in compile_circuit to num_nodes + |index|.
"""
branch_index(circuit::MNACircuit, branch::BranchVar) = -branch.index

"""
Resolve an index that may be a branch placeholder (negative) to final index.
"""
resolve_index(circuit::MNACircuit, idx::Int) = idx > 0 ? idx : circuit.num_nodes - idx

#=
Stamping functions for various circuit elements.
These add contributions to the MNA matrices.
=#

"""
    stamp_conductance!(circuit, n1, n2, g)

Stamp a conductance g between nodes n1 and n2 into the G matrix.
"""
function stamp_conductance!(circuit::MNACircuit, n1::MNANet, n2::MNANet, g::Float64)
    i1, i2 = node_index(n1), node_index(n2)

    # Diagonal terms (positive)
    if i1 > 0
        push!(circuit.G_i, i1)
        push!(circuit.G_j, i1)
        push!(circuit.G_v, g)
    end
    if i2 > 0
        push!(circuit.G_i, i2)
        push!(circuit.G_j, i2)
        push!(circuit.G_v, g)
    end

    # Off-diagonal terms (negative)
    if i1 > 0 && i2 > 0
        push!(circuit.G_i, i1)
        push!(circuit.G_j, i2)
        push!(circuit.G_v, -g)

        push!(circuit.G_i, i2)
        push!(circuit.G_j, i1)
        push!(circuit.G_v, -g)
    end
end

"""
    stamp_capacitance!(circuit, n1, n2, c)

Stamp a capacitance c between nodes n1 and n2 into the C matrix.
"""
function stamp_capacitance!(circuit::MNACircuit, n1::MNANet, n2::MNANet, c::Float64)
    i1, i2 = node_index(n1), node_index(n2)

    # Diagonal terms (positive)
    if i1 > 0
        push!(circuit.C_i, i1)
        push!(circuit.C_j, i1)
        push!(circuit.C_v, c)
    end
    if i2 > 0
        push!(circuit.C_i, i2)
        push!(circuit.C_j, i2)
        push!(circuit.C_v, c)
    end

    # Off-diagonal terms (negative)
    if i1 > 0 && i2 > 0
        push!(circuit.C_i, i1)
        push!(circuit.C_j, i2)
        push!(circuit.C_v, -c)

        push!(circuit.C_i, i2)
        push!(circuit.C_j, i1)
        push!(circuit.C_v, -c)
    end
end

"""
    stamp_voltage_source!(circuit, n_pos, n_neg, branch, voltage)

Stamp a voltage source between n_pos and n_neg with the given branch current variable.
"""
function stamp_voltage_source!(circuit::MNACircuit, n_pos::MNANet, n_neg::MNANet,
                                branch::BranchVar, voltage::Float64)
    i_pos = node_index(n_pos)
    i_neg = node_index(n_neg)
    i_br = branch_index(circuit, branch)

    # B matrix: connection of branch current to KCL equations
    # Current flows from pos to neg, so +I at neg, -I at pos
    if i_pos > 0
        push!(circuit.G_i, i_pos)
        push!(circuit.G_j, i_br)
        push!(circuit.G_v, 1.0)
    end
    if i_neg > 0
        push!(circuit.G_i, i_neg)
        push!(circuit.G_j, i_br)
        push!(circuit.G_v, -1.0)
    end

    # C matrix: voltage constraint V_pos - V_neg = voltage
    if i_pos > 0
        push!(circuit.G_i, i_br)
        push!(circuit.G_j, i_pos)
        push!(circuit.G_v, 1.0)
    end
    if i_neg > 0
        push!(circuit.G_i, i_br)
        push!(circuit.G_j, i_neg)
        push!(circuit.G_v, -1.0)
    end

    # Source value
    push!(circuit.b_i, i_br)
    push!(circuit.b_v, voltage)
end

"""
    stamp_current_source!(circuit, n_pos, n_neg, current)

Stamp a current source from n_pos to n_neg (current flows from pos to neg).
"""
function stamp_current_source!(circuit::MNACircuit, n_pos::MNANet, n_neg::MNANet, current::Float64)
    i_pos = node_index(n_pos)
    i_neg = node_index(n_neg)

    # Current enters at neg, exits at pos
    if i_pos > 0
        push!(circuit.b_i, i_pos)
        push!(circuit.b_v, -current)
    end
    if i_neg > 0
        push!(circuit.b_i, i_neg)
        push!(circuit.b_v, current)
    end
end

"""
    stamp_inductor!(circuit, n_pos, n_neg, branch, inductance)

Stamp an inductor between n_pos and n_neg.
The inductor equation is: V = L * dI/dt
This requires a branch current variable.
"""
function stamp_inductor!(circuit::MNACircuit, n_pos::MNANet, n_neg::MNANet,
                         branch::BranchVar, inductance::Float64)
    i_pos = node_index(n_pos)
    i_neg = node_index(n_neg)
    i_br = branch_index(circuit, branch)

    # KCL: branch current enters pos, exits neg
    if i_pos > 0
        push!(circuit.G_i, i_pos)
        push!(circuit.G_j, i_br)
        push!(circuit.G_v, 1.0)
    end
    if i_neg > 0
        push!(circuit.G_i, i_neg)
        push!(circuit.G_j, i_br)
        push!(circuit.G_v, -1.0)
    end

    # Inductor equation: V_pos - V_neg - L * dI/dt = 0
    # Rearranged for MNA: V_pos - V_neg = L * dI/dt
    # In the C matrix (dynamic part): L * dI_br/dt
    if i_pos > 0
        push!(circuit.G_i, i_br)
        push!(circuit.G_j, i_pos)
        push!(circuit.G_v, 1.0)
    end
    if i_neg > 0
        push!(circuit.G_i, i_br)
        push!(circuit.G_j, i_neg)
        push!(circuit.G_v, -1.0)
    end

    # Dynamic term: -L * dI/dt (moved to LHS as part of C*dx/dt)
    push!(circuit.C_i, i_br)
    push!(circuit.C_j, i_br)
    push!(circuit.C_v, -inductance)
end

"""
    stamp_vcvs!(circuit, n_pos, n_neg, nc_pos, nc_neg, branch, gain)

Voltage-Controlled Voltage Source: V(n_pos, n_neg) = gain * V(nc_pos, nc_neg)
"""
function stamp_vcvs!(circuit::MNACircuit, n_pos::MNANet, n_neg::MNANet,
                     nc_pos::MNANet, nc_neg::MNANet, branch::BranchVar, gain::Float64)
    i_pos = node_index(n_pos)
    i_neg = node_index(n_neg)
    ic_pos = node_index(nc_pos)
    ic_neg = node_index(nc_neg)
    i_br = branch_index(circuit, branch)

    # B matrix entries
    if i_pos > 0
        push!(circuit.G_i, i_pos)
        push!(circuit.G_j, i_br)
        push!(circuit.G_v, 1.0)
    end
    if i_neg > 0
        push!(circuit.G_i, i_neg)
        push!(circuit.G_j, i_br)
        push!(circuit.G_v, -1.0)
    end

    # Constraint: V_pos - V_neg - gain*(Vc_pos - Vc_neg) = 0
    if i_pos > 0
        push!(circuit.G_i, i_br)
        push!(circuit.G_j, i_pos)
        push!(circuit.G_v, 1.0)
    end
    if i_neg > 0
        push!(circuit.G_i, i_br)
        push!(circuit.G_j, i_neg)
        push!(circuit.G_v, -1.0)
    end
    if ic_pos > 0
        push!(circuit.G_i, i_br)
        push!(circuit.G_j, ic_pos)
        push!(circuit.G_v, -gain)
    end
    if ic_neg > 0
        push!(circuit.G_i, i_br)
        push!(circuit.G_j, ic_neg)
        push!(circuit.G_v, gain)
    end
end

"""
    stamp_vccs!(circuit, n_pos, n_neg, nc_pos, nc_neg, gain)

Voltage-Controlled Current Source: I(n_pos -> n_neg) = gain * V(nc_pos, nc_neg)
"""
function stamp_vccs!(circuit::MNACircuit, n_pos::MNANet, n_neg::MNANet,
                     nc_pos::MNANet, nc_neg::MNANet, gain::Float64)
    i_pos = node_index(n_pos)
    i_neg = node_index(n_neg)
    ic_pos = node_index(nc_pos)
    ic_neg = node_index(nc_neg)

    # Current into n_neg, out of n_pos, proportional to control voltage
    if i_pos > 0 && ic_pos > 0
        push!(circuit.G_i, i_pos)
        push!(circuit.G_j, ic_pos)
        push!(circuit.G_v, -gain)
    end
    if i_pos > 0 && ic_neg > 0
        push!(circuit.G_i, i_pos)
        push!(circuit.G_j, ic_neg)
        push!(circuit.G_v, gain)
    end
    if i_neg > 0 && ic_pos > 0
        push!(circuit.G_i, i_neg)
        push!(circuit.G_j, ic_pos)
        push!(circuit.G_v, gain)
    end
    if i_neg > 0 && ic_neg > 0
        push!(circuit.G_i, i_neg)
        push!(circuit.G_j, ic_neg)
        push!(circuit.G_v, -gain)
    end
end

"""
    add_nonlinear_element!(circuit, element_func)

Add a nonlinear element to the circuit.
element_func(x, params) should return (residual, jacobian_entries)
where jacobian_entries is a vector of (i, j, value) tuples.
"""
function add_nonlinear_element!(circuit::MNACircuit, element_func)
    push!(circuit.nonlinear_elements, element_func)
end

"""
    add_time_source!(circuit, source_func)

Add a time-dependent source to the circuit.
source_func(t, params) should return a vector of (index, value) tuples.
"""
function add_time_source!(circuit::MNACircuit, source_func)
    push!(circuit.time_sources, source_func)
end

#=
Circuit compilation and solving
=#

"""
    compile_circuit(circuit::MNACircuit)

Compile the circuit into sparse matrices ready for solving.
Returns (G, C, b) where G*x + C*dx/dt = b for the linear part.
"""
function compile_circuit(circuit::MNACircuit)
    n = circuit.num_nodes + circuit.num_branches

    # Resolve branch indices (negative placeholders become num_nodes + |index|)
    G_i_resolved = [resolve_index(circuit, i) for i in circuit.G_i]
    G_j_resolved = [resolve_index(circuit, j) for j in circuit.G_j]

    # Build sparse G matrix
    G = sparse(G_i_resolved, G_j_resolved, circuit.G_v, n, n)

    # Build sparse C matrix (for dynamics)
    if isempty(circuit.C_i)
        C = spzeros(n, n)
    else
        C_i_resolved = [resolve_index(circuit, i) for i in circuit.C_i]
        C_j_resolved = [resolve_index(circuit, j) for j in circuit.C_j]
        C = sparse(C_i_resolved, C_j_resolved, circuit.C_v, n, n)
    end

    # Build source vector with resolved indices
    b = zeros(n)
    for (i, v) in zip(circuit.b_i, circuit.b_v)
        b[resolve_index(circuit, i)] += v
    end

    return G, C, b
end

"""
    solve_dc!(circuit::MNACircuit; maxiter=100, tol=1e-9)

Solve for DC operating point using Newton-Raphson iteration.
Returns the solution vector.
"""
function solve_dc!(circuit::MNACircuit; maxiter=100, tol=1e-9)
    circuit.mode = :dcop
    G, C, b = compile_circuit(circuit)
    n = size(G, 1)

    # Initial guess
    x = zeros(n)

    # Add gmin to node equations only (not branch equations)
    # gmin provides a tiny conductance to ground for numerical stability
    for i in 1:circuit.num_nodes
        G[i, i] += circuit.gmin
    end

    if isempty(circuit.nonlinear_elements)
        # Pure linear circuit - direct solve
        x = G \ b
    else
        # Newton-Raphson for nonlinear circuits
        for iter in 1:maxiter
            # Evaluate nonlinear contributions
            f = copy(b)
            J = copy(G)

            for elem_func in circuit.nonlinear_elements
                residual, jac_entries = elem_func(x, circuit.params, circuit)
                f .-= residual
                for (i, j, v) in jac_entries
                    J[i, j] += v
                end
            end

            # Residual
            r = J * x - f

            # Check convergence
            if norm(r) < tol
                return x
            end

            # Newton update
            dx = J \ r
            x .-= dx

            if norm(dx) < tol
                return x
            end
        end

        @warn "DC operating point did not converge after $maxiter iterations"
    end

    return x
end

"""
    DiodeModel

Parameters for the SPICE diode model.
"""
Base.@kwdef struct DiodeModel
    IS::Float64 = 1e-14      # Saturation current
    N::Float64 = 1.0         # Emission coefficient
    RS::Float64 = 0.0        # Series resistance
    BV::Float64 = Inf        # Breakdown voltage
    IBV::Float64 = 1e-3      # Current at breakdown
    CJO::Float64 = 0.0       # Zero-bias junction capacitance
    VJ::Float64 = 1.0        # Junction potential
    M::Float64 = 0.5         # Grading coefficient
    TT::Float64 = 0.0        # Transit time
    AREA::Float64 = 1.0      # Device area multiplier
end

"""
    stamp_diode!(circuit, n_anode, n_cathode, model, name)

Stamp a diode as a nonlinear element.
"""
function stamp_diode!(circuit::MNACircuit, n_anode::MNANet, n_cathode::MNANet,
                      model::DiodeModel, name::Symbol)
    i_a = node_index(n_anode)
    i_c = node_index(n_cathode)

    # Physical constants
    q = 1.602176634e-19  # electron charge
    k = 1.380649e-23     # Boltzmann constant
    T = circuit.temp + 273.15  # Temperature in Kelvin
    Vt = k * T / q       # Thermal voltage

    # Add nonlinear element function
    add_nonlinear_element!(circuit, (x, params, ckt) -> begin
        # Get voltage across diode
        Va = i_a > 0 ? x[i_a] : 0.0
        Vc = i_c > 0 ? x[i_c] : 0.0
        Vd = Va - Vc

        # Diode current (Shockley equation with limiting)
        IS = model.IS * model.AREA
        Nvt = model.N * Vt

        # Limit voltage for numerical stability
        Vd_crit = Nvt * log(Nvt / (sqrt(2) * IS))
        if Vd > Vd_crit
            Vd_eff = Vd_crit + Nvt * log(1 + (Vd - Vd_crit) / Nvt)
        else
            Vd_eff = Vd
        end

        # Forward current
        if Vd_eff > -5 * Nvt
            Id = IS * (exp(Vd_eff / Nvt) - 1)
            Gd = IS / Nvt * exp(Vd_eff / Nvt)  # conductance (dI/dV)
        else
            Id = -IS
            Gd = 0.0
        end

        # Add gmin for convergence
        Id += ckt.gmin * Vd
        Gd += ckt.gmin

        # Residual contribution: I flows from anode to cathode
        residual = zeros(length(x))
        if i_a > 0
            residual[i_a] = Id
        end
        if i_c > 0
            residual[i_c] = -Id
        end

        # Jacobian entries
        jac_entries = Tuple{Int, Int, Float64}[]
        if i_a > 0
            push!(jac_entries, (i_a, i_a, Gd))
            if i_c > 0
                push!(jac_entries, (i_a, i_c, -Gd))
            end
        end
        if i_c > 0
            push!(jac_entries, (i_c, i_c, Gd))
            if i_a > 0
                push!(jac_entries, (i_c, i_a, -Gd))
            end
        end

        return (residual, jac_entries)
    end)
end

#=
ODE System generation for DifferentialEquations.jl
=#

"""
    MNAODESystem

Compiled MNA system ready for time-domain simulation.
"""
struct MNAODESystem
    # Compiled matrices
    G::SparseMatrixCSC{Float64, Int}
    C::SparseMatrixCSC{Float64, Int}
    b::Vector{Float64}

    # Circuit reference for nonlinear evaluation
    circuit::MNACircuit

    # System size
    n::Int

    # Index of algebraic variables (where C has zero diagonal)
    differential_vars::BitVector

    # Mass matrix for DAE form
    mass_matrix::SparseMatrixCSC{Float64, Int}
end

"""
    compile_ode_system(circuit::MNACircuit)

Compile circuit into an ODE/DAE system for DifferentialEquations.jl
"""
function compile_ode_system(circuit::MNACircuit)
    G, C, b = compile_circuit(circuit)
    n = size(G, 1)

    # Add gmin to node equations only (not branch constraint equations)
    for i in 1:circuit.num_nodes
        G[i, i] += circuit.gmin
    end

    # Determine which variables are differential vs algebraic
    differential_vars = BitVector(undef, n)
    for i in 1:n
        # A variable is differential if it has a nonzero entry in C
        differential_vars[i] = any(j -> C[i, j] != 0 || C[j, i] != 0, 1:n)
    end

    # Mass matrix is the C matrix (for DAE form: M * du/dt = f(u))
    mass_matrix = C

    return MNAODESystem(G, C, b, circuit, n, differential_vars, mass_matrix)
end

"""
    (sys::MNAODESystem)(du, u, p, t)

ODE right-hand side function for DifferentialEquations.jl
Computes: C * du/dt = b(t) - G * u - f_nonlinear(u)
Rearranged: du/dt = C^{-1} * (b(t) - G * u - f_nonlinear(u))

For DAE form, this computes the residual: C * du - (b - G*u - f_nl) = 0
"""
function (sys::MNAODESystem)(du, u, p, t)
    # Start with linear contribution: -G * u + b
    rhs = sys.b - sys.G * u

    # Add time-dependent sources
    for source_func in sys.circuit.time_sources
        contributions = source_func(t, p)
        for (i, v) in contributions
            rhs[i] += v
        end
    end

    # Add nonlinear contributions
    for elem_func in sys.circuit.nonlinear_elements
        residual, _ = elem_func(u, p, sys.circuit)
        rhs .-= residual
    end

    # For differential equations: C * du/dt = rhs
    # We need to solve for du/dt
    # For now, handle the simple case where C is diagonal or use DAE form

    # Use the DAE residual form: 0 = C * du - rhs
    # DifferentialEquations.jl will handle this with mass matrix
    du .= rhs
end

"""
    make_ode_function(circuit::MNACircuit)

Create an ODEFunction compatible with DifferentialEquations.jl
"""
function make_ode_function(circuit::MNACircuit)
    sys = compile_ode_system(circuit)

    # Create the ODE function
    function f!(du, u, p, t)
        sys(du, u, p, t)
    end

    return f!, sys
end

"""
    transient_problem(circuit::MNACircuit, tspan; u0=nothing)

Create an ODEProblem for transient simulation.
"""
function transient_problem(circuit::MNACircuit, tspan; u0=nothing)
    circuit.mode = :tran
    f!, sys = make_ode_function(circuit)

    # Get initial conditions from DC operating point if not provided
    if u0 === nothing
        circuit.mode = :dcop
        u0 = solve_dc!(circuit)
        circuit.mode = :tran
    end

    # For circuits with capacitors/inductors, we need a DAE formulation
    # Check if we have dynamic elements
    has_dynamics = !isempty(circuit.C_i)

    if has_dynamics
        # Use mass matrix formulation: M * du/dt = f(u, t)
        # where M = C (the capacitance matrix)
        mass_matrix = sys.mass_matrix

        # For singular mass matrix (algebraic constraints), use DAE solver
        return (f!, u0, tspan, mass_matrix, sys)
    else
        # Pure algebraic system (resistive circuit)
        return (f!, u0, tspan, nothing, sys)
    end
end
