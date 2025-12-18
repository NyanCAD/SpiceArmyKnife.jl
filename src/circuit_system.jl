# Circuit System - Clean SciML-compatible API
# Provides named solution access via SymbolicIndexingInterface's SymbolCache

"""
    CircuitSystem

Wrapper around a compiled circuit that provides SciML-compatible
symbolic indexing. This allows solution access by name:

    sol[:vcc]       # Node voltage
    sol[:R1_I]      # Branch current (voltage source current)

Uses SymbolCache from SymbolicIndexingInterface for automatic implementation
of all required interface methods.
"""
struct CircuitSystem{C, SC}
    circuit::C
    symbol_cache::SC

    # Keep track of which symbols are nodes vs branches for convenience
    node_names::Set{Symbol}
    branch_names::Set{Symbol}
end

"""
    CircuitSystem(circuit::MNACircuit)

Create a CircuitSystem from an MNACircuit for SciML integration.
"""
function CircuitSystem(circuit::MNACircuit)
    # Build variable index dictionary
    variables = Dict{Symbol, Int}()
    node_names = Set{Symbol}()
    branch_names = Set{Symbol}()

    # Add node voltages
    for (name, net) in circuit.nets
        if net.index > 0
            variables[name] = net.index
            push!(node_names, name)
        end
    end

    # Add branch currents (offset by num_nodes)
    for (name, branch) in circuit.branches
        full_idx = circuit.num_nodes + branch.index
        # Use _I suffix for branch currents to distinguish from node voltages
        branch_sym = Symbol(name, :_I)
        variables[branch_sym] = full_idx
        push!(branch_names, name)
        # Also allow direct branch name access
        variables[name] = full_idx
    end

    # Create SymbolCache - this provides all SymbolicIndexingInterface methods
    symbol_cache = SymbolicIndexingInterface.SymbolCache(
        variables,          # variables dict
        nothing,            # no parameters (handled separately via ParamLens)
        [:t]                # independent variable (time)
    )

    CircuitSystem(circuit, symbol_cache, node_names, branch_names)
end

# ============================================================
# Forward SymbolicIndexingInterface methods to SymbolCache
# ============================================================

# The symbolic_container function tells SII to use our symbol_cache
SymbolicIndexingInterface.symbolic_container(sys::CircuitSystem) = sys.symbol_cache

# ============================================================
# Convenience functions
# ============================================================

"""
    voltage(sys::CircuitSystem, node::Symbol)

Get the index for a node voltage in the solution vector.
"""
function voltage(sys::CircuitSystem, node::Symbol)
    if node in sys.node_names
        return SymbolicIndexingInterface.variable_index(sys.symbol_cache, node)
    end
    nothing
end

"""
    current(sys::CircuitSystem, branch::Symbol)

Get the index for a branch current in the solution vector.
"""
function current(sys::CircuitSystem, branch::Symbol)
    if branch in sys.branch_names
        # Try direct branch name
        return SymbolicIndexingInterface.variable_index(sys.symbol_cache, branch)
    end
    nothing
end

"""
    node_voltage(sys::CircuitSystem, sol, node::Symbol)

Get voltage value for a node from a solution.
"""
function node_voltage(sys::CircuitSystem, sol, node::Symbol)
    idx = voltage(sys, node)
    idx === nothing && error("Node $node not found in circuit")
    if sol isa AbstractVector
        return sol[idx]
    else
        # Assume ODE solution
        return [u[idx] for u in sol.u]
    end
end

"""
    branch_current(sys::CircuitSystem, sol, branch::Symbol)

Get current value for a branch from a solution.
"""
function branch_current(sys::CircuitSystem, sol, branch::Symbol)
    idx = current(sys, branch)
    idx === nothing && error("Branch $branch not found in circuit")
    if sol isa AbstractVector
        return sol[idx]
    else
        return [u[idx] for u in sol.u]
    end
end

# ============================================================
# ODEFunction creation with symbolic indexing
# ============================================================

"""
    create_ode_function(sys::CircuitSystem; kwargs...)

Create an ODEFunction with symbolic indexing support.
"""
function create_ode_function(sys::CircuitSystem; kwargs...)
    circuit = sys.circuit

    # Get the underlying ODE function
    f!, ode_sys = make_ode_function(circuit)

    # Create ODEFunction with our system for symbolic indexing
    DiffEqBase.ODEFunction(f!; sys=sys, kwargs...)
end

"""
    create_problem(sys::CircuitSystem, tspan; u0=nothing, solver_kwargs...)

Create an ODEProblem with symbolic indexing support.
"""
function create_problem(sys::CircuitSystem, tspan; u0=nothing, kwargs...)
    circuit = sys.circuit
    circuit.mode = :tran

    # Get initial conditions from DC operating point if not provided
    if u0 === nothing
        circuit.mode = :dcop
        u0 = solve_dc!(circuit)
        circuit.mode = :tran
    end

    # Initialize transient state
    circuit.time = tspan[1]
    circuit.prev_time = tspan[1]
    circuit.dt = 0.0
    initialize_charges!(circuit, u0)

    # Get ODE function components
    f!, ode_sys = make_ode_function(circuit)
    has_dynamics = !isempty(circuit.C_i)

    if has_dynamics
        DiffEqBase.ODEProblem(
            DiffEqBase.ODEFunction(f!; mass_matrix=ode_sys.mass_matrix, sys=sys),
            u0, tspan;
            kwargs...
        )
    else
        DiffEqBase.ODEProblem(
            DiffEqBase.ODEFunction(f!; sys=sys),
            u0, tspan;
            kwargs...
        )
    end
end

"""
    solve_tran(sys::CircuitSystem, tspan; u0=nothing, solver=nothing, kwargs...)

Convenience function to create and solve a transient problem with named solution access.
"""
function solve_tran(sys::CircuitSystem, tspan; u0=nothing, solver=nothing, kwargs...)
    prob = create_problem(sys, tspan; u0=u0)

    # Choose solver if not specified - use autodiff=false for compatibility
    if solver === nothing
        circuit = sys.circuit
        has_dynamics = !isempty(circuit.C_i)
        if has_dynamics
            # Use Rodas5 with autodiff disabled for DAE systems
            solver = OrdinaryDiffEq.Rodas5(autodiff=false)
        else
            solver = OrdinaryDiffEq.TRBDF2(autodiff=false)
        end
    end

    DiffEqBase.solve(prob, solver; kwargs...)
end
