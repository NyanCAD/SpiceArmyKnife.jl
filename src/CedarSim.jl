module CedarSim

using LinearAlgebra
using SparseArrays
using DiffEqBase
using OrdinaryDiffEq
using Printf

# MNA Core
include("mna.jl")
include("mna_devices.jl")

# Re-exports
export MNACircuit, MNANet, MNAODESystem
export resistor!, capacitor!, inductor!, vsource!, isource!, ground!, diode!
export vcvs!, vccs!
export solve_dc!, transient_problem, compile_circuit, compile_ode_system
export get_net!, get_branch!, node_index, branch_index
export dc!, tran!, get_voltage, get_time

#=
Simulation Interface
=#

"""
    DCResult

Result of DC operating point analysis.
"""
struct DCResult
    circuit::MNACircuit
    solution::Vector{Float64}
    node_names::Vector{Symbol}
    branch_names::Vector{Symbol}
end

function Base.show(io::IO, result::DCResult)
    println(io, "DC Operating Point:")
    println(io, "  Node Voltages:")
    for (i, (name, net)) in enumerate(result.circuit.nets)
        if net.index > 0
            @printf(io, "    V(%s) = %.6g V\n", name, result.solution[net.index])
        end
    end
    if !isempty(result.circuit.branches)
        println(io, "  Branch Currents:")
        for (name, branch) in result.circuit.branches
            idx = result.circuit.num_nodes + branch.index
            @printf(io, "    I(%s) = %.6g A\n", name, result.solution[idx])
        end
    end
end

"""
    dc!(circuit::MNACircuit; kwargs...) -> DCResult

Perform DC operating point analysis.
"""
function dc!(circuit::MNACircuit; kwargs...)
    solution = solve_dc!(circuit; kwargs...)
    node_names = [name for (name, net) in circuit.nets if net.index > 0]
    branch_names = [name for (name, branch) in circuit.branches]
    return DCResult(circuit, solution, node_names, branch_names)
end

"""
    TransientResult

Result of transient simulation.
"""
struct TransientResult
    circuit::MNACircuit
    solution::Any  # ODE solution
    sys::MNAODESystem
end

function Base.show(io::IO, result::TransientResult)
    println(io, "Transient Analysis:")
    println(io, "  Time span: ", result.solution.t[1], " to ", result.solution.t[end])
    println(io, "  Time points: ", length(result.solution.t))
end

"""
    get_voltage(result::TransientResult, node::Symbol)

Get voltage waveform for a node.
"""
function get_voltage(result::TransientResult, node::Symbol)
    net = result.circuit.nets[node]
    if net.index == 0
        return zeros(length(result.solution.t))
    end
    return [u[net.index] for u in result.solution.u]
end

"""
    get_time(result::TransientResult)

Get time vector from transient result.
"""
get_time(result::TransientResult) = result.solution.t

"""
    tran!(circuit::MNACircuit, tspan; u0=nothing, solver=nothing, kwargs...)

Perform transient simulation.
"""
function tran!(circuit::MNACircuit, tspan; u0=nothing, solver=nothing, kwargs...)
    f!, u0_calc, tspan_calc, mass_matrix, sys = transient_problem(circuit, tspan; u0=u0)

    # Choose solver
    if solver === nothing
        if mass_matrix !== nothing && !isdiag(mass_matrix)
            # DAE with mass matrix - use Rodas5 or similar
            solver = Rodas5()
        else
            # ODE - use TRBDF2
            solver = TRBDF2()
        end
    end

    # Create ODE problem
    if mass_matrix !== nothing
        prob = ODEProblem(
            ODEFunction(f!; mass_matrix=mass_matrix),
            u0_calc, tspan_calc;
            kwargs...
        )
    else
        prob = ODEProblem(f!, u0_calc, tspan_calc; kwargs...)
    end

    # Solve
    sol = solve(prob, solver)

    return TransientResult(circuit, sol, sys)
end

#=
Convenience circuit building functions
=#

"""
    @circuit(block)

Macro for building circuits with a DSL.
"""
macro circuit(block)
    # Simple circuit builder DSL
    quote
        circuit = MNACircuit()
        $(esc(block))
        circuit
    end
end

#=
Precompilation
=#

using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # Simple RC circuit
        circuit = MNACircuit()
        vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V1)
        resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
        capacitor!(circuit, :out, :gnd, 1e-6; name=:C1)
        ground!(circuit, :gnd)

        # DC solve
        result = dc!(circuit)
    end
end

end # module CedarSim
