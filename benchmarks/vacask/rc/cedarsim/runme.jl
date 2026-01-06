#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: RC Circuit
#
# RC circuit excited by a pulse train.
# This is a simple linear circuit - no rejected timepoints.
#
# Benchmark target: ~1 million timepoints, ~2 million iterations
#
# Usage: julia runme.jl [solver]
#   solver: IDA (default), FBDF, or Rodas5P
#==============================================================================#

using CedarSim
using CedarSim.MNA
using Sundials: IDA
using OrdinaryDiffEq: FBDF, Rodas5P
using BenchmarkTools
using Printf

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation()
    circuit = MNACircuit(rc_circuit)
    # Perform DC operating point to initialize the circuit
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(solver; dt=1e-6, maxiters=10_000_000)
    tspan = (0.0, 1.0)  # 1 second simulation
    solver_name = nameof(typeof(solver))

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with $solver_name (dtmax=$dt)...")
    bench = @benchmark tran!($circuit, $tspan; dtmax=$dt, solver=$solver, maxiters=$maxiters, dense=false) samples=3 evals=1 seconds=300

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; dtmax=dt, solver=solver, maxiters=maxiters, dense=false)

    println("\n=== Results ($solver_name) ===")
    @printf("Timepoints:  %d\n", length(sol.t))
    @printf("Expected:    ~%d\n", round(Int, (tspan[2] - tspan[1]) / dt) + 1)
    @printf("NR iters:    %d\n", sol.stats.nnonliniter)
    @printf("Iter/step:   %.2f\n", sol.stats.nnonliniter / length(sol.t))
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    solver_name = length(ARGS) >= 1 ? ARGS[1] : "IDA"
    solver = if solver_name == "IDA"
        IDA(max_error_test_failures=20)
    elseif solver_name == "FBDF"
        FBDF()
    elseif solver_name == "Rodas5P"
        Rodas5P()
    else
        error("Unknown solver: $solver_name. Use IDA, FBDF, or Rodas5P")
    end
    run_benchmark(solver)
end
