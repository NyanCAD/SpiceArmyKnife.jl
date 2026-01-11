#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Ring Oscillator with PSP103 MOSFETs
#
# 9-stage ring oscillator using PSP103 MOSFET model.
#
# Benchmark target: ~1 million timepoints
#
# Usage: julia runme.jl [solver]
#   solver: IDA, FBDF, or Rodas5P (default)
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarDCOp
using Sundials: IDA
using OrdinaryDiffEq: FBDF, Rodas5P
using BenchmarkTools
using Printf

# Import pre-parsed PSP103 model from PSPModels package
using PSPModels

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
# Pass PSP103VA_module so the SPICE parser knows about our VA device
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation()
    circuit = MNACircuit(ring_circuit)
    # Perform DC operating point to initialize the circuit
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(solver; dtmax=0.05e-9, maxiters=10_000_000)
    tspan = (0.0, 1e-6)  # 1us simulation (same as ngspice)
    solver_name = nameof(typeof(solver))

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # TODO: Ring oscillators need special initialization (no stable DC equilibrium)
    # CedarDCOp + use_shampine or CedarUICOp both timeout/OOM on this circuit
    # For now, use default CedarDCOp - the current pulse (i0) may help kick-start
    init = CedarDCOp()

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with $solver_name (dtmax=$dtmax)...")
    bench = @benchmark tran!($circuit, $tspan; dtmax=$dtmax, solver=$solver, initializealg=$init, maxiters=$maxiters, dense=false) samples=3 evals=1 seconds=300

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; dtmax=dtmax, solver=solver, initializealg=init, maxiters=maxiters, dense=false)

    println("\n=== Results ($solver_name) ===")
    @printf("Timepoints: %d\n", length(sol.t))
    @printf("NR iters:   %d\n", sol.stats.nnonliniter)
    @printf("Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    solver_name = length(ARGS) >= 1 ? ARGS[1] : "Rodas5P"
    solver = if solver_name == "IDA"
        IDA(max_nonlinear_iters=100, max_error_test_failures=20)
    elseif solver_name == "FBDF"
        FBDF()
    elseif solver_name == "Rodas5P"
        Rodas5P()
    else
        error("Unknown solver: $solver_name. Use IDA, FBDF, or Rodas5P")
    end
    run_benchmark(solver)
end
