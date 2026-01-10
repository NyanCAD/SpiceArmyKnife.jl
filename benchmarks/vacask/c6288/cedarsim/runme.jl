#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: C6288 16x16 Multiplier
#
# A 16x16 bit multiplier circuit using PSP103 MOSFETs.
#
# Benchmark target: High complexity digital circuit (154k variables)
#
# STATUS: Uses CedarUICOp initialization (pseudo-transient relaxation)
#
# Notes:
# Digital circuits often don't have a valid DC solution. ngspice handles
# this with 'uic' (use initial conditions). CedarUICOp provides similar
# functionality using pseudo-transient relaxation.
#
# Usage: julia runme.jl [solver]
#   solver: IDA, FBDF, or Rodas5P (default)
#==============================================================================#

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: CedarUICOp
using Sundials: IDA
using OrdinaryDiffEq: FBDF, Rodas5P
using BenchmarkTools
using Printf

# Import pre-parsed PSP103 model from PSPModels package
using PSPModels

# Alias for compatibility with existing code
const PSP103VA_module = sp_psp103va_module
println("PSP103VA loaded from PSPModels package")

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")

# Parse SPICE file to code, then evaluate to get the builder function
const circuit_code = parse_spice_file_to_mna(spice_file; circuit_name=:c6288_circuit,
                                              imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
"""
function setup_simulation()
    circuit = MNACircuit(c6288_circuit)
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(solver; reltol=1e-3, maxiters=10_000_000)
    tspan = (0.0, 2e-9)  # 2ns simulation (same as ngspice)
    solver_name = nameof(typeof(solver))

    # Setup the simulation outside the timed region
    circuit = setup_simulation()
    n = MNA.system_size(circuit)
    println("Circuit size: $n variables")

    # Use CedarUICOp for initialization - digital circuits often don't have
    # a valid DC solution. ngspice handles this with 'uic' (use initial conditions)
    # CedarUICOp uses pseudo-transient relaxation to initialize
    init = CedarUICOp()

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with $solver_name (reltol=$reltol)...")
    bench = @benchmark tran!($circuit, $tspan; solver=$solver, reltol=$reltol, maxiters=$maxiters, initializealg=$init, dense=false) samples=3 evals=1 seconds=300

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; solver=solver, reltol=reltol, maxiters=maxiters, initializealg=init, dense=false)

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
