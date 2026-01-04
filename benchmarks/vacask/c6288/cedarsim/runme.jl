#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: C6288 16x16 Multiplier
#
# A 16x16 bit multiplier circuit using PSP103 MOSFETs.
#
# Benchmark target: High complexity digital circuit
#
# Note: Uses Sundials IDA (variable-order BDF) with adaptive stepping.
# IDA uses our explicit Jacobian for optimal performance. reltol is tuned
# to match VACASK's tran_lteratio=1.5 for ~1000 timepoints.
#==============================================================================#

using CedarSim
using CedarSim.MNA
using Sundials
using BenchmarkTools
using Printf
using VerilogAParser
using DiffEqBase: BrownFullBasicInit

# Load the PSP103 model
const psp103_path = joinpath(@__DIR__, "..", "..", "..", "..", "test", "vadistiller", "models", "psp103v4", "psp103.va")

if isfile(psp103_path)
    println("Loading PSP103 from: ", psp103_path)
    va = VerilogAParser.parsefile(psp103_path)
    if !va.ps.errored
        Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
        println("PSP103VA_module loaded successfully")
    else
        error("Failed to parse PSP103 VA model")
    end
else
    error("PSP103 VA model not found at $psp103_path")
end

# Load and parse the SPICE netlist from file
# Use parse_spice_file_to_mna to preserve file path context for .include resolution
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

function run_benchmark(; reltol=1e-3)
    tspan = (0.0, 2e-9)  # 2ns simulation (same as ngspice)

    # Use Sundials IDA (variable-order BDF) with adaptive stepping.
    # IDA uses our explicit Jacobian for optimal performance.
    # reltol is tuned to match VACASK's tran_lteratio=1.5 (~1000 timepoints)
    solver = IDA(max_nonlinear_iters=100, max_error_test_failures=20)

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Use BrownFullBasicInit with relaxed tolerance for DAE initialization
    init = BrownFullBasicInit(abstol=1e-3)

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with IDA (reltol=$reltol)...")
    bench = @benchmark tran!($circuit, $tspan; solver=$solver, reltol=$reltol, initializealg=$init) samples=6 evals=1 seconds=600

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; solver=solver, reltol=reltol, initializealg=init)

    println("\n=== Results ===")
    @printf("Timepoints: %d\n", length(sol.t))
    @printf("NR iters:   %d\n", sol.stats.nnonliniter)
    @printf("Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
