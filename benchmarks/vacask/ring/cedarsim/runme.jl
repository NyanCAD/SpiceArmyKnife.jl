#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Ring Oscillator with PSP103 MOSFETs
#
# 9-stage ring oscillator using PSP103 MOSFET model.
#
# Benchmark target: ~20,000-30,000 timepoints (approximately matching VACASK)
# VACASK reference: 26,066 timepoints, 81,875 iterations, 1.18s (trapezoidal)
# Ngspice reference: 20,556 timepoints, 80,018 iterations, 1.60s
#
# Note: Uses Sundials IDA (variable-order BDF) with dtmax=0.05ns to enforce
# timestep constraint. Tolerances reltol=1e-3, abstol=1e-9 provide stable
# simulation with step counts comparable to VACASK/Ngspice.
# IDA uses our explicit Jacobian for optimal performance.
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

# Parse and eval the PSP103 model
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

function run_benchmark(; dtmax=0.05e-9, reltol=1e-3, abstol=1e-9)
    tspan = (0.0, 1e-6)  # 1us simulation (same as ngspice/VACASK)

    # Use Sundials IDA (variable-order BDF) with dtmax to enforce timestep constraint.
    # IDA uses our explicit Jacobian for optimal performance.
    # Tolerances tuned to provide stable simulation with step counts comparable to VACASK.
    # - reltol=1e-3 allows reasonable step sizes while maintaining accuracy
    # - abstol=1e-9 is tight enough to avoid convergence issues
    solver = IDA(max_nonlinear_iters=100, max_error_test_failures=20)

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Use BrownFullBasicInit with relaxed tolerance for DAE initialization
    # Ring oscillators don't have a stable equilibrium, so we need to accept
    # approximate initial conditions and let the current pulse kick-start oscillation
    init = BrownFullBasicInit(abstol=1e-3)

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with IDA...")
    println("  dtmax=$dtmax, reltol=$reltol, abstol=$abstol")
    bench = @benchmark tran!($circuit, $tspan; dtmax=$dtmax, reltol=$reltol, abstol=$abstol,
                             solver=$solver, initializealg=$init) samples=6 evals=1 seconds=600

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; dtmax=dtmax, reltol=reltol, abstol=abstol,
                solver=solver, initializealg=init)

    # VACASK reference: 26,066 timepoints, 81,875 iterations
    # Ngspice reference: 20,556 timepoints, 80,018 iterations
    println("\n=== Results ===")
    @printf("Timepoints: %d (VACASK: 26,066, Ngspice: 20,556)\n", length(sol.t))
    @printf("NR iters:   %d (VACASK: 81,875, Ngspice: 80,018)\n", sol.stats.nnonliniter)
    @printf("Iter/step:  %.2f\n", sol.stats.nnonliniter / length(sol.t))
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
