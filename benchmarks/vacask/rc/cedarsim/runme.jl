#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: RC Circuit
#
# RC circuit excited by a pulse train.
# This is a simple linear circuit - no rejected timepoints.
#
# Benchmark target: ~1 million timepoints, ~2 million iterations
#==============================================================================#

using CedarSim
using CedarSim.MNA
using OrdinaryDiffEq
using BenchmarkTools
using Printf

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

"""
    setup_simulation(; dtmax=1e-6)

Create and return a fully-prepared MNASim ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation(; dtmax=1e-6)
    sim = MNASim(rc_circuit)
    # Perform DC operating point to initialize the circuit
    MNA.assemble!(sim)
    return sim
end

function run_benchmark(; warmup=true, dtmax=1e-6)
    tspan = (0.0, 1.0)  # 1 second simulation

    # Warmup run (compiles everything)
    if warmup
        println("Warmup run...")
        sim = setup_simulation(; dtmax=dtmax)
        tran!(sim, (0.0, 0.001); dtmax=dtmax)
    end

    # Setup the simulation outside the timed region
    sim = setup_simulation(; dtmax=dtmax)

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis...")
    bench = @benchmark tran!($sim, $tspan; dtmax=$dtmax) samples=6 evals=1 seconds=600

    # Also run once to get solution statistics
    sim = setup_simulation(; dtmax=dtmax)
    sol = tran!(sim, tspan; dtmax=dtmax)

    println("\n=== Results ===")
    @printf("Timepoints: %d\n", length(sol.t))
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
