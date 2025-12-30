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
using Printf
using Statistics

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

function run_benchmark(; warmup=true, dtmax=1e-6)
    tspan = (0.0, 1.0)  # 1 second simulation

    # Warmup run (if requested)
    if warmup
        println("Warmup run...")
        sim = MNASim(rc_circuit)
        tran!(sim, (0.0, 0.001); dtmax=dtmax)
    end

    # Benchmark runs
    n_runs = 6
    times = Float64[]
    timepoints = Int[]

    for i in 1:n_runs
        sim = MNASim(rc_circuit)

        t_start = time()
        sol = tran!(sim, tspan; dtmax=dtmax)
        t_elapsed = time() - t_start

        if i > 1  # Skip first run (still filling caches)
            push!(times, t_elapsed)
            push!(timepoints, length(sol.t))
        end

        @printf("Run %d: %.3f s, %d timepoints\n", i, t_elapsed, length(sol.t))
    end

    avg_time = mean(times)
    avg_timepoints = mean(timepoints)

    println("\n=== Results ===")
    @printf("Average time: %.3f s\n", avg_time)
    @printf("Average timepoints: %.0f\n", avg_timepoints)
    @printf("Std deviation: %.3f s (%.1f%%)\n", std(times), 100*std(times)/avg_time)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
