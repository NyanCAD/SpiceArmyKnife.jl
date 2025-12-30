#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: C6288 16x16 Multiplier
#
# A 16x16 bit multiplier circuit using PSP103 MOSFETs.
#
# Benchmark target: High complexity digital circuit
#==============================================================================#

using CedarSim
using CedarSim.MNA
using OrdinaryDiffEq
using Printf
using Statistics
using VerilogAParser

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
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:c6288_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)

function run_benchmark(; warmup=true, dtmax=2e-12)
    tspan = (0.0, 2e-9)  # 2ns simulation (same as ngspice)

    # Warmup run
    if warmup
        println("Warmup run...")
        try
            sim = MNASim(c6288_circuit)
            tran!(sim, (0.0, 10e-12); dtmax=dtmax)
            println("Warmup completed")
        catch e
            println("Warmup failed: ", e)
            showerror(stdout, e, catch_backtrace())
            return
        end
    end

    # Benchmark runs
    n_runs = 6
    times = Float64[]
    timepoints = Int[]

    for i in 1:n_runs
        sim = MNASim(c6288_circuit)

        t_start = time()
        sol = tran!(sim, tspan; dtmax=dtmax)
        t_elapsed = time() - t_start

        if i > 1
            push!(times, t_elapsed)
            push!(timepoints, length(sol.t))
        end

        @printf("Run %d: %.3f s, %d timepoints\n", i, t_elapsed, length(sol.t))
    end

    if length(times) > 0
        avg_time = mean(times)
        avg_timepoints = mean(timepoints)

        println("\n=== Results ===")
        @printf("Average time: %.3f s\n", avg_time)
        @printf("Average timepoints: %.0f\n", avg_timepoints)
        if length(times) > 1
            @printf("Std deviation: %.3f s (%.1f%%)\n", std(times), 100*std(times)/avg_time)
        end
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
