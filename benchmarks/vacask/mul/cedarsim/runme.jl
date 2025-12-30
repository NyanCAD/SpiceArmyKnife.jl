#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Diode Voltage Multiplier
#
# A voltage multiplier (4 diodes, 4 capacitors) with a series resistor at its
# input excited by a sinusoidal voltage.
#
# Benchmark target: ~500k timepoints
#==============================================================================#

using CedarSim
using CedarSim.MNA
using OrdinaryDiffEq
using Printf
using Statistics
using VerilogAParser

# Load the vadistiller diode model
const diode_va_path = joinpath(@__DIR__, "..", "..", "..", "..", "test", "vadistiller", "models", "diode.va")

if isfile(diode_va_path)
    va = VerilogAParser.parsefile(diode_va_path)
    if !va.ps.errored
        Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
    else
        error("Failed to parse diode VA model")
    end
else
    error("Diode VA model not found at $diode_va_path")
end

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
# Pass sp_diode_module so the SPICE parser knows about our VA device
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:mul_circuit,
                                         imported_hdl_modules=[sp_diode_module])
eval(circuit_code)

function run_benchmark(; warmup=true, dtmax=0.01e-6)
    tspan = (0.0, 5e-3)  # 5ms simulation

    # Warmup run
    if warmup
        println("Warmup run...")
        try
            sim = MNASim(mul_circuit)
            tran!(sim, (0.0, 0.0001); dtmax=dtmax)
        catch e
            println("Warmup failed: ", e)
            showerror(stdout, e, catch_backtrace())
        end
    end

    # Benchmark runs
    n_runs = 6
    times = Float64[]
    timepoints = Int[]

    for i in 1:n_runs
        sim = MNASim(mul_circuit)

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
