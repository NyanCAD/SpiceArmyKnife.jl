#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: C6288 16x16 Multiplier
#
# A 16x16 bit multiplier circuit using PSP103 MOSFETs.
#
# Benchmark target: High complexity digital circuit
#
# Note: Uses ImplicitEuler solver since the small timestep (2ps) is artificially
# enforced rather than physics-driven. First-order methods are more efficient
# when the timestep is forced small.
#==============================================================================#

using CedarSim
using CedarSim.MNA
using OrdinaryDiffEq
using BenchmarkTools
using Printf
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

"""
    setup_simulation(; dtmax=2e-12)

Create and return a fully-prepared MNASim ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation(; dtmax=2e-12)
    sim = MNASim(c6288_circuit)
    # Perform DC operating point to initialize the circuit
    MNA.assemble!(sim)
    return sim
end

function run_benchmark(; warmup=true, dtmax=2e-12)
    tspan = (0.0, 2e-9)  # 2ns simulation (same as ngspice)

    # Use ImplicitEuler for forced small timesteps - first-order is more efficient
    # when the timestep is artificially constrained rather than physics-driven
    solver = ImplicitEuler()

    # Warmup run (compiles everything)
    if warmup
        println("Warmup run...")
        try
            sim = setup_simulation(; dtmax=dtmax)
            tran!(sim, (0.0, 10e-12); dtmax=dtmax, solver=solver)
            println("Warmup completed")
        catch e
            println("Warmup failed: ", e)
            showerror(stdout, e, catch_backtrace())
            return nothing, nothing
        end
    end

    # Setup the simulation outside the timed region
    sim = setup_simulation(; dtmax=dtmax)

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with ImplicitEuler...")
    bench = @benchmark tran!($sim, $tspan; dtmax=$dtmax, solver=$solver) samples=6 evals=1 seconds=600

    # Also run once to get solution statistics
    sim = setup_simulation(; dtmax=dtmax)
    sol = tran!(sim, tspan; dtmax=dtmax, solver=solver)

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
