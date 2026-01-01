#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Graetz Bridge (Full-wave Rectifier)
#
# A full wave rectifier (4 diodes) with a capacitor and a resistor as load
# excited by a sinusoidal voltage.
#
# Uses IDA (DAE solver) which correctly handles voltage-dependent junction
# capacitance in the diode model. IDA is much more efficient than ODE solvers
# for this circuit (599 steps vs ~10k+ with ImplicitEuler).
#==============================================================================#

using CedarSim
using CedarSim.MNA
using Sundials
using BenchmarkTools
using Printf
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
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:graetz_circuit,
                                         imported_hdl_modules=[sp_diode_module])
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation()
    circuit = MNACircuit(graetz_circuit)
    # Perform DC operating point to initialize the circuit
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(; warmup=true)
    tspan = (0.0, 1.0)  # 1 second simulation

    # Use IDA (DAE solver) which correctly handles voltage-dependent junction
    # capacitance in the diode model. Much more efficient than ODE solvers.
    # IDA uses adaptive timesteps based on error control.

    # Warmup run (compiles everything)
    if warmup
        println("Warmup run...")
        try
            circuit = setup_simulation()
            tran!(circuit, (0.0, 0.02); abstol=1e-6, reltol=1e-4)
        catch e
            println("Warmup failed: ", e)
            showerror(stdout, e, catch_backtrace())
            return nothing, nothing
        end
    end

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Benchmark the actual simulation (not setup)
    # IDA uses adaptive timesteps - no dtmax needed
    println("\nBenchmarking transient analysis with IDA (DAE solver)...")
    bench = @benchmark tran!($circuit, $tspan; abstol=1e-6, reltol=1e-4) samples=6 evals=1 seconds=600

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; abstol=1e-6, reltol=1e-4)

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
