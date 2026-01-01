#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Diode Voltage Multiplier
#
# A voltage multiplier (4 diodes, 4 capacitors) with a series resistor at its
# input excited by a sinusoidal voltage.
#
# Benchmark target: ~500k timepoints, ~1M NR iterations
#
# Note: Uses ImplicitEuler solver with fixed timesteps (adaptive=false) to match
# the VACASK benchmark methodology. The sin source uses phase=90 to start at
# peak voltage (dV/dt=0), avoiding Newton convergence issues at t=0 that occur
# with the cascaded diode topology when starting at phase=0.
#==============================================================================#

using CedarSim
using CedarSim.MNA
using OrdinaryDiffEq
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
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:mul_circuit,
                                         imported_hdl_modules=[sp_diode_module])
eval(circuit_code)

"""
    setup_simulation()

Create and return a fully-prepared MNACircuit ready for transient analysis.
This separates problem setup from solve time for accurate benchmarking.
"""
function setup_simulation()
    circuit = MNACircuit(mul_circuit)
    # Perform DC operating point to initialize the circuit
    MNA.assemble!(circuit)
    return circuit
end

function run_benchmark(; warmup=true, dt=1e-8)
    tspan = (0.0, 5e-3)  # 5ms simulation (~500k timepoints with dt=10ns)

    # Use ImplicitEuler with fixed timesteps for benchmarking
    # adaptive=false forces the solver to use the specified dt
    # Loose tolerances (1e-3) ensure Newton converges at each step
    solver = ImplicitEuler(nlsolve=NLNewton(max_iter=100))

    # Warmup run (compiles everything)
    if warmup
        println("Warmup run...")
        try
            circuit = setup_simulation()
            tran!(circuit, (0.0, 1e-5); dt=dt, adaptive=false, solver=solver, abstol=1e-3, reltol=1e-3)
        catch e
            println("Warmup failed: ", e)
            showerror(stdout, e, catch_backtrace())
            return nothing, nothing
        end
    end

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with ImplicitEuler (fixed dt=$dt)...")
    bench = @benchmark tran!($circuit, $tspan; dt=$dt, adaptive=false, solver=$solver, abstol=1e-3, reltol=1e-3) samples=6 evals=1 seconds=600

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; dt=dt, adaptive=false, solver=solver, abstol=1e-3, reltol=1e-3)

    println("\n=== Results ===")
    @printf("Timepoints:  %d\n", length(sol.t))
    @printf("Expected:    ~%d\n", round(Int, (tspan[2] - tspan[1]) / dt) + 1)
    @printf("NR iters:    %d\n", sol.stats.nnonliniter)
    @printf("Iter/step:   %.2f\n", sol.stats.nnonliniter / length(sol.t))
    display(bench)
    println()

    return bench, sol
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
