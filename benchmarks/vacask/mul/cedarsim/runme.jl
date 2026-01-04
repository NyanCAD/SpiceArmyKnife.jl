#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark: Diode Voltage Multiplier
#
# A voltage multiplier (4 diodes, 4 capacitors) with a series resistor at its
# input excited by a sinusoidal voltage.
#
# Benchmark target: ~500k timepoints, ~500k-1M NR iterations
#
# Note: Uses Sundials IDA (variable-order BDF, orders 1-5) with dtmax to enforce
# fixed timesteps. IDA is comparable to ngspice's Gear method and uses our
# explicit Jacobian for 5x memory reduction vs DABDF2 autodiff.
#
# The sin source uses phase=90 to start at peak voltage (dV/dt=0), avoiding
# Newton convergence issues at t=0 with the cascaded diode topology.
#
# Important: This benchmark requires dt=1ns (not 10ns like ngspice) due to the
# stiff diode switching in the VA model. The simulation time is reduced to 0.5ms
# to maintain the ~500k timestep target.
#
# TODO: Investigate whether gmin is fully hooked up to VA models. The smaller
# timestep requirement may be related to gmin not being properly applied.
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

function run_benchmark(; dt=1e-9)
    tspan = (0.0, 0.5e-3)  # 0.5ms simulation (~500k timepoints with dt=1ns)

    # Use Sundials IDA (variable-order BDF) with dtmax to enforce timestep constraint.
    # IDA uses our explicit Jacobian (5x less memory than DABDF2 autodiff).
    # - max_nonlinear_iters=100: High limit ensures Newton converges at each step
    # - max_error_test_failures=20: More retries for difficult transient points
    solver = IDA(max_nonlinear_iters=100, max_error_test_failures=20)

    # Setup the simulation outside the timed region
    circuit = setup_simulation()

    # Benchmark the actual simulation (not setup)
    println("\nBenchmarking transient analysis with IDA (dtmax=$dt)...")
    bench = @benchmark tran!($circuit, $tspan; dtmax=$dt, solver=$solver, abstol=1e-3, reltol=1e-3) samples=6 evals=1 seconds=600

    # Also run once to get solution statistics
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; dtmax=dt, solver=solver, abstol=1e-3, reltol=1e-3)

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
