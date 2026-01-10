#!/usr/bin/env julia
#==============================================================================#
# Test Rosenbrock-W Methods on Graetz Benchmark
#
# Compare Rosenbrock-W methods with Rodas5P to see:
# 1. Which W methods support mass matrix
# 2. Stability comparison
# 3. Allocation comparison (W methods should allocate less due to reusing Jacobian)
#==============================================================================#

using CedarSim
using CedarSim.MNA
using OrdinaryDiffEq
using VerilogAParser

# Load the vadistiller diode model
const diode_va_path = joinpath(@__DIR__, "..", "..", "..", "test", "vadistiller", "models", "diode.va")

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

# Load and parse the SPICE netlist
const spice_file = joinpath(@__DIR__, "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)

const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:graetz_circuit,
                                         imported_hdl_modules=[sp_diode_module])
eval(circuit_code)

function setup_simulation()
    circuit = MNACircuit(graetz_circuit)
    MNA.assemble!(circuit)
    return circuit
end

# List of Rosenbrock-W methods to test
# These are W-methods that may reuse approximate Jacobians
const w_methods = [
    ("Rosenbrock23", Rosenbrock23()),
    ("Rosenbrock32", Rosenbrock32()),
    ("ROS34PW1a", ROS34PW1a()),
    ("ROS34PW1b", ROS34PW1b()),
    ("ROS34PW2", ROS34PW2()),
    ("ROS34PW3", ROS34PW3()),
]

# Baseline method
const baseline = ("Rodas5P", Rodas5P())

function test_solver(name, solver; tspan=(0.0, 0.01), dtmax=1e-5)
    circuit = setup_simulation()

    println("\n=== Testing $name ===")

    # Time the simulation
    try
        GC.gc()
        allocs_before = Base.gc_live_bytes()
        t_start = time()

        sol = tran!(circuit, tspan; dtmax=dtmax, solver=solver,
                    abstol=1e-3, reltol=1e-3, maxiters=100_000, dense=false)

        t_elapsed = time() - t_start
        GC.gc()
        allocs_after = Base.gc_live_bytes()

        # Check if successful
        if sol.retcode == :Success || sol.retcode == SciMLBase.ReturnCode.Success
            println("  Status:      SUCCESS")
            println("  Time:        $(round(t_elapsed, digits=3)) s")
            println("  Timepoints:  $(length(sol.t))")
            println("  NR iters:    $(sol.stats.nnonliniter)")
            println("  Iter/step:   $(round(sol.stats.nnonliniter / length(sol.t), digits=2))")
            return (success=true, time=t_elapsed, nsteps=length(sol.t), niters=sol.stats.nnonliniter)
        else
            println("  Status:      FAILED ($(sol.retcode))")
            return (success=false, time=NaN, nsteps=0, niters=0)
        end
    catch e
        println("  Status:      ERROR")
        println("  Error:       $e")
        return (success=false, time=NaN, nsteps=0, niters=0)
    end
end

function main()
    println("=" ^ 70)
    println("Rosenbrock-W Method Mass Matrix Test on Graetz Benchmark")
    println("=" ^ 70)

    # Test baseline first
    baseline_result = test_solver(baseline[1], baseline[2])

    # Test all W methods
    results = Dict{String, NamedTuple}()
    results[baseline[1]] = baseline_result

    for (name, solver) in w_methods
        results[name] = test_solver(name, solver)
    end

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()

    # Table header
    println("Method         | Success | Time (s) | Steps | NR Iters | Iter/Step")
    println("-" ^ 70)

    for (name, r) in sort(collect(results), by=x->x[1])
        if r.success
            println("$(rpad(name, 14)) |   ✓     | $(lpad(round(r.time, digits=3), 8)) | $(lpad(r.nsteps, 5)) | $(lpad(r.niters, 8)) | $(round(r.niters/r.nsteps, digits=2))")
        else
            println("$(rpad(name, 14)) |   ✗     |      N/A |   N/A |      N/A |    N/A")
        end
    end
end

main()
