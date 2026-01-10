#!/usr/bin/env julia
#==============================================================================#
# Benchmark Rosenbrock-W Methods vs Rodas5P on Graetz
#
# Focus on ROS34PW2 and ROS34PW3 which work well with non-diagonal mass matrices
#==============================================================================#

using CedarSim
using CedarSim.MNA
using OrdinaryDiffEq
using VerilogAParser
using BenchmarkTools

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

function run_tran(solver; tspan=(0.0, 0.01), dtmax=1e-5)
    circuit = setup_simulation()
    sol = tran!(circuit, tspan; dtmax=dtmax, solver=solver,
                abstol=1e-3, reltol=1e-3, maxiters=100_000, dense=false)
    return sol
end

function benchmark_solver(name, solver)
    println("\n=== $name ===")

    # Warm-up
    println("  Warming up...")
    _ = run_tran(solver)
    GC.gc()

    # Benchmark (3 samples, allowing up to 60 seconds)
    println("  Benchmarking...")
    bench = @benchmark run_tran($solver) samples=3 evals=1 seconds=60

    # Get solution stats from a fresh run
    sol = run_tran(solver)

    median_time = median(bench).time / 1e9  # Convert ns to s
    allocs = bench.allocs
    memory = bench.memory / 1024 / 1024  # Convert to MiB

    println("  Median time:   $(round(median_time, digits=3)) s")
    println("  Allocations:   $allocs")
    println("  Memory:        $(round(memory, digits=1)) MiB")
    println("  Timepoints:    $(length(sol.t))")
    println("  Function evals: $(sol.stats.nf)")

    return (name=name, time=median_time, allocs=allocs, memory=memory,
            nsteps=length(sol.t), nf=sol.stats.nf)
end

function main()
    println("=" ^ 70)
    println("Rosenbrock-W Method Benchmark on Graetz")
    println("=" ^ 70)

    # Methods to benchmark - focus on working W methods
    solvers = [
        ("Rodas5P", Rodas5P()),
        ("ROS34PW2", ROS34PW2()),
        ("ROS34PW3", ROS34PW3()),
    ]

    results = []
    for (name, solver) in solvers
        push!(results, benchmark_solver(name, solver))
    end

    # Summary table
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)
    println()
    println("Method     | Time (s) | Allocs    | Memory (MiB) | Steps | f evals")
    println("-" ^ 70)

    for r in results
        println("$(rpad(r.name, 10)) | $(lpad(round(r.time, digits=3), 8)) | $(lpad(r.allocs, 9)) | $(lpad(round(r.memory, digits=1), 12)) | $(lpad(r.nsteps, 5)) | $(lpad(r.nf, 7))")
    end

    # Comparison with baseline
    baseline = results[1]
    println("\n--- Comparison to Rodas5P ---")
    for r in results[2:end]
        speedup = baseline.time / r.time
        alloc_ratio = r.allocs / baseline.allocs
        println("$(r.name): $(round(speedup, digits=2))x speedup, $(round(alloc_ratio*100, digits=1))% allocations")
    end
end

main()
