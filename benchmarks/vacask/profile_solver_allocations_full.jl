#!/usr/bin/env julia
#==============================================================================#
# Memory Profiling: Full Benchmark Simulation
#
# Profile actual benchmark scenario to understand memory allocation patterns.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CedarSim
using CedarSim.MNA
using Sundials: IDA
using OrdinaryDiffEq: FBDF, Rodas5P
using SciMLBase
using Printf

println("=" ^ 70)
println("Memory Profiling: Full Benchmark Simulation (1 second, dtmax=1e-6)")
println("=" ^ 70)

#==============================================================================#
# Load RC Circuit
#==============================================================================#

const spice_file = joinpath(@__DIR__, "rc", "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

#==============================================================================#
# Profile Full Simulation
#==============================================================================#

function profile_full_solve(solver, solver_name; tspan=(0.0, 1.0), dtmax=1e-6)
    println("\n" * "=" ^ 50)
    println("$solver_name")
    println("=" ^ 50)

    # Warmup (short simulation)
    println("Warming up...")
    circuit_warmup = MNACircuit(rc_circuit)
    MNA.assemble!(circuit_warmup)

    if solver isa IDA
        prob_warmup = SciMLBase.DAEProblem(circuit_warmup, (0.0, 0.001); explicit_jacobian=true)
    else
        prob_warmup = SciMLBase.ODEProblem(circuit_warmup, (0.0, 0.001))
    end
    solve(prob_warmup, solver; dtmax=dtmax, maxiters=10000, dense=false)

    GC.gc()

    # Full simulation with measurement
    println("Running full simulation (this may take a while)...")

    circuit = MNACircuit(rc_circuit)
    MNA.assemble!(circuit)

    if solver isa IDA
        prob = SciMLBase.DAEProblem(circuit, tspan; explicit_jacobian=true)
    else
        prob = SciMLBase.ODEProblem(circuit, tspan)
    end

    # Time the solve
    start_time = time()
    allocs_start = Base.gc_live_bytes()

    sol = solve(prob, solver; dtmax=dtmax, maxiters=10_000_000, dense=false)

    allocs_end = Base.gc_live_bytes()
    elapsed = time() - start_time

    # Also measure with @allocated
    GC.gc()

    circuit2 = MNACircuit(rc_circuit)
    MNA.assemble!(circuit2)

    if solver isa IDA
        prob2 = SciMLBase.DAEProblem(circuit2, tspan; explicit_jacobian=true)
    else
        prob2 = SciMLBase.ODEProblem(circuit2, tspan)
    end

    allocs = @allocated begin
        sol2 = solve(prob2, solver; dtmax=dtmax, maxiters=10_000_000, dense=false)
    end

    println("\nResults:")
    @printf("  Timepoints:        %d\n", length(sol.t))
    @printf("  NR iterations:     %d\n", sol.stats.nnonliniter)
    @printf("  Rejected steps:    %d\n", hasfield(typeof(sol.stats), :nreject) ? sol.stats.nreject : 0)
    @printf("  Return code:       %s\n", sol.retcode)
    @printf("  Wall time:         %.2f s\n", elapsed)
    @printf("  Memory allocated:  %.2f MB\n", allocs / 1e6)
    @printf("  Allocs/timepoint:  %.2f KB\n", allocs / length(sol.t) / 1000)
    if sol.stats.nnonliniter > 0
        @printf("  Allocs/NR iter:    %.2f KB\n", allocs / sol.stats.nnonliniter / 1000)
    end

    return (
        solver = solver_name,
        timepoints = length(sol.t),
        nr_iters = sol.stats.nnonliniter,
        rejected = hasfield(typeof(sol.stats), :nreject) ? sol.stats.nreject : 0,
        allocs = allocs,
        time = elapsed,
        retcode = sol.retcode
    )
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    results = []

    # Use the actual benchmark parameters
    tspan = (0.0, 1.0)
    dtmax = 1e-6

    println("\nSimulation parameters: tspan=$tspan, dtmax=$dtmax")

    push!(results, profile_full_solve(IDA(max_error_test_failures=20), "IDA"; tspan, dtmax))
    push!(results, profile_full_solve(FBDF(), "FBDF"; tspan, dtmax))
    push!(results, profile_full_solve(Rodas5P(), "Rodas5P"; tspan, dtmax))

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY: RC Circuit (1 second, 1µs dtmax)")
    println("=" ^ 70)

    println("\n| Solver   | Time (s) | Timepoints | NR Iters  | Memory (MB) | KB/step |")
    println("|----------|----------|------------|-----------|-------------|---------|")
    for r in results
        @printf("| %-8s | %8.2f | %10d | %9d | %11.1f | %7.2f |\n",
                r.solver, r.time, r.timepoints, r.nr_iters,
                r.allocs/1e6, r.allocs/r.timepoints/1000)
    end

    # Analyze why Rodas5P allocates more
    println("\n" * "=" ^ 70)
    println("ANALYSIS")
    println("=" ^ 70)

    ida_result = results[1]
    fbdf_result = results[2]
    rodas_result = results[3]

    println("\nRodas5P vs FBDF comparison:")
    @printf("  Rodas5P allocates:      %.1fx more memory\n",
            rodas_result.allocs / fbdf_result.allocs)
    @printf("  Rodas5P takes:          %.1fx more steps\n",
            rodas_result.timepoints / fbdf_result.timepoints)
    @printf("  Per-step allocation ratio: %.2f\n",
            (rodas_result.allocs / rodas_result.timepoints) /
            (fbdf_result.allocs / fbdf_result.timepoints))

    println("\nKey observations:")
    println("  - Rodas5P is a Rosenbrock method (no Newton iteration, computes Jacobian every step)")
    println("  - FBDF reuses Jacobians (Newton iteration with lazy Jacobian updates)")
    println("  - IDA (Sundials) uses similar Newton strategy to FBDF")
end

main()
