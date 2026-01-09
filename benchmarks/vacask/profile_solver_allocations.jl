#!/usr/bin/env julia
#==============================================================================#
# Memory Profiling: Solver Comparison
#
# Profile memory allocations for different ODE/DAE solvers to understand why
# Rodas5P allocates so much more than IDA/FBDF.
#
# Key questions:
# 1. Why does Rodas5P allocate 2.1GB vs 196MB for FBDF on RC circuit?
# 2. Why does Graetz allocate more than RC on the same solver?
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CedarSim
using CedarSim.MNA
using Sundials: IDA
using OrdinaryDiffEq: FBDF, Rodas5P
using SciMLBase
using Printf
using LinearAlgebra
using SparseArrays
using Profile

println("=" ^ 70)
println("Memory Profiling: Solver Comparison")
println("=" ^ 70)

#==============================================================================#
# Load RC Circuit
#==============================================================================#

const spice_file = joinpath(@__DIR__, "rc", "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

#==============================================================================#
# Helper Functions
#==============================================================================#

function measure_allocations(f::Function; warmup_iters::Int=3, measure_iters::Int=1)
    # Warmup
    for _ in 1:warmup_iters
        f()
    end
    GC.gc()

    # Measure
    allocs = @allocated begin
        for _ in 1:measure_iters
            f()
        end
    end

    return allocs / measure_iters
end

function profile_solve(solver, solver_name; tspan=(0.0, 0.01), dtmax=1e-4)
    println("\n--- $solver_name ---")

    # Setup
    circuit = MNACircuit(rc_circuit)
    MNA.assemble!(circuit)

    # Create problem
    if solver isa IDA
        prob = SciMLBase.DAEProblem(circuit, tspan; explicit_jacobian=true)
    else
        prob = SciMLBase.ODEProblem(circuit, tspan)
    end

    println("System size: $(length(prob.u0))")

    # Warmup solve
    sol = solve(prob, solver; dtmax=dtmax, maxiters=10000, dense=false)
    println("Warmup solve: $(length(sol.t)) timepoints, retcode=$(sol.retcode)")

    GC.gc()

    # Create fresh problem for measurement
    circuit2 = MNACircuit(rc_circuit)
    MNA.assemble!(circuit2)

    if solver isa IDA
        prob2 = SciMLBase.DAEProblem(circuit2, tspan; explicit_jacobian=true)
    else
        prob2 = SciMLBase.ODEProblem(circuit2, tspan)
    end

    # Measure allocations
    allocs = @allocated begin
        sol2 = solve(prob2, solver; dtmax=dtmax, maxiters=10000, dense=false)
    end

    @printf("Total allocations: %.2f MB\n", allocs / 1e6)
    @printf("Timepoints: %d\n", length(sol2.t))
    @printf("NR iterations: %d\n", sol2.stats.nnonliniter)
    @printf("Rejected steps: %d\n", hasfield(typeof(sol2.stats), :nreject) ? sol2.stats.nreject : 0)
    @printf("Allocs per timepoint: %.2f KB\n", allocs / length(sol2.t) / 1000)
    @printf("Allocs per NR iter: %.2f KB\n", allocs / sol2.stats.nnonliniter / 1000)

    return (
        solver = solver_name,
        allocs = allocs,
        timepoints = length(sol2.t),
        nr_iters = sol2.stats.nnonliniter,
        rejected = hasfield(typeof(sol2.stats), :nreject) ? sol2.stats.nreject : 0
    )
end

#==============================================================================#
# Profile Individual Operations
#==============================================================================#

function profile_operations()
    println("\n" * "=" ^ 70)
    println("Profiling Individual Operations")
    println("=" ^ 70)

    circuit = MNACircuit(rc_circuit)
    MNA.assemble!(circuit)

    # Get the workspace via compile
    ws = MNA.compile(circuit)
    cs = ws.structure

    n = MNA.system_size(cs)
    println("\nSystem size: $n")

    # Test vectors
    u_test = ones(n)
    du_test = zeros(n)
    resid = zeros(n)
    t_test = 0.001
    gamma = 1.0

    # Pre-allocate J with same sparsity
    J = copy(cs.G + cs.C)

    # Profile fast_rebuild!
    allocs_rebuild = measure_allocations(warmup_iters=100, measure_iters=1000) do
        MNA.fast_rebuild!(ws, u_test, t_test)
    end
    @printf("\nfast_rebuild!:   %.1f bytes/call\n", allocs_rebuild)

    # Profile fast_residual!
    allocs_residual = measure_allocations(warmup_iters=100, measure_iters=1000) do
        MNA.fast_residual!(resid, du_test, u_test, ws, t_test)
    end
    @printf("fast_residual!:  %.1f bytes/call\n", allocs_residual)

    # Profile fast_jacobian!
    allocs_jacobian = measure_allocations(warmup_iters=100, measure_iters=1000) do
        MNA.fast_jacobian!(J, du_test, u_test, ws, gamma, t_test)
    end
    @printf("fast_jacobian!:  %.1f bytes/call\n", allocs_jacobian)

    return (
        rebuild = allocs_rebuild,
        residual = allocs_residual,
        jacobian = allocs_jacobian
    )
end

#==============================================================================#
# Profile Solvers
#==============================================================================#

function profile_solvers()
    println("\n" * "=" ^ 70)
    println("Profiling Solvers (short simulation)")
    println("=" ^ 70)

    results = []

    # Short simulation for quick profiling
    tspan = (0.0, 0.01)  # 10ms instead of 1s
    dtmax = 1e-4

    push!(results, profile_solve(IDA(max_error_test_failures=20), "IDA"; tspan, dtmax))
    push!(results, profile_solve(FBDF(), "FBDF"; tspan, dtmax))
    push!(results, profile_solve(Rodas5P(), "Rodas5P"; tspan, dtmax))

    return results
end

#==============================================================================#
# Profile Longer Simulation for Scaling Analysis
#==============================================================================#

function profile_scaling()
    println("\n" * "=" ^ 70)
    println("Profiling Allocation Scaling")
    println("=" ^ 70)

    # Test at different simulation lengths to see if allocations scale with steps
    for tmax in [0.001, 0.01, 0.1]
        println("\n--- tspan = (0.0, $tmax) ---")

        # Create problem
        circuit = MNACircuit(rc_circuit)
        MNA.assemble!(circuit)
        prob = SciMLBase.ODEProblem(circuit, (0.0, tmax))

        # Warmup
        sol = solve(prob, Rodas5P(); dtmax=1e-4, maxiters=100000, dense=false)
        GC.gc()

        # Create fresh for measurement
        circuit2 = MNACircuit(rc_circuit)
        MNA.assemble!(circuit2)
        prob2 = SciMLBase.ODEProblem(circuit2, (0.0, tmax))

        allocs = @allocated begin
            sol2 = solve(prob2, Rodas5P(); dtmax=1e-4, maxiters=100000, dense=false)
        end

        @printf("  Timepoints: %d, Allocs: %.2f MB, Allocs/step: %.2f KB\n",
                length(sol2.t), allocs/1e6, allocs/length(sol2.t)/1000)
    end
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    # Profile individual operations
    op_results = profile_operations()

    # Profile solvers
    solver_results = profile_solvers()

    # Profile scaling
    profile_scaling()

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)

    println("\nOperation allocations (per call):")
    @printf("  fast_rebuild!:   %.1f bytes\n", op_results.rebuild)
    @printf("  fast_residual!:  %.1f bytes\n", op_results.residual)
    @printf("  fast_jacobian!:  %.1f bytes\n", op_results.jacobian)

    println("\nSolver allocations (short simulation):")
    for r in solver_results
        @printf("  %-10s: %8.2f MB (%d steps, %d NR iters)\n",
                r.solver, r.allocs/1e6, r.timepoints, r.nr_iters)
    end

    # Calculate per-step overhead
    println("\nPer-step allocation (excluding setup):")
    for r in solver_results
        setup_estimate = 1e6  # Assume ~1MB setup cost
        per_step = (r.allocs - setup_estimate) / r.timepoints
        @printf("  %-10s: %.2f KB/step\n", r.solver, per_step/1000)
    end
end

main()
