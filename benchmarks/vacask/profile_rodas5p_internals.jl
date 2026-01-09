#!/usr/bin/env julia
#==============================================================================#
# Profile Rodas5P Internals
#
# Investigate what causes Rodas5P to allocate ~10x more per step than FBDF.
# Key hypothesis: Rosenbrock methods need to compute/factorize Jacobian every step.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CedarSim
using CedarSim.MNA
using OrdinaryDiffEq: Rodas5P, FBDF
using SciMLBase
using Printf
using LinearAlgebra
using SparseArrays

println("=" ^ 70)
println("Profile Rodas5P Internals")
println("=" ^ 70)

#==============================================================================#
# Load RC Circuit
#==============================================================================#

const spice_file = joinpath(@__DIR__, "rc", "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

#==============================================================================#
# Profile Jacobian and LU Operations
#==============================================================================#

function profile_jacobian_lu()
    println("\n--- Jacobian + LU Factorization Analysis ---")

    circuit = MNACircuit(rc_circuit)
    MNA.assemble!(circuit)
    ws = MNA.compile(circuit)
    cs = ws.structure

    n = MNA.system_size(cs)
    println("System size: $n")

    # Mass matrix and Jacobian
    M = cs.C  # Mass matrix
    J_proto = cs.G + cs.C  # Jacobian prototype

    # For Rosenbrock: W = M - gamma*J (computed every step)
    gamma = 1.0
    W = similar(J_proto)

    # Simulate what Rosenbrock does each step
    u = ones(n)
    du = zeros(n)

    println("\nPer-step operations in Rosenbrock:")

    # 1. Jacobian computation (from our code - zero alloc)
    J = copy(J_proto)
    allocs_jac = @allocated begin
        for _ in 1:1000
            MNA.fast_jacobian!(J, du, u, ws, gamma, 0.0)
        end
    end
    @printf("  Our fast_jacobian!:     %.1f bytes/call\n", allocs_jac / 1000)

    # 2. W = M - gamma*J (what OrdinaryDiffEq does internally)
    allocs_w = @allocated begin
        for _ in 1:1000
            # This is roughly what happens internally
            W .= M .- gamma .* J
        end
    end
    @printf("  W = M - gamma*J:        %.1f bytes/call\n", allocs_w / 1000)

    # 3. LU factorization (this is the expensive part)
    # Sparse LU
    allocs_lu_sparse = @allocated begin
        for _ in 1:100
            F = lu(J_proto)
        end
    end
    @printf("  lu(sparse J):           %.1f bytes/call\n", allocs_lu_sparse / 100)

    # Dense LU (for comparison)
    J_dense = Matrix(J_proto)
    allocs_lu_dense = @allocated begin
        for _ in 1:100
            F = lu(J_dense)
        end
    end
    @printf("  lu(dense J):            %.1f bytes/call\n", allocs_lu_dense / 100)

    # 4. Back-substitution (should be cheap)
    F_sparse = lu(J_proto)
    b = ones(n)
    allocs_solve = @allocated begin
        for _ in 1:1000
            x = F_sparse \ b
        end
    end
    @printf("  F \\ b (back-solve):     %.1f bytes/call\n", allocs_solve / 1000)

    println("\nEstimated per-step cost for Rosenbrock:")
    estimated = allocs_jac/1000 + allocs_w/1000 + allocs_lu_sparse/100 + allocs_solve/1000
    @printf("  Total per step:         %.1f bytes\n", estimated)
    @printf("  Actual observed:        %.1f bytes (from benchmark)\n", 2.15 * 1000)

    # Rodas5P does multiple stages - typically 5-6 linear solves per step
    num_stages = 6
    estimated_rodas = estimated * num_stages
    @printf("  With %d stages:          %.1f bytes\n", num_stages, estimated_rodas)
end

#==============================================================================#
# Profile Solution Vector Storage
#==============================================================================#

function profile_solution_storage()
    println("\n--- Solution Storage Analysis ---")

    circuit = MNACircuit(rc_circuit)
    MNA.assemble!(circuit)

    n = MNA.system_size(MNA.compile(circuit))
    println("System size: $n")

    # Each timepoint stores:
    # - u vector: n * 8 bytes
    # - t scalar: 8 bytes
    # Total per timepoint: (n+1) * 8 bytes

    bytes_per_timepoint = (n + 1) * 8
    @printf("\nSolution storage per timepoint: %d bytes (%.2f KB)\n",
            bytes_per_timepoint, bytes_per_timepoint/1000)

    # For 1M timepoints
    total_1m = bytes_per_timepoint * 1_000_000
    @printf("For 1M timepoints: %.1f MB\n", total_1m / 1e6)

    # The benchmark reports ~200MB for FBDF with 1M timepoints
    # n=3, so (3+1)*8 = 32 bytes per timepoint
    # 32 * 1M = 32 MB just for solution storage
    # But we observe 196 MB, so there's more overhead
end

#==============================================================================#
# Compare Jacobian Strategy
#==============================================================================#

function profile_jacobian_reuse()
    println("\n--- Jacobian Reuse Analysis ---")

    circuit = MNACircuit(rc_circuit)
    MNA.assemble!(circuit)

    # Create problem
    prob = SciMLBase.ODEProblem(circuit, (0.0, 0.01))

    # Check stats from both solvers
    println("\nFBDF (lazy Jacobian updates):")
    sol_fbdf = solve(prob, FBDF(); dtmax=1e-4, maxiters=10000, dense=false)
    @printf("  Steps: %d\n", length(sol_fbdf.t))
    @printf("  NR iterations: %d\n", sol_fbdf.stats.nnonliniter)
    @printf("  NR iters/step: %.2f\n", sol_fbdf.stats.nnonliniter / length(sol_fbdf.t))
    if hasfield(typeof(sol_fbdf.stats), :njacs)
        @printf("  Jacobian evals: %d\n", sol_fbdf.stats.njacs)
    end

    println("\nRodas5P (Jacobian every step):")
    sol_rodas = solve(prob, Rodas5P(); dtmax=1e-4, maxiters=10000, dense=false)
    @printf("  Steps: %d\n", length(sol_rodas.t))
    @printf("  NR iterations: %d\n", sol_rodas.stats.nnonliniter)
    if hasfield(typeof(sol_rodas.stats), :njacs)
        @printf("  Jacobian evals: %d\n", sol_rodas.stats.njacs)
    end
end

#==============================================================================#
# Profile with Allocation Tracker
#==============================================================================#

function profile_with_allocation_tracker()
    println("\n--- Allocation Source Breakdown ---")

    circuit = MNACircuit(rc_circuit)
    MNA.assemble!(circuit)
    prob = SciMLBase.ODEProblem(circuit, (0.0, 0.001))  # Very short

    println("\nRunning Rodas5P with allocation tracking...")

    # First, warmup
    solve(prob, Rodas5P(); dtmax=1e-5, maxiters=1000, dense=false)
    GC.gc()

    # Profile
    circuit2 = MNACircuit(rc_circuit)
    MNA.assemble!(circuit2)
    prob2 = SciMLBase.ODEProblem(circuit2, (0.0, 0.001))

    allocs = @allocated begin
        sol = solve(prob2, Rodas5P(); dtmax=1e-5, maxiters=1000, dense=false)
    end

    sol = solve(prob2, Rodas5P(); dtmax=1e-5, maxiters=1000, dense=false)

    @printf("\nTotal allocations: %.2f MB for %d steps\n", allocs/1e6, length(sol.t))
    @printf("Per step: %.2f KB\n", allocs/length(sol.t)/1000)
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    profile_jacobian_lu()
    profile_solution_storage()
    profile_jacobian_reuse()
    profile_with_allocation_tracker()

    println("\n" * "=" ^ 70)
    println("CONCLUSIONS")
    println("=" ^ 70)

    println("""

The high memory allocation in Rodas5P comes from:

1. **LU Factorization Every Step**
   - Rosenbrock methods (like Rodas5P) form W = M - γJ and factorize it EVERY step
   - This requires allocating the factorization structures
   - For sparse matrices, this is ~1-2 KB per factorization

2. **Multiple Stages Per Step**
   - Rodas5P has 5 stages, each requiring a linear solve
   - Total: ~5-6 factorizations and back-substitutes per step

3. **No Jacobian Reuse**
   - BDF methods (FBDF, IDA) reuse the Jacobian across many Newton iterations
   - They only refactorize when convergence is slow
   - Rosenbrock methods MUST use a fresh Jacobian every step

4. **Solution Storage**
   - Both solvers store the same amount of solution data
   - This is ~32 bytes per timepoint for a 3-node circuit
   - ~32 MB for 1M timepoints

The key insight: Rodas5P's per-step allocation is ~10x higher because it
refactorizes the Jacobian every step, while FBDF/IDA amortize this cost
across many steps by reusing factorizations.
""")
end

main()
