#!/usr/bin/env julia
#==============================================================================#
# Clean Memory Profiling Script for VACASK RC Benchmark
#
# Uses proper warmup to avoid JIT compilation overhead.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using CedarSim
using CedarSim.MNA
using Sundials: IDA
using SciMLBase
using Printf
using LinearAlgebra
using SparseArrays

# Load and parse the SPICE netlist from file
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code, then evaluate to get the builder function
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit)
eval(circuit_code)

println("=" ^ 60)
println("Clean Memory Profiling: VACASK RC Benchmark")
println("=" ^ 60)

# Setup simulation
circuit = MNACircuit(rc_circuit)
MNA.assemble!(circuit)

# Create DAEProblem
tspan = (0.0, 0.01)
solver = IDA(max_error_test_failures=20)

prob = SciMLBase.DAEProblem(circuit, tspan; explicit_jacobian=true)
println("System size: $(length(prob.u0))")

# Get the workspace
ws = prob.p
cs = ws.structure
dctx = ws.dctx

# Test vectors
u_test = copy(prob.u0)
du_test = copy(prob.du0)
resid = zeros(length(prob.u0))
t_test = 0.001
gamma = 1.0

# Pre-allocate J - this should be the actual prototype used by the solver
J = cs.G + cs.C  # Same sparsity as passed to jac_prototype

#==============================================================================#
# Helper to measure with proper warmup
#==============================================================================#
function measure_allocations(f::Function, warmup_iters::Int=10, measure_iters::Int=1000)
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

#==============================================================================#
# Profile individual operations
#==============================================================================#
println("\n--- Profiling with proper warmup ---")

# 1. fast_rebuild!
allocs_rebuild = measure_allocations() do
    MNA.fast_rebuild!(ws, u_test, t_test)
end
@printf("\nfast_rebuild!:    %.1f bytes/call\n", allocs_rebuild)

# 2. fast_residual!
allocs_residual = measure_allocations() do
    MNA.fast_residual!(resid, du_test, u_test, ws, t_test)
end
@printf("fast_residual!:   %.1f bytes/call\n", allocs_residual)

# 3. fast_jacobian!
allocs_jacobian = measure_allocations() do
    MNA.fast_jacobian!(J, du_test, u_test, ws, gamma, t_test)
end
@printf("fast_jacobian!:   %.1f bytes/call\n", allocs_jacobian)

#==============================================================================#
# Breakdown fast_jacobian!
#==============================================================================#
println("\n--- fast_jacobian! breakdown ---")

# The fast_jacobian! function does:
# 1. fast_rebuild!(ws, u, t) - already measured as 0
# 2. copyto!(J, cs.G)
# 3. J .+= gamma .* cs.C

# Test copyto!
allocs_copyto = measure_allocations() do
    copyto!(J, cs.G)
end
@printf("\ncopyto!(J, cs.G): %.1f bytes/call\n", allocs_copyto)

# Test broadcast
allocs_broadcast = measure_allocations() do
    J .+= gamma .* cs.C
end
@printf("J .+= gamma .* cs.C: %.1f bytes/call\n", allocs_broadcast)

# Test intermediate allocation
allocs_intermediate = measure_allocations() do
    _ = gamma .* cs.C
end
@printf("gamma .* cs.C (temp): %.1f bytes/call\n", allocs_intermediate)

#==============================================================================#
# Test alternative zero-alloc implementation
#==============================================================================#
println("\n--- Alternative implementations ---")

# Since G and C have the same sparsity pattern as J (from jac_prototype = G + C),
# we need to handle this correctly

# Check if sparsity patterns match
println("\nSparsity pattern check:")
println("  J nnz: $(nnz(J))")
println("  G nnz: $(nnz(cs.G))")
println("  C nnz: $(nnz(cs.C))")
println("  G+C nnz: $(nnz(cs.G + cs.C))")

# The issue: J has sparsity of G+C, but G and C individually have different patterns
# We need to use the sparse broadcast which handles sparsity correctly

# Alternative: Use axpy! style operation if possible
# For sparse matrices: J = G + gamma*C can be done without allocation
# by storing G and C with compatible structure

# Let's check the actual pattern
# Using sparse broadcasting with pre-allocation
G_plus_C = similar(cs.G)

allocs_prealloc = measure_allocations() do
    # J = G + gamma*C using sparse operations
    copyto!(J, cs.G)  # Start with G
    # Add gamma*C: need to handle different sparsity
    for j in 1:size(J, 2)
        for k in SparseArrays.nzrange(cs.C, j)
            i = rowvals(cs.C)[k]
            v = nonzeros(cs.C)[k]
            # Find and update corresponding entry in J
            for jk in SparseArrays.nzrange(J, j)
                if rowvals(J)[jk] == i
                    nonzeros(J)[jk] += gamma * v
                    break
                end
            end
        end
    end
end
@printf("\nManual sparse update: %.1f bytes/call\n", allocs_prealloc)

#==============================================================================#
# Full solve profiling
#==============================================================================#
println("\n--- Full solve profiling ---")

# Warmup solve
sol1 = solve(prob, solver; dtmax=1e-4, maxiters=1000, dense=false)
GC.gc()

# Create fresh problem for measurement
prob2 = SciMLBase.DAEProblem(circuit, tspan; explicit_jacobian=true)

allocs_solve = @allocated begin
    sol2 = solve(prob2, solver; dtmax=1e-4, maxiters=10000, dense=false)
end

@printf("\nTotal solve allocations: %.2f MB\n", allocs_solve / 1e6)
@printf("Timepoints: %d\n", length(sol2.t))
@printf("NR iterations: %d\n", sol2.stats.nnonliniter)
@printf("Allocations per NR iteration: %.2f KB\n", allocs_solve / sol2.stats.nnonliniter / 1000)

# Estimate allocation sources
jacobian_allocs = allocs_jacobian * sol2.stats.nnonliniter
residual_allocs = allocs_residual * sol2.stats.nnonliniter

@printf("\nEstimated jacobian contribution: %.2f KB\n", jacobian_allocs / 1000)
@printf("Estimated residual contribution: %.2f KB\n", residual_allocs / 1000)
@printf("Remaining (solver internal): %.2f KB\n", (allocs_solve - jacobian_allocs - residual_allocs) / 1000)

#==============================================================================#
# Summary
#==============================================================================#
println("\n" * "=" ^ 60)
println("SUMMARY")
println("=" ^ 60)

if allocs_jacobian > 100
    println("\n⚠ ALLOCATION FOUND: fast_jacobian! allocates $(allocs_jacobian) bytes/call")
    println("  This is likely from: J .+= gamma .* cs.C")
    println("  The broadcast creates an intermediate sparse matrix")
    println("\n  SUGGESTED FIX:")
    println("  Replace the broadcast with in-place sparse addition")
end

if allocs_rebuild > 0
    println("\n⚠ ALLOCATION FOUND: fast_rebuild! allocates $(allocs_rebuild) bytes/call")
end

if allocs_residual > 0
    println("\n⚠ ALLOCATION FOUND: fast_residual! allocates $(allocs_residual) bytes/call")
end
