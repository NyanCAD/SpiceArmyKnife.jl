#!/usr/bin/env julia
#==============================================================================#
# Profile Graetz vs RC Memory Allocations
#
# Investigate why Graetz (nonlinear) allocates 2.08 GB vs RC's 196 MB with FBDF.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CedarSim
using CedarSim.MNA
using OrdinaryDiffEq: FBDF
using Sundials: IDA
using SciMLBase
using Printf
using LinearAlgebra
using SparseArrays
using VerilogAParser

println("=" ^ 70)
println("Profile Graetz vs RC Memory Allocations")
println("=" ^ 70)

#==============================================================================#
# Load RC Circuit
#==============================================================================#

println("\nLoading RC Circuit...")
const rc_spice_file = joinpath(@__DIR__, "rc", "cedarsim", "runme.sp")
const rc_spice_code = read(rc_spice_file, String)
const rc_circuit_code = parse_spice_to_mna(rc_spice_code; circuit_name=:rc_circuit)
eval(rc_circuit_code)

#==============================================================================#
# Load Graetz Circuit (with VA diode model)
#==============================================================================#

println("Loading Graetz Circuit (with VA diode)...")
const diode_va_path = joinpath(@__DIR__, "..", "..", "test", "vadistiller", "models", "diode.va")
const va = VerilogAParser.parsefile(diode_va_path)
if !va.ps.errored
    Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
else
    error("Failed to parse diode VA model")
end

const graetz_spice_file = joinpath(@__DIR__, "graetz", "cedarsim", "runme.sp")
const graetz_spice_code = read(graetz_spice_file, String)
const graetz_circuit_code = parse_spice_to_mna(graetz_spice_code; circuit_name=:graetz_circuit,
                                                imported_hdl_modules=[sp_diode_module])
eval(graetz_circuit_code)

#==============================================================================#
# Compare Circuit Structure
#==============================================================================#

function compare_circuit_structure()
    println("\n" * "=" ^ 50)
    println("Circuit Structure Comparison")
    println("=" ^ 50)

    # RC Circuit
    println("\nRC Circuit:")
    rc_circ = MNACircuit(rc_circuit)
    rc_ws = MNA.compile(rc_circ)
    rc_cs = rc_ws.structure
    @printf("  System size:  %d\n", MNA.system_size(rc_cs))
    @printf("  Nodes:        %d\n", rc_cs.n_nodes)
    @printf("  Currents:     %d\n", rc_cs.n_currents)
    @printf("  G nnz:        %d\n", nnz(rc_cs.G))
    @printf("  C nnz:        %d\n", nnz(rc_cs.C))

    # Graetz Circuit
    println("\nGraetz Circuit:")
    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    graetz_cs = graetz_ws.structure
    @printf("  System size:  %d\n", MNA.system_size(graetz_cs))
    @printf("  Nodes:        %d\n", graetz_cs.n_nodes)
    @printf("  Currents:     %d\n", graetz_cs.n_currents)
    @printf("  G nnz:        %d\n", nnz(graetz_cs.G))
    @printf("  C nnz:        %d\n", nnz(graetz_cs.C))

    return (
        rc_size = MNA.system_size(rc_cs),
        graetz_size = MNA.system_size(graetz_cs)
    )
end

#==============================================================================#
# Profile Fast Operations
#==============================================================================#

function profile_fast_operations()
    println("\n" * "=" ^ 50)
    println("Fast Operation Allocations (per call)")
    println("=" ^ 50)

    # RC Circuit operations
    println("\nRC Circuit:")
    rc_circ = MNACircuit(rc_circuit)
    rc_ws = MNA.compile(rc_circ)
    rc_cs = rc_ws.structure
    n_rc = MNA.system_size(rc_cs)

    u_rc = ones(n_rc)
    du_rc = zeros(n_rc)
    resid_rc = zeros(n_rc)
    J_rc = copy(rc_cs.G + rc_cs.C)

    allocs_rebuild_rc = @allocated begin
        for _ in 1:1000
            MNA.fast_rebuild!(rc_ws, u_rc, 0.001)
        end
    end
    @printf("  fast_rebuild!:  %.1f bytes/call\n", allocs_rebuild_rc / 1000)

    allocs_residual_rc = @allocated begin
        for _ in 1:1000
            MNA.fast_residual!(resid_rc, du_rc, u_rc, rc_ws, 0.001)
        end
    end
    @printf("  fast_residual!: %.1f bytes/call\n", allocs_residual_rc / 1000)

    # Graetz Circuit operations
    println("\nGraetz Circuit:")
    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    graetz_cs = graetz_ws.structure
    n_graetz = MNA.system_size(graetz_cs)

    u_graetz = ones(n_graetz)
    du_graetz = zeros(n_graetz)
    resid_graetz = zeros(n_graetz)
    J_graetz = copy(graetz_cs.G + graetz_cs.C)

    allocs_rebuild_graetz = @allocated begin
        for _ in 1:1000
            MNA.fast_rebuild!(graetz_ws, u_graetz, 0.001)
        end
    end
    @printf("  fast_rebuild!:  %.1f bytes/call\n", allocs_rebuild_graetz / 1000)

    allocs_residual_graetz = @allocated begin
        for _ in 1:1000
            MNA.fast_residual!(resid_graetz, du_graetz, u_graetz, graetz_ws, 0.001)
        end
    end
    @printf("  fast_residual!: %.1f bytes/call\n", allocs_residual_graetz / 1000)

    return (
        rc_rebuild = allocs_rebuild_rc / 1000,
        rc_residual = allocs_residual_rc / 1000,
        graetz_rebuild = allocs_rebuild_graetz / 1000,
        graetz_residual = allocs_residual_graetz / 1000
    )
end

#==============================================================================#
# Profile Full Simulations
#==============================================================================#

function profile_simulation(circuit_builder, circuit_name, solver; tspan=(0.0, 0.1), dtmax=1e-5)
    println("\n--- $circuit_name with $(nameof(typeof(solver))) ---")

    # Warmup
    circuit = MNACircuit(circuit_builder)
    MNA.assemble!(circuit)

    if solver isa IDA
        prob = SciMLBase.DAEProblem(circuit, (0.0, 0.001); explicit_jacobian=true)
    else
        prob = SciMLBase.ODEProblem(circuit, (0.0, 0.001))
    end
    solve(prob, solver; dtmax=dtmax, abstol=1e-3, reltol=1e-3, maxiters=10000, dense=false)
    GC.gc()

    # Full simulation
    circuit2 = MNACircuit(circuit_builder)
    MNA.assemble!(circuit2)

    if solver isa IDA
        prob2 = SciMLBase.DAEProblem(circuit2, tspan; explicit_jacobian=true)
    else
        prob2 = SciMLBase.ODEProblem(circuit2, tspan)
    end

    allocs = @allocated begin
        sol = solve(prob2, solver; dtmax=dtmax, abstol=1e-3, reltol=1e-3, maxiters=1_000_000, dense=false)
    end

    sol = solve(prob2, solver; dtmax=dtmax, abstol=1e-3, reltol=1e-3, maxiters=1_000_000, dense=false)

    @printf("  Timepoints:    %d\n", length(sol.t))
    @printf("  NR iterations: %d\n", sol.stats.nnonliniter)
    if hasfield(typeof(sol.stats), :njacs)
        @printf("  Jacobian evals: %d\n", sol.stats.njacs)
    end
    @printf("  Memory:        %.2f MB\n", allocs / 1e6)
    @printf("  Per step:      %.2f KB\n", allocs / length(sol.t) / 1000)
    if sol.stats.nnonliniter > 0
        @printf("  Per NR iter:   %.2f KB\n", allocs / sol.stats.nnonliniter / 1000)
    end

    return (
        circuit = circuit_name,
        solver = nameof(typeof(solver)),
        timepoints = length(sol.t),
        nr_iters = sol.stats.nnonliniter,
        allocs = allocs
    )
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    sizes = compare_circuit_structure()
    ops = profile_fast_operations()

    println("\n" * "=" ^ 50)
    println("Full Simulation Comparison (0.1s, dtmax=1e-5)")
    println("=" ^ 50)

    results = []

    push!(results, profile_simulation(rc_circuit, "RC", FBDF()))
    push!(results, profile_simulation(graetz_circuit, "Graetz", FBDF()))

    push!(results, profile_simulation(rc_circuit, "RC", IDA(max_error_test_failures=20)))
    push!(results, profile_simulation(graetz_circuit, "Graetz", IDA(max_error_test_failures=20)))

    # Summary
    println("\n" * "=" ^ 70)
    println("SUMMARY")
    println("=" ^ 70)

    println("\nCircuit sizes:")
    @printf("  RC:     %d unknowns\n", sizes.rc_size)
    @printf("  Graetz: %d unknowns\n", sizes.graetz_size)

    println("\nFast operation allocations:")
    @printf("  RC fast_rebuild!:     %.1f bytes\n", ops.rc_rebuild)
    @printf("  Graetz fast_rebuild!: %.1f bytes\n", ops.graetz_rebuild)

    println("\nSimulation allocations per step:")
    for r in results
        @printf("  %-15s %-10s: %.2f KB/step, %.2f KB/NR-iter\n",
                r.circuit, string(r.solver),
                r.allocs/r.timepoints/1000,
                r.nr_iters > 0 ? r.allocs/r.nr_iters/1000 : 0.0)
    end

    # Ratio analysis
    println("\nGraetz/RC allocation ratio:")
    rc_fbdf = results[1]
    graetz_fbdf = results[2]
    @printf("  Per step:    %.2fx\n", (graetz_fbdf.allocs/graetz_fbdf.timepoints) /
            (rc_fbdf.allocs/rc_fbdf.timepoints))
    if graetz_fbdf.nr_iters > 0 && rc_fbdf.nr_iters > 0
        @printf("  Per NR iter: %.2fx\n", (graetz_fbdf.allocs/graetz_fbdf.nr_iters) /
                (rc_fbdf.allocs/rc_fbdf.nr_iters))
    end

    println("\n" * "=" ^ 70)
    println("ANALYSIS")
    println("=" ^ 70)

    println("""

Key factors for higher Graetz allocations:

1. **Larger system size**: Graetz has more nodes/equations than RC
   - More memory for vectors, matrices, factorizations

2. **Nonlinear devices (diodes)**: Graetz has 4 VA diodes
   - Newton iterations update the Jacobian more frequently
   - More Jacobian evaluations = more LU factorizations

3. **More NR iterations per step**: Nonlinear circuits need more iterations
   - Each Newton iteration may trigger Jacobian update

4. **Solution storage**: Scales with system size × timepoints
   - Graetz has larger vectors to store

5. **Jacobian refactorization**: Key allocation source
   - For nonlinear circuits, Jacobian changes each step
   - FBDF refactorizes when Newton convergence is slow
""")
end

main()
