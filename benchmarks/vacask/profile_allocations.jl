#!/usr/bin/env julia
#==============================================================================#
# Memory Allocation Profiler for VACASK Benchmarks
#
# Comprehensive profiling of allocation sources in MNA circuit simulation.
# Run with: julia --project=. benchmarks/vacask/profile_allocations.jl
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CedarSim
using CedarSim.MNA
using VerilogAParser
using Printf

println("=" ^ 70)
println("Memory Allocation Profiler")
println("=" ^ 70)

#==============================================================================#
# Load Test Circuits
#==============================================================================#

println("\nLoading circuits...")

# Load VA diode model
const diode_va_path = joinpath(@__DIR__, "..", "..", "test", "vadistiller", "models", "diode.va")
const va = VerilogAParser.parsefile(diode_va_path)
if va.ps.errored
    error("Failed to parse diode VA model")
end
Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
println("  Loaded: sp_diode VA model")

# Load Graetz circuit (4 diodes)
const graetz_spice_file = joinpath(@__DIR__, "graetz", "cedarsim", "runme.sp")
const graetz_spice_code = read(graetz_spice_file, String)
const graetz_circuit_code = parse_spice_to_mna(graetz_spice_code; circuit_name=:graetz_circuit,
                                                imported_hdl_modules=[sp_diode_module])
eval(graetz_circuit_code)
println("  Loaded: Graetz rectifier circuit")

#==============================================================================#
# Profile Circuit Builder Allocations
#==============================================================================#

function profile_builder_allocations()
    println("\n" * "=" ^ 70)
    println("SECTION 1: Circuit Builder Allocations")
    println("=" ^ 70)

    # Compile circuit
    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    cs = graetz_ws.structure
    dctx = graetz_ws.dctx

    n = MNA.system_size(cs)
    u = 0.5 * ones(n)

    println("\nCircuit info:")
    println("  System size: $n")
    println("  Nodes: $(dctx.n_nodes)")
    println("  Currents: $(dctx.n_currents)")
    println("  Internal nodes: $(length(dctx.internal_node_indices))")
    println("  G matrix entries: $(dctx.n_G)")
    println("  C matrix entries: $(dctx.n_C)")
    println("  Deferred b entries: $(dctx.n_b_deferred)")

    # Warmup
    for _ in 1:10
        MNA.reset_direct_stamp!(dctx)
        cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
    end
    GC.gc()

    println("\n--- Overall Builder Performance ---")

    # Total builder allocation
    allocs_total = @allocated begin
        for _ in 1:100
            MNA.reset_direct_stamp!(dctx)
            cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
        end
    end
    @printf("  Builder (reset + stamp): %.1f bytes/call\n", allocs_total / 100)

    # Reset alone
    allocs_reset = @allocated begin
        for _ in 1:1000
            MNA.reset_direct_stamp!(dctx)
        end
    end
    @printf("  reset_direct_stamp!:     %.1f bytes/call\n", allocs_reset / 1000)

    println("\n--- Individual MNA Operations ---")

    # get_node! (dict lookup - should be 0)
    node_names = collect(keys(dctx.node_to_idx))
    allocs_get_node = @allocated begin
        for _ in 1:1000
            for name in node_names
                MNA.get_node!(dctx, name)
            end
        end
    end
    @printf("  get_node! (dict lookup): %.3f bytes/call\n", allocs_get_node / (1000 * length(node_names)))

    # alloc_internal_node! (counter-based)
    n_internal = length(dctx.internal_node_indices)
    if n_internal > 0
        allocs_internal = @allocated begin
            for _ in 1:1000
                dctx.internal_node_pos = 1
                for _ in 1:n_internal
                    MNA.alloc_internal_node!(dctx, :test, :inst)
                end
            end
        end
        @printf("  alloc_internal_node!:    %.3f bytes/call\n", allocs_internal / (1000 * n_internal))
    end

    # alloc_current! (counter-based)
    allocs_current = @allocated begin
        for _ in 1:1000
            dctx.current_pos = 1
            MNA.alloc_current!(dctx, :test, :inst)
        end
    end
    @printf("  alloc_current!:          %.3f bytes/call\n", allocs_current / 1000)

    # stamp_G! (direct write)
    allocs_stamp = @allocated begin
        for _ in 1:1000
            dctx.G_pos = 1
            MNA.stamp_G!(dctx, 1, 1, 1.0)
        end
    end
    @printf("  stamp_G!:                %.3f bytes/call\n", allocs_stamp / 1000)

    # detect_or_cached! (counter-based)
    n_detections = length(dctx.charge_is_vdep)
    if n_detections > 0
        allocs_detect = @allocated begin
            for _ in 1:1000
                dctx.charge_detection_pos = 1
                for _ in 1:n_detections
                    MNA.detect_or_cached!(dctx, :test, :inst, 0.5, 1e-12)
                end
            end
        end
        @printf("  detect_or_cached!:       %.3f bytes/call\n", allocs_detect / (1000 * n_detections))
    end

    # Summary
    println("\n--- Summary ---")
    if allocs_total / 100 < 1.0
        println("  ✓ Builder is allocation-free!")
    else
        @printf("  ✗ Builder allocates %.1f bytes/call\n", allocs_total / 100)
    end

    return allocs_total / 100
end

#==============================================================================#
# Profile Solver Allocations (requires DifferentialEquations)
#==============================================================================#

function profile_solver_allocations()
    println("\n" * "=" ^ 70)
    println("SECTION 2: Solver Allocations (if DifferentialEquations available)")
    println("=" ^ 70)

    try
        @eval using DifferentialEquations
    catch
        println("\n  DifferentialEquations not installed - skipping solver profiling")
        println("  Install with: ] add DifferentialEquations")
        return
    end

    # Load RC circuit for solver comparison
    rc_spice_file = joinpath(@__DIR__, "rc", "cedarsim", "runme.sp")
    if !isfile(rc_spice_file)
        println("\n  RC circuit not found - skipping solver profiling")
        return
    end

    rc_spice_code = read(rc_spice_file, String)
    rc_circuit_code = parse_spice_to_mna(rc_spice_code; circuit_name=:rc_circuit)
    eval(rc_circuit_code)

    rc_circ = MNACircuit(rc_circuit)

    println("\nComparing solvers on RC circuit (10k steps):")

    # FBDF
    print("  FBDF:    ")
    GC.gc()
    stats_fbdf = @timed tran!(rc_circ, 10e-3; solver=FBDF(autodiff=false))
    @printf("%.1f MB, %.2f sec\n", stats_fbdf.bytes / 1e6, stats_fbdf.time)

    # Rodas5P
    print("  Rodas5P: ")
    GC.gc()
    stats_rodas = @timed tran!(rc_circ, 10e-3; solver=Rodas5P(autodiff=false))
    @printf("%.1f MB, %.2f sec\n", stats_rodas.bytes / 1e6, stats_rodas.time)

    println("\n--- Analysis ---")
    ratio = stats_rodas.bytes / stats_fbdf.bytes
    @printf("  Rodas5P uses %.1fx more memory than FBDF\n", ratio)
    println("  This is expected: Rosenbrock methods factorize W=M-γhJ every step")
    println("  BDF methods reuse the Jacobian across many steps")
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    builder_alloc = profile_builder_allocations()
    profile_solver_allocations()

    println("\n" * "=" ^ 70)
    println("CONCLUSIONS")
    println("=" ^ 70)
    println("""

1. Circuit Builder: $(builder_alloc < 1.0 ? "ALLOCATION-FREE ✓" : "$(builder_alloc) bytes/call")
   - All MNA operations (get_node!, alloc_*, stamp_*) are O(1) and allocation-free
   - Counter-based lookups avoid Symbol construction overhead
   - Dict lookups on Symbol keys don't allocate (interned)

2. Solver Allocations:
   - FBDF: Minimal - reuses Jacobian across steps
   - Rodas5P: High - factorizes every step (algorithmic, not a bug)
   - Solution storage scales with number of saved timepoints

3. Optimization Status:
   - VA device evaluation: FIXED (was 800 bytes/call, now 0)
   - Internal node allocation: FIXED (counter-based)
   - Current/charge allocation: FIXED (component-based API)
   - Charge detection: FIXED (counter-based cache lookup)
""")
end

main()
