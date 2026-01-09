#!/usr/bin/env julia
#==============================================================================#
# Profile VA Allocation Source
#
# Use Julia's allocation profiler to find exactly where the 800 bytes/call
# allocation is coming from in the VA diode evaluation.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CedarSim
using CedarSim.MNA
using Printf
using Profile
using VerilogAParser

println("=" ^ 70)
println("Profile VA Allocation Source")
println("=" ^ 70)

#==============================================================================#
# Load Graetz Circuit
#==============================================================================#

println("\nLoading Graetz Circuit...")
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
# Profile with --track-allocation
#==============================================================================#

function profile_allocations_detailed()
    println("\n" * "=" ^ 50)
    println("Detailed Allocation Tracking")
    println("=" ^ 50)

    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    cs = graetz_ws.structure
    dctx = graetz_ws.dctx

    n = MNA.system_size(cs)
    u = ones(n)

    # Warmup
    for _ in 1:10
        MNA.reset_direct_stamp!(dctx)
        cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
    end
    GC.gc()

    # Profile with allocation tracking
    println("\nRunning allocation profiler...")
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1.0 begin
        for _ in 1:100
            MNA.reset_direct_stamp!(dctx)
            cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
        end
    end

    # Get results
    results = Profile.Allocs.fetch()
    println("\nAllocation profile results:")
    println("Total allocations: $(length(results.allocs))")

    # Group by type
    type_counts = Dict{Any, Int}()
    type_sizes = Dict{Any, Int}()
    for alloc in results.allocs
        t = alloc.type
        type_counts[t] = get(type_counts, t, 0) + 1
        type_sizes[t] = get(type_sizes, t, 0) + alloc.size
    end

    println("\nTop allocation types by count:")
    sorted_counts = sort(collect(type_counts), by=x->x[2], rev=true)
    for (t, count) in sorted_counts[1:min(10, length(sorted_counts))]
        size = type_sizes[t]
        @printf("  %6d allocations, %8d bytes total: %s\n", count, size, t)
    end

    println("\nTop allocation types by size:")
    sorted_sizes = sort(collect(type_sizes), by=x->x[2], rev=true)
    for (t, size) in sorted_sizes[1:min(10, length(sorted_sizes))]
        count = type_counts[t]
        @printf("  %8d bytes, %6d allocations: %s\n", size, count, t)
    end
end

#==============================================================================#
# Manual inspection of builder
#==============================================================================#

function inspect_builder()
    println("\n" * "=" ^ 50)
    println("Builder Inspection")
    println("=" ^ 50)

    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    cs = graetz_ws.structure
    dctx = graetz_ws.dctx

    n = MNA.system_size(cs)
    u = ones(n)

    # Show builder type
    println("\nBuilder function: $(typeof(cs.builder))")
    println("Params type: $(typeof(cs.params))")
    println("Spec type: $(typeof(cs.spec))")

    # Check if there are any captured variables causing boxing
    println("\nInspecting builder methods...")
    methods_list = methods(cs.builder)
    println("Number of methods: $(length(methods_list))")

    # Profile individual components
    println("\nBreaking down the builder call...")

    # The builder is graetz_circuit which stamps multiple devices
    # Let's trace through what happens

    # First, let's see if the allocation is from specific operations
    MNA.reset_direct_stamp!(dctx)

    # Measure just the stamp operations
    allocs_before = @allocated begin
        # This is what the builder does internally
        # We need to examine the generated graetz_circuit function
    end

    println("\nTo debug further, we need to examine the generated circuit function.")
    println("The builder is: graetz_circuit")
    println("\nLet's check what types are being allocated...")
end

#==============================================================================#
# Test individual stamp operations
#==============================================================================#

function test_stamp_operations()
    println("\n" * "=" ^ 50)
    println("Testing Individual Stamp Operations")
    println("=" ^ 50)

    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    cs = graetz_ws.structure
    dctx = graetz_ws.dctx

    n = MNA.system_size(cs)

    # Test basic stamp operations
    println("\nBasic stamp operations:")

    # stamp_G!
    allocs_stamp_g = @allocated begin
        for _ in 1:1000
            MNA.stamp_G!(dctx, 1, 1, 1.0)
            MNA.stamp_G!(dctx, 1, 2, -1.0)
        end
    end
    @printf("  stamp_G!: %.1f bytes/call\n", allocs_stamp_g / 2000)

    # stamp_C!
    allocs_stamp_c = @allocated begin
        for _ in 1:1000
            MNA.stamp_C!(dctx, 1, 1, 1.0)
        end
    end
    @printf("  stamp_C!: %.1f bytes/call\n", allocs_stamp_c / 1000)

    # stamp_b!
    allocs_stamp_b = @allocated begin
        for _ in 1:1000
            MNA.stamp_b!(dctx, 1, 1.0)
        end
    end
    @printf("  stamp_b!: %.1f bytes/call\n", allocs_stamp_b / 1000)

    # get_node!
    allocs_get_node = @allocated begin
        for _ in 1:1000
            MNA.get_node!(dctx, :inp)
        end
    end
    @printf("  get_node!: %.1f bytes/call\n", allocs_get_node / 1000)
end

#==============================================================================#
# Check VA device stamp method
#==============================================================================#

function check_va_stamp()
    println("\n" * "=" ^ 50)
    println("Checking VA Device stamp! Method")
    println("=" ^ 50)

    # Get the sp_diode type
    println("\nsp_diode type: $(sp_diode)")

    # Check stamp! method
    println("\nstamp! methods for sp_diode:")
    for m in methods(MNA.stamp!, (sp_diode, Any, Vararg))
        println("  $m")
    end

    # Create a test device
    dev = sp_diode(is=76.9e-12, rs=42.0e-3, cjo=26.5e-12, m=0.333, n=1.45)
    println("\nTest device: $dev")

    # Create a simple context for testing
    ctx = MNA.MNAContext()
    p = MNA.get_node!(ctx, :p)
    n_node = MNA.get_node!(ctx, :n)

    # Profile the stamp! call
    println("\nProfiling stamp! call...")

    # Warmup
    for _ in 1:10
        ctx2 = MNA.MNAContext()
        p2 = MNA.get_node!(ctx2, :p)
        n2 = MNA.get_node!(ctx2, :n)
        MNA.stamp!(dev, ctx2, p2, n2; instance_name=:xd1)
    end
    GC.gc()

    # Measure
    allocs = @allocated begin
        for _ in 1:100
            ctx3 = MNA.MNAContext()
            p3 = MNA.get_node!(ctx3, :p)
            n3 = MNA.get_node!(ctx3, :n)
            MNA.stamp!(dev, ctx3, p3, n3; instance_name=:xd1)
        end
    end

    @printf("\nstamp! allocation: %.1f bytes/call (includes context creation)\n", allocs / 100)

    # Now test with DirectStampContext
    println("\nTesting with DirectStampContext (restamping mode)...")

    # Build once to get structure
    ctx_init = MNA.MNAContext()
    p_init = MNA.get_node!(ctx_init, :p)
    n_init = MNA.get_node!(ctx_init, :n)
    MNA.stamp!(dev, ctx_init, p_init, n_init; instance_name=:xd1)
    sys = MNA.assemble!(ctx_init)

    # Now we need to create a DirectStampContext and test restamping
    # This requires going through the compiled structure path
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    test_stamp_operations()
    check_va_stamp()

    # Run allocation profiler (requires Julia to be started with special flags)
    try
        profile_allocations_detailed()
    catch e
        println("\nAllocation profiler failed (may need --track-allocation flag)")
        println("Error: $e")
    end

    inspect_builder()

    println("\n" * "=" ^ 70)
    println("NEXT STEPS")
    println("=" ^ 70)

    println("""

To find the exact allocation source:

1. Look at the generated code for graetz_circuit
2. Check the stamp! method for sp_diode
3. Look for:
   - Closure captures (boxing)
   - Dynamic dispatch
   - String operations
   - Array allocations
   - Temporary struct allocations

The 800 bytes per call for 4 diodes = 200 bytes per diode.
This suggests each diode stamp! allocates about 200 bytes.
""")
end

main()
