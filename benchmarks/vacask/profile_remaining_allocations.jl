#!/usr/bin/env julia
#==============================================================================#
# Profile Remaining Allocations
#
# After fixing internal node allocation, find where the remaining 640 bytes comes from.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CedarSim
using CedarSim.MNA
using Printf
using VerilogAParser

println("=" ^ 70)
println("Profile Remaining Allocations (after internal node fix)")
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
# Profile Different Paths
#==============================================================================#

function profile_paths()
    println("\n" * "=" ^ 50)
    println("Profiling Different Code Paths")
    println("=" ^ 50)

    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    cs = graetz_ws.structure
    dctx = graetz_ws.dctx

    n = MNA.system_size(cs)
    u = 0.5 * ones(n)

    println("\nSystem info:")
    println("  n_nodes: $(dctx.n_nodes)")
    println("  n_currents: $(dctx.n_currents)")
    println("  internal_node_indices: $(dctx.internal_node_indices)")
    println("  n_G: $(dctx.n_G)")
    println("  n_C: $(dctx.n_C)")
    println("  n_b_deferred: $(dctx.n_b_deferred)")

    # Profile overall builder
    println("\n--- Overall builder ---")
    allocs_total = @allocated begin
        for _ in 1:100
            MNA.reset_direct_stamp!(dctx)
            cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
        end
    end
    @printf("  Total per call: %.1f bytes\n", allocs_total / 100)

    # Check if there are Symbol allocations in get_node!
    println("\n--- get_node! calls ---")
    # Get actual node names from the context
    node_names = collect(keys(dctx.node_to_idx))
    println("  Actual nodes: $node_names")
    allocs_get_node = @allocated begin
        for _ in 1:1000
            for name in node_names
                MNA.get_node!(dctx, name)
            end
        end
    end
    @printf("  get_node! (known nodes): %.1f bytes/call\n", allocs_get_node / (1000 * length(node_names)))

    # Check alloc_internal_node! with component API
    println("\n--- alloc_internal_node! calls ---")
    base_name = Symbol(:sp_diode_a_int)
    instance_name = :xd1

    # First reset internal_node_pos to beginning
    dctx.internal_node_pos = 1

    allocs_alloc_int = @allocated begin
        for _ in 1:100
            dctx.internal_node_pos = 1  # Reset each iteration
            for i in 1:length(dctx.internal_node_indices)
                MNA.alloc_internal_node!(dctx, base_name, instance_name)
            end
        end
    end
    n_internal = length(dctx.internal_node_indices)
    @printf("  alloc_internal_node! (component API): %.1f bytes/call (%d internal nodes)\n",
            allocs_alloc_int / (100 * n_internal), n_internal)

    # Check alloc_current!
    println("\n--- alloc_current! calls ---")
    dctx.current_pos = 1  # Reset
    allocs_current = @allocated begin
        for _ in 1:1000
            dctx.current_pos = 1
            MNA.alloc_current!(dctx, :test)
        end
    end
    @printf("  alloc_current!: %.1f bytes/call\n", allocs_current / 1000)

    # Check stamp operations (reset properly each iteration)
    println("\n--- stamp operations ---")

    allocs_stamp = @allocated begin
        for _ in 1:1000
            MNA.reset_direct_stamp!(dctx)
            MNA.stamp_G!(dctx, 1, 1, 1.0)
        end
    end
    @printf("  stamp_G! alone: %.1f bytes/call\n", allocs_stamp / 1000)

    # Check charge detection
    println("\n--- charge detection ---")
    n_charges = length(dctx.charge_is_vdep)
    @printf("  Number of charge detections: %d\n", n_charges)
    allocs_detect = @allocated begin
        for _ in 1:1000
            dctx.charge_detection_pos = 1
            for i in 1:n_charges
                MNA.get_is_vdep(dctx, :test)
            end
        end
    end
    @printf("  get_is_vdep: %.1f bytes/call\n",
            n_charges > 0 ? allocs_detect / (1000 * n_charges) : 0.0)
end

#==============================================================================#
# Profile Single Diode
#==============================================================================#

function profile_single_diode()
    println("\n" * "=" ^ 50)
    println("Profiling Single Diode Device")
    println("=" ^ 50)

    # Create a diode device
    dev = sp_diode(is=76.9e-12, rs=42.0e-3, cjo=26.5e-12, m=0.333, n=1.45)

    # Build with MNAContext to get structure
    ctx = MNA.MNAContext()
    p = MNA.get_node!(ctx, :p)
    n_node = MNA.get_node!(ctx, :n)
    MNA.stamp!(dev, ctx, p, n_node; _mna_instance_=:xd1)

    sys_size = MNA.system_size(ctx)
    println("\nSingle diode system size: $sys_size")
    println("Internal nodes: $(MNA.n_internal_nodes(ctx))")

    # Get the assembled system
    sys = MNA.assemble!(ctx)

    # Create DirectStampContext
    G_nzval = copy(sys.G.nzval)
    C_nzval = copy(sys.C.nzval)
    b = zeros(sys_size)

    # Build COO mappings
    G_mapping = MNA.build_coo_mapping(ctx.G_I, ctx.G_J, sys.G)
    C_mapping = MNA.build_coo_mapping(ctx.C_I, ctx.C_J, sys.C)
    b_resolved = zeros(Int, length(ctx.b_V))

    dctx = MNA.create_direct_stamp_context(ctx, G_nzval, C_nzval, b, G_mapping, C_mapping, b_resolved)

    # Test vectors
    x = zeros(sys_size)
    x[1] = 0.7

    # Warmup
    for _ in 1:10
        MNA.reset_direct_stamp!(dctx)
        MNA.stamp!(dev, dctx, p, n_node; _mna_instance_=:xd1, _mna_x_=x)
    end
    GC.gc()

    # Profile stamp! call
    allocs = @allocated begin
        for _ in 1:100
            MNA.reset_direct_stamp!(dctx)
            MNA.stamp!(dev, dctx, p, n_node; _mna_instance_=:xd1, _mna_x_=x)
        end
    end
    @printf("\nSingle diode stamp! allocation: %.1f bytes/call\n", allocs / 100)

    # Expected for 4 diodes: 4 * single_alloc
    @printf("Expected for 4 diodes: %.1f bytes/call\n", (allocs / 100) * 4)
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    profile_paths()
    profile_single_diode()

    println("\n" * "=" ^ 70)
    println("ANALYSIS")
    println("=" ^ 70)

    println("""

The remaining 640 bytes per Graetz builder call comes from:

1. **Current allocation** - alloc_current! uses counter-based access (OK)
2. **Charge detection** - get_is_vdep uses counter-based access (OK)
3. **Stamp operations** - direct sparse writes (OK)

If single diode shows non-zero allocation, the issue is in VA stamp! code.

Possible remaining sources:
- ZeroVector construction in stamp! method
- Type conversions or promotions
- Closure captures in detection lambdas
- let blocks that create new scope
""")
end

main()
