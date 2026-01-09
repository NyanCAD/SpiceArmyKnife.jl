#!/usr/bin/env julia
#==============================================================================#
# Profile VA Allocation - Exact Source
#
# Isolate exactly where the 800 bytes/call allocation comes from.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CedarSim
using CedarSim.MNA
using Printf
using VerilogAParser
using ForwardDiff
using ForwardDiff: Dual, value, partials

println("=" ^ 70)
println("Profile VA Allocation - Exact Source")
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
# Profile Core Operations
#==============================================================================#

function profile_dual_creation()
    println("\n" * "=" ^ 50)
    println("Dual Creation Profiling")
    println("=" ^ 50)

    # Small circuit (like RC) - 3 nodes
    println("\nSmall circuit (3 nodes):")
    allocs_3 = @allocated begin
        for _ in 1:1000
            partials = (1.0, 0.0, 0.0)
            d1 = Dual{MNA.JacobianTag}(0.7, partials...)
            d2 = Dual{MNA.JacobianTag}(0.0, 0.0, 1.0, 0.0)
            d3 = Dual{MNA.JacobianTag}(0.0, 0.0, 0.0, 1.0)
        end
    end
    @printf("  3 duals: %.1f bytes/iter\n", allocs_3 / 1000)

    # Larger circuit - 8 nodes (Graetz has ~8 nodes plus internal)
    println("\nGraetz circuit (~10+ nodes):")
    allocs_10 = @allocated begin
        for _ in 1:1000
            # Create 10 duals with 10-element partials
            duals = ntuple(Val(10)) do i
                partials = ntuple(j -> i == j ? 1.0 : 0.0, Val(10))
                Dual{MNA.JacobianTag}(0.7, partials...)
            end
        end
    end
    @printf("  10 duals with 10 partials each: %.1f bytes/iter\n", allocs_10 / 1000)

    # Test with 17 nodes (Graetz system size)
    allocs_17 = @allocated begin
        for _ in 1:1000
            duals = ntuple(Val(17)) do i
                partials = ntuple(j -> i == j ? 1.0 : 0.0, Val(17))
                Dual{MNA.JacobianTag}(0.7, partials...)
            end
        end
    end
    @printf("  17 duals with 17 partials each: %.1f bytes/iter\n", allocs_17 / 1000)
end

#==============================================================================#
# Profile VA Device Evaluation
#==============================================================================#

function profile_diode_operations()
    println("\n" * "=" ^ 50)
    println("Diode Operations Profiling")
    println("=" ^ 50)

    # Create a diode device
    dev = sp_diode(is=76.9e-12, rs=42.0e-3, cjo=26.5e-12, m=0.333, n=1.45)
    println("Device type: $(typeof(dev))")

    # Create MNA context and stamp once to get structure
    println("\nInitial stamp (structure discovery):")
    ctx = MNA.MNAContext()
    p = MNA.get_node!(ctx, :p)
    n_node = MNA.get_node!(ctx, :n)

    # Profile the stamp! call with MNAContext
    allocs_stamp_mna = @allocated begin
        for _ in 1:100
            ctx_new = MNA.MNAContext()
            p_new = MNA.get_node!(ctx_new, :p)
            n_new = MNA.get_node!(ctx_new, :n)
            MNA.stamp!(dev, ctx_new, p_new, n_new; _mna_instance_=:xd1)
        end
    end
    @printf("  stamp! with fresh MNAContext: %.1f bytes/call\n", allocs_stamp_mna / 100)

    # Test stamp! with different x vectors
    println("\nStamp with solution vector:")
    ctx2 = MNA.MNAContext()
    p2 = MNA.get_node!(ctx2, :p)
    n2 = MNA.get_node!(ctx2, :n)
    MNA.stamp!(dev, ctx2, p2, n2; _mna_instance_=:xd1)
    sys_size = MNA.system_size(ctx2)
    println("  System size after stamp: $sys_size")

    x_test = zeros(sys_size)
    x_test[1] = 0.7  # Vp

    allocs_stamp_x = @allocated begin
        for _ in 1:100
            ctx_new = MNA.MNAContext()
            p_new = MNA.get_node!(ctx_new, :p)
            n_new = MNA.get_node!(ctx_new, :n)
            MNA.stamp!(dev, ctx_new, p_new, n_new; _mna_instance_=:xd1, _mna_x_=x_test)
        end
    end
    @printf("  stamp! with x vector: %.1f bytes/call\n", allocs_stamp_x / 100)
end

#==============================================================================#
# Profile Graetz Builder
#==============================================================================#

function profile_graetz_builder()
    println("\n" * "=" ^ 50)
    println("Graetz Builder Profiling")
    println("=" ^ 50)

    # Build once to get structure
    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    cs = graetz_ws.structure
    dctx = graetz_ws.dctx

    n = MNA.system_size(cs)
    println("System size: $n")

    u = ones(n)
    u[1:8] .= [0.7, 0.3, 0.5, 0.2, 0.6, 0.4, 0.1, 0.8]

    # Profile just reset + builder
    println("\nBuilder breakdown:")

    # Reset
    allocs_reset = @allocated begin
        for _ in 1:1000
            MNA.reset_direct_stamp!(dctx)
        end
    end
    @printf("  reset_direct_stamp!: %.1f bytes/call\n", allocs_reset / 1000)

    # Builder alone
    MNA.reset_direct_stamp!(dctx)
    allocs_builder = @allocated begin
        for _ in 1:100
            MNA.reset_direct_stamp!(dctx)
            cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
        end
    end
    @printf("  builder call: %.1f bytes/call (includes reset)\n", allocs_builder / 100)

    # Just builder without reset
    MNA.reset_direct_stamp!(dctx)
    cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)  # Warmup

    allocs_builder_only = @allocated begin
        for _ in 1:100
            cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
        end
    end
    @printf("  builder only (no reset): %.1f bytes/call\n", allocs_builder_only / 100)

    # Check if allocation varies with u
    println("\nAllocation with different operating points:")
    for voltage in [0.0, 0.5, 0.7, 1.0]
        u_test = voltage * ones(n)
        MNA.reset_direct_stamp!(dctx)

        allocs_v = @allocated begin
            for _ in 1:100
                MNA.reset_direct_stamp!(dctx)
                cs.builder(cs.params, cs.spec, 0.001; x=u_test, ctx=dctx)
            end
        end
        @printf("  V=%.1f: %.1f bytes/call\n", voltage, allocs_v / 100)
    end
end

#==============================================================================#
# Profile Detection Phase
#==============================================================================#

function profile_detection()
    println("\n" * "=" ^ 50)
    println("Charge Detection Profiling")
    println("=" ^ 50)

    # The detect_or_cached! function and is_voltage_dependent_charge
    # These might be causing allocations

    # Simple detection test
    contrib_fn = V -> MNA.va_ddt(1e-12 * V)  # Linear capacitor

    allocs_detect = @allocated begin
        for _ in 1:100
            MNA.is_voltage_dependent_charge(contrib_fn, 0.7, 0.0)
        end
    end
    @printf("\nis_voltage_dependent_charge (linear): %.1f bytes/call\n", allocs_detect / 100)

    # Nonlinear capacitor
    contrib_fn_nl = V -> MNA.va_ddt(1e-12 * V^2)  # Nonlinear

    allocs_detect_nl = @allocated begin
        for _ in 1:100
            MNA.is_voltage_dependent_charge(contrib_fn_nl, 0.7, 0.0)
        end
    end
    @printf("is_voltage_dependent_charge (nonlinear): %.1f bytes/call\n", allocs_detect_nl / 100)

    # Test with context-based detection
    ctx = MNA.MNAContext()

    allocs_cached = @allocated begin
        for _ in 1:100
            ctx.charge_detection_pos = 1
            MNA.detect_or_cached!(ctx, :test, contrib_fn, 0.7, 0.0)
        end
    end
    @printf("detect_or_cached! (linear, context): %.1f bytes/call\n", allocs_cached / 100)
end

#==============================================================================#
# Profile ForwardDiff Operations
#==============================================================================#

function profile_forwarddiff()
    println("\n" * "=" ^ 50)
    println("ForwardDiff Operations Profiling")
    println("=" ^ 50)

    # Test basic operations that happen during stamp!

    # 1. exp() with dual
    Vt = 0.026
    Is = 1e-14

    allocs_exp = @allocated begin
        for _ in 1:1000
            V_dual = Dual{MNA.JacobianTag}(0.7, 1.0, 0.0)
            result = Is * (exp(V_dual / Vt) - 1)
        end
    end
    @printf("\nexp() with dual: %.1f bytes/call\n", allocs_exp / 1000)

    # 2. log() with dual
    allocs_log = @allocated begin
        for _ in 1:1000
            V_dual = Dual{MNA.JacobianTag}(0.7, 1.0, 0.0)
            result = log(V_dual + 1.0)
        end
    end
    @printf("log() with dual: %.1f bytes/call\n", allocs_log / 1000)

    # 3. pow() with dual
    allocs_pow = @allocated begin
        for _ in 1:1000
            V_dual = Dual{MNA.JacobianTag}(0.7, 1.0, 0.0)
            result = (1 - V_dual / 0.7)^0.5
        end
    end
    @printf("pow() with dual: %.1f bytes/call\n", allocs_pow / 1000)

    # 4. Complex expression (diode-like)
    allocs_diode = @allocated begin
        for _ in 1:1000
            V_dual = Dual{MNA.JacobianTag}(0.5, 1.0, 0.0)
            I = Is * (exp(V_dual / Vt) - 1)
            C = 1e-12 * (1 - V_dual / 0.7)^(-0.5)
            total = I + MNA.va_ddt(C * V_dual)
        end
    end
    @printf("Complex diode expression: %.1f bytes/call\n", allocs_diode / 1000)
end

#==============================================================================#
# Check Symbol/String Operations
#==============================================================================#

function profile_symbol_ops()
    println("\n" * "=" ^ 50)
    println("Symbol/String Operations")
    println("=" ^ 50)

    # Symbol creation is often a source of allocations

    allocs_symbol = @allocated begin
        for _ in 1:1000
            s = Symbol("xd1")
        end
    end
    @printf("\nSymbol(\"xd1\"): %.1f bytes/call\n", allocs_symbol / 1000)

    allocs_symbol_interp = @allocated begin
        for i in 1:1000
            s = Symbol("xd", i)
        end
    end
    @printf("Symbol(\"xd\", i): %.1f bytes/call\n", allocs_symbol_interp / 1000)

    # alloc_internal_node! uses Symbol
    ctx = MNA.MNAContext()
    MNA.get_node!(ctx, :p)

    allocs_alloc_int = @allocated begin
        for i in 1:1000
            name = Symbol("int_", i)
            MNA.alloc_internal_node!(ctx, name)
        end
    end
    @printf("alloc_internal_node!: %.1f bytes/call\n", allocs_alloc_int / 1000)
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    profile_dual_creation()
    profile_forwarddiff()
    profile_symbol_ops()
    profile_detection()
    profile_diode_operations()
    profile_graetz_builder()

    println("\n" * "=" ^ 70)
    println("ANALYSIS")
    println("=" ^ 70)

    println("""

Expected allocation sources (per VA device stamp!):

1. **Internal node allocation** - Symbol creation
   - Each stamp! creates instance-qualified symbols for internal nodes
   - Symbol("device_name.a_int") allocates ~64+ bytes per unique name

2. **ForwardDiff Dual creation**
   - Creating duals with large partials tuples
   - For Graetz (17-element system), each Dual has 17 partials
   - NTuple{17,Float64} = 136 bytes, but should be stack-allocated

3. **Charge detection lambdas**
   - `is_voltage_dependent_charge` creates closures
   - ForwardDiff.derivative may allocate

4. **dict/vector operations in MNAContext**
   - node_to_idx dictionary lookups/insertions
   - push! to G_I, G_J, etc vectors

Key insight: For DirectStampContext (fast path), most of these should NOT allocate:
- No new Symbol creation (uses cached node_to_idx)
- No new Dual allocation (should be stack)
- No vector push! (uses counter-based index)

If DirectStampContext still allocates 800 bytes, the issue is likely:
- Closure capture in generated code
- Type instability causing boxing
- Dynamic dispatch
""")
end

main()
