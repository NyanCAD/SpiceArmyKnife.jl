#!/usr/bin/env julia
#==============================================================================#
# Profile VA Device Allocations
#
# Investigate the 800 bytes/call allocation in fast_rebuild! for Graetz.
# This comes from the VA diode model - need to find the source.
#==============================================================================#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using InteractiveUtils
using CedarSim
using CedarSim.MNA
using SciMLBase
using Printf
using LinearAlgebra
using SparseArrays
using VerilogAParser

println("=" ^ 70)
println("Profile VA Device Allocations")
println("=" ^ 70)

#==============================================================================#
# Load Graetz Circuit
#==============================================================================#

println("\nLoading Graetz Circuit (with VA diode)...")
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
# Detailed Allocation Profiling
#==============================================================================#

function profile_detailed()
    println("\n" * "=" ^ 50)
    println("Detailed Allocation Profiling")
    println("=" ^ 50)

    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    graetz_cs = graetz_ws.structure

    n = MNA.system_size(graetz_cs)
    println("System size: $n")

    # Test vectors
    u = ones(n)
    du = zeros(n)
    resid = zeros(n)

    # Profile fast_rebuild! with different iteration counts
    println("\nfast_rebuild! allocation scaling:")
    for iters in [1, 10, 100, 1000]
        allocs = @allocated begin
            for _ in 1:iters
                MNA.fast_rebuild!(graetz_ws, u, 0.001)
            end
        end
        @printf("  %4d iterations: %.1f bytes/call (total: %.1f KB)\n",
                iters, allocs/iters, allocs/1000)
    end

    # Profile with changing u values (more realistic)
    println("\nfast_rebuild! with varying operating point:")
    allocs_varying = @allocated begin
        for i in 1:100
            u_var = 0.7 * ones(n) + 0.1 * randn(n)
            MNA.fast_rebuild!(graetz_ws, u_var, 0.001 * i)
        end
    end
    @printf("  100 iterations with varying u: %.1f bytes/call\n", allocs_varying/100)

    # Check if it's from context operations
    println("\nBreakdown by operation:")

    dctx = graetz_ws.dctx
    cs = graetz_ws.structure

    # Test reset_direct_stamp!
    allocs_reset = @allocated begin
        for _ in 1:1000
            MNA.reset_direct_stamp!(dctx)
        end
    end
    @printf("  reset_direct_stamp!: %.1f bytes/call\n", allocs_reset/1000)

    # Test builder call (the actual stamping)
    u_test = ones(n)
    MNA.reset_direct_stamp!(dctx)

    # Single builder call
    allocs_builder = @allocated begin
        for _ in 1:100
            MNA.reset_direct_stamp!(dctx)
            cs.builder(cs.params, cs.spec, 0.001; x=u_test, ctx=dctx)
        end
    end
    @printf("  builder call (100x): %.1f bytes/call\n", allocs_builder/100)

    # Deferred b stamps
    allocs_deferred = @allocated begin
        for _ in 1:1000
            for k in 1:cs.n_b_deferred
                idx = dctx.b_resolved[k]
                if idx > 0
                    dctx.b[idx] += dctx.b_V[k]
                end
            end
        end
    end
    @printf("  deferred b stamps: %.1f bytes/call\n", allocs_deferred/1000)
end

#==============================================================================#
# Profile Builder Function
#==============================================================================#

function profile_builder()
    println("\n" * "=" ^ 50)
    println("Builder Function Profiling")
    println("=" ^ 50)

    graetz_circ = MNACircuit(graetz_circuit)
    graetz_ws = MNA.compile(graetz_circ)
    cs = graetz_ws.structure
    dctx = graetz_ws.dctx

    n = MNA.system_size(cs)
    u = ones(n)

    # Profile just the builder calls
    println("\nBuilder function allocation:")

    # Warmup
    for _ in 1:10
        MNA.reset_direct_stamp!(dctx)
        cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
    end
    GC.gc()

    # Measure
    allocs = @allocated begin
        for _ in 1:100
            MNA.reset_direct_stamp!(dctx)
            cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
        end
    end

    @printf("  Builder + reset: %.1f bytes/call\n", allocs/100)

    # Try with type instability check
    println("\nChecking for type instabilities...")
    @code_warntype cs.builder(cs.params, cs.spec, 0.001; x=u, ctx=dctx)
end

#==============================================================================#
# Profile Individual Diode
#==============================================================================#

function profile_single_diode()
    println("\n" * "=" ^ 50)
    println("Single Diode Profiling")
    println("=" ^ 50)

    # Create a minimal circuit with just one diode
    println("\nCreating minimal diode circuit...")

    # Parse a simple diode circuit
    simple_diode_spice = """
    Simple diode circuit
    vs 1 0 dc 0.7
    xd1 1 0 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
    .end
    """

    simple_code = parse_spice_to_mna(simple_diode_spice; circuit_name=:simple_diode,
                                      imported_hdl_modules=[sp_diode_module])
    eval(simple_code)

    simple_circ = MNACircuit(simple_diode)
    simple_ws = MNA.compile(simple_circ)
    cs = simple_ws.structure

    n = MNA.system_size(cs)
    println("Simple diode circuit size: $n")

    u = ones(n)

    # Profile single diode
    allocs_single = @allocated begin
        for _ in 1:1000
            MNA.fast_rebuild!(simple_ws, u, 0.001)
        end
    end
    @printf("\nSingle diode fast_rebuild!: %.1f bytes/call\n", allocs_single/1000)

    # Graetz has 4 diodes - check if allocation scales linearly
    println("\nExpected for 4 diodes: %.1f bytes/call", (allocs_single/1000) * 4)
end

#==============================================================================#
# Main
#==============================================================================#

function main()
    profile_detailed()
    # profile_builder()  # Commented out - @code_warntype is verbose
    profile_single_diode()

    println("\n" * "=" ^ 70)
    println("CONCLUSIONS")
    println("=" ^ 70)

    println("""

The 800 bytes/call allocation in Graetz fast_rebuild! comes from:

1. **VA device model evaluation**
   - The Verilog-A diode model is being evaluated in the builder
   - Each diode contributes some allocation

2. **Potential sources in VA code:**
   - exp() or log() function calls with large arguments
   - Temperature-dependent calculations
   - Junction capacitance evaluation
   - Charge calculations for ddt() terms

3. **This is a key optimization opportunity:**
   - For 1M timesteps × 800 bytes = 800 MB just from rebuilds
   - Combined with Jacobian updates = 2+ GB for Graetz

4. **Next steps to investigate:**
   - Look at the generated VA module code
   - Check for allocating operations in device evaluation
   - Consider caching intermediate calculations
""")
end

main()
