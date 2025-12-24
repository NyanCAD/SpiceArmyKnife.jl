#==============================================================================#
# Phase 6 Test: BSIMCMG Inverter with MNA Backend
#
# This is a simplified version of bsimcmg_spectre.jl that uses the MNA backend
# instead of DAECompiler.
#==============================================================================#

module bsimcmg_mna

using CedarSim
using CedarSim.VerilogAParser
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: VoltageSource, Resistor, MNACircuit
using SpectreNetlistParser
using CedarSim.SpectreEnvironment
using Test
using SciMLBase

# Step 1: Load the BSIMCMG Verilog-A model
# This uses make_mna_module internally to generate stamp! methods
println("Step 1: Loading BSIMCMG model...")
# Use the local copy from VerilogAParser
const bsimcmg_path = joinpath(dirname(pathof(VerilogAParser)), "..", "cmc_models", "bsimcmg107", "bsimcmg.va")
@time const bsimcmg = CedarSim.ModelLoader.load_VA_model(bsimcmg_path)
println("  BSIMCMG module type: ", typeof(bsimcmg))

# Test that we can instantiate a BSIMCMG device
println("\nStep 2: Testing device instantiation...")
try
    dev = bsimcmg()
    println("  Created device: ", typeof(dev))
catch e
    println("  ERROR creating device: ", e)
    rethrow(e)
end

# Step 3: Try to stamp a simple circuit with BSIMCMG
println("\nStep 3: Testing stamp! method...")
try
    ctx = MNAContext()
    d = get_node!(ctx, :d)
    g = get_node!(ctx, :g)
    s = get_node!(ctx, :s)
    b = get_node!(ctx, :b)

    # Stamp voltage sources
    stamp!(VoltageSource(1.0; name=:Vdd), ctx, d, 0)
    stamp!(VoltageSource(0.5; name=:Vg), ctx, g, 0)
    stamp!(VoltageSource(0.0; name=:Vs), ctx, s, 0)
    stamp!(VoltageSource(0.0; name=:Vb), ctx, b, 0)

    # Try to stamp BSIMCMG device
    dev = bsimcmg()
    x = zeros(10)  # Dummy solution vector
    MNA.stamp!(dev, ctx, d, g, s, b; t=0.0, mode=:dcop, x=x)

    sys = assemble!(ctx)
    println("  SUCCESS: stamp! method works")
    println("  System size: ", MNA.system_size(sys))
catch e
    println("  ERROR in stamp!: ", e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

end # module bsimcmg_mna
