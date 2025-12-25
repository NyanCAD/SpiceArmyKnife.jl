#==============================================================================#
# Phase 6 Test: BSIMCMG with MNA Backend
#
# Tests that BSIMCMG Verilog-A model loads and stamp! method works correctly.
# This is a key test for Phase 6 (full VA & DAE support).
#==============================================================================#

module bsimcmg_mna

using CedarSim
using CedarSim.VerilogAParser
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!
using CedarSim.MNA: VoltageSource, Resistor
using Test

@testset "BSIMCMG MNA Backend" begin
    # Step 1: Load the BSIMCMG Verilog-A model
    @testset "Model Loading" begin
        bsimcmg_path = joinpath(dirname(pathof(VerilogAParser)), "..", "cmc_models", "bsimcmg107", "bsimcmg.va")
        @test isfile(bsimcmg_path)

        bsimcmg = CedarSim.ModelLoader.load_VA_model(bsimcmg_path)
        @test bsimcmg isa DataType

        # Test device instantiation
        dev = bsimcmg()
        @test dev !== nothing
    end

    # Step 2: Test stamp! method
    @testset "Stamp Method" begin
        bsimcmg_path = joinpath(dirname(pathof(VerilogAParser)), "..", "cmc_models", "bsimcmg107", "bsimcmg.va")
        bsimcmg = CedarSim.ModelLoader.load_VA_model(bsimcmg_path)

        ctx = MNAContext()
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)
        s = get_node!(ctx, :s)
        b = get_node!(ctx, :b)

        # Stamp voltage sources for biasing
        stamp!(VoltageSource(1.0; name=:Vdd), ctx, d, 0)
        stamp!(VoltageSource(0.5; name=:Vg), ctx, g, 0)
        stamp!(VoltageSource(0.0; name=:Vs), ctx, s, 0)
        stamp!(VoltageSource(0.0; name=:Vb), ctx, b, 0)

        # Stamp BSIMCMG device
        dev = bsimcmg()
        x = zeros(10)  # Dummy solution vector
        MNA.stamp!(dev, ctx, d, g, s, b; t=0.0, mode=:dcop, x=x)

        # Assemble and check system was created
        sys = assemble!(ctx)
        @test MNA.system_size(sys) > 0
        @test ctx.n_nodes >= 4  # d, g, s, b + internal nodes (si, di)
        @test ctx.n_currents >= 4  # At least 4 voltage sources
    end

    # Step 3: Test with PMOS (DEVTYPE=1)
    @testset "PMOS Device" begin
        bsimcmg_path = joinpath(dirname(pathof(VerilogAParser)), "..", "cmc_models", "bsimcmg107", "bsimcmg.va")
        bsimcmg = CedarSim.ModelLoader.load_VA_model(bsimcmg_path)

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

        # Create PMOS device (DEVTYPE=1)
        pmos = bsimcmg(; DEVTYPE=1)
        x = zeros(10)
        MNA.stamp!(pmos, ctx, d, g, s, b; t=0.0, mode=:dcop, x=x)

        sys = assemble!(ctx)
        @test MNA.system_size(sys) > 0
    end
end

end # module bsimcmg_mna
