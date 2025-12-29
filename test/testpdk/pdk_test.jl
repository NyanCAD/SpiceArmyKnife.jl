"""
Test PDK precompilation with MNA builder functions.

This tests the `load_mna_modules()` function which generates precompilable
MNA builder functions from PDK SPICE files.
"""

using Test
using CedarSim
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc, VoltageSource
using CedarSim: ParamLens

const testpdk_path = joinpath(@__DIR__, "testpdk.spice")

# Load PDK modules directly into this module (like a real PDK package would do)
# This is the preferred API for precompilation - evals internally and returns modules
const corners = CedarSim.load_mna_modules(@__MODULE__, testpdk_path)

@testset "PDK MNA Module Generation" begin

    @testset "load_mna_modules into module" begin
        # Check that modules were created
        @test haskey(corners, :typical)
        @test haskey(corners, :fast)
        @test haskey(corners, :slow)

        # Check that submodules are defined in this module
        @test isdefined(@__MODULE__, :typical)
        @test isdefined(@__MODULE__, :fast)
        @test isdefined(@__MODULE__, :slow)
    end

    @testset "load_mna_modules expression form" begin
        # Test backward compatible expression-returning form
        expr = CedarSim.load_mna_modules(testpdk_path)
        @test expr.head == :toplevel
        @test length(expr.args) == 3
    end

    @testset "load_mna_modules with names filter" begin
        # Load only typical section (expression form)
        expr = CedarSim.load_mna_modules(testpdk_path; names=["typical"])
        @test length(expr.args) == 1
    end

    @testset "load_mna_pdk single section expression" begin
        expr = CedarSim.load_mna_pdk(testpdk_path; section="typical")
        @test expr.head == :module  # Direct module expression
    end

    @testset "PDK module structure" begin
        # Check that builder functions are exported
        @test isdefined(typical, :nmos_1v8_mna_builder)
        @test isdefined(typical, :pmos_1v8_mna_builder)
        @test isdefined(typical, :inv_x1_mna_builder)
        @test isdefined(typical, :nand2_x1_mna_builder)
    end

    @testset "Use PDK builders in circuit" begin
        # Build a circuit using the PDK
        function build_inverter_test(params, spec::MNASpec)
            ctx = MNAContext()
            lens = ParamLens(params)

            # Nodes
            vdd = get_node!(ctx, :vdd)
            vss = get_node!(ctx, :vss)
            inp = get_node!(ctx, :inp)
            out = get_node!(ctx, :out)

            # Use the PDK inverter (lowercase parameter names from SPICE)
            typical.inv_x1_mna_builder(lens, spec, ctx, inp, out, vdd, vss, (;);
                                       wn=360e-9, wp=720e-9, l=180e-9)

            # Power supplies (VoltageSource.stamp! takes ctx, p, n - no spec)
            stamp!(VoltageSource(1.8), ctx, vdd, 0)
            stamp!(VoltageSource(0.0), ctx, vss, 0)

            # Input voltage
            stamp!(VoltageSource(0.9), ctx, inp, 0)

            return ctx
        end

        # Solve DC
        ctx = build_inverter_test((;), MNASpec())
        sys = assemble!(ctx)
        sol = solve_dc(sys)

        # Check that we get reasonable voltages
        vdd_idx = ctx.node_to_idx[:vdd]
        vss_idx = ctx.node_to_idx[:vss]
        out_idx = ctx.node_to_idx[:out]

        @test sol.x[vdd_idx] ≈ 1.8
        @test sol.x[vss_idx] ≈ 0.0
        # Output should be somewhere between vdd and vss
        @test 0.0 < sol.x[out_idx] < 1.8
    end

    @testset "Compare corners" begin
        function get_nmos_current(corner)
            # Build simple circuit with just nmos
            function build_nmos_test(params, spec::MNASpec)
                ctx = MNAContext()
                lens = ParamLens(params)

                vdd = get_node!(ctx, :vdd)
                gnd = get_node!(ctx, :gnd)

                # NMOS between vdd and gnd (lowercase parameter names from SPICE)
                corner.nmos_1v8_mna_builder(lens, spec, ctx, vdd, gnd, gnd, gnd, (;);
                                           w=1e-6, l=180e-9)

                # 1V supply (VoltageSource.stamp! takes ctx, p, n - no spec)
                stamp!(VoltageSource(1.0), ctx, vdd, 0)

                return ctx
            end

            ctx = build_nmos_test((;), MNASpec())
            sys = assemble!(ctx)
            sol = solve_dc(sys)

            # Get current through voltage source (last element typically)
            # Current = 1V / R, so higher current means lower resistance
            return abs(sol.x[end])  # Return absolute current
        end

        i_typical = get_nmos_current(typical)
        i_fast = get_nmos_current(fast)
        i_slow = get_nmos_current(slow)

        # Fast should have highest current (lowest R)
        # Slow should have lowest current (highest R)
        @test i_fast > i_typical
        @test i_typical > i_slow
    end

end
