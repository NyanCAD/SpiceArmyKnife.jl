"""
Test PDK and VA device precompilation with MNA.

This tests:
- `load_mna_modules()` for SPICE PDK files with subcircuits
- `load_mna_va_module()` for Verilog-A device files
- SPICE circuits that reference VA devices via `imported_hdl_modules`
"""

using Test
using CedarSim
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, VoltageSource, MNACircuit
using CedarSim: dc!
using CedarSim: ParamLens

const testpdk_path = joinpath(@__DIR__, "testpdk.spice")
const test_va_path = joinpath(@__DIR__, "test_resistor.va")
const circuit_with_va_path = joinpath(@__DIR__, "circuit_with_va.spice")

# Load PDK modules directly into this module (like a real PDK package would do)
# This is the preferred API for precompilation - evals internally and returns modules
const corners = CedarSim.load_mna_modules(@__MODULE__, testpdk_path)

# Load VA device module (like a device package would do)
const va_device_mod = CedarSim.load_mna_va_module(@__MODULE__, test_va_path)

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
        function build_inverter_test(params, spec::MNASpec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end
            lens = ParamLens(params)

            # Nodes
            vdd = get_node!(ctx, :vdd)
            vss = get_node!(ctx, :vss)
            inp = get_node!(ctx, :inp)
            out = get_node!(ctx, :out)

            # Use the PDK inverter (lowercase parameter names from SPICE)
            typical.inv_x1_mna_builder(lens, spec, 0.0, ctx, inp, out, vdd, vss, (;), Float64[];
                                       wn=360e-9, wp=720e-9, l=180e-9)

            # Power supplies (VoltageSource.stamp! takes ctx, p, n - no spec)
            stamp!(VoltageSource(1.8), ctx, vdd, 0)
            stamp!(VoltageSource(0.0), ctx, vss, 0)

            # Input voltage
            stamp!(VoltageSource(0.9), ctx, inp, 0)

            return ctx
        end

        # Solve DC
        circuit = MNACircuit(build_inverter_test)
        sol = dc!(circuit)

        # Build ctx to get node indices
        ctx = build_inverter_test((;), MNASpec())
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
            function build_nmos_test(params, spec::MNASpec, t::Real=0.0; x=Float64[], ctx=nothing)
                if ctx === nothing
                    ctx = MNAContext()
                else
                    CedarSim.MNA.reset_for_restamping!(ctx)
                end
                lens = ParamLens(params)

                vdd = get_node!(ctx, :vdd)
                gnd = get_node!(ctx, :gnd)

                # NMOS between vdd and gnd (lowercase parameter names from SPICE)
                corner.nmos_1v8_mna_builder(lens, spec, 0.0, ctx, vdd, gnd, gnd, gnd, (;), Float64[];
                                           w=1e-6, l=180e-9)

                # 1V supply (VoltageSource.stamp! takes ctx, p, n - no spec)
                stamp!(VoltageSource(1.0), ctx, vdd, 0)

                return ctx
            end

            circuit = MNACircuit(build_nmos_test)
            sol = dc!(circuit)

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

@testset "VA Device Precompilation" begin

    @testset "load_mna_va_module into module" begin
        # Check that the module was created
        @test va_device_mod isa Module
        @test isdefined(@__MODULE__, :test_resistor_module)

        # Check that device type is exported
        @test isdefined(test_resistor_module, :test_resistor)
    end

    @testset "load_mna_va_module expression form" begin
        # Test expression-returning form
        expr = CedarSim.load_mna_va_module(test_va_path)
        @test expr.head == :toplevel
        @test length(expr.args) == 2  # module def + using statement
    end

    @testset "Use VA device in circuit" begin
        # Build a simple circuit using the VA resistor
        function build_va_resistor_test(params, spec::MNASpec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end

            # Nodes
            vin = get_node!(ctx, :vin)
            out = get_node!(ctx, :out)

            # Use the VA resistor device
            dev = test_resistor_module.test_resistor(R=500.0)
            stamp!(dev, ctx, vin, out; _mna_t_=0.0, _mna_mode_=:dcop, _mna_x_=Float64[])

            # Voltage source
            stamp!(VoltageSource(1.0), ctx, vin, 0)

            # Load resistor to ground (using built-in Resistor)
            stamp!(CedarSim.MNA.Resistor(500.0), ctx, out, 0)

            return ctx
        end

        # Solve DC
        circuit = MNACircuit(build_va_resistor_test)
        sol = dc!(circuit)

        # Build ctx to get node indices
        ctx = build_va_resistor_test((;), MNASpec())
        out_idx = ctx.node_to_idx[:out]
        @test sol.x[out_idx] ≈ 0.5 atol=1e-6
    end

    @testset "VA device with different parameters" begin
        # Test that parameter changes work
        function get_output_voltage(R_value)
            function va_divider_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
                if ctx === nothing
                    ctx = MNAContext()
                else
                    CedarSim.MNA.reset_for_restamping!(ctx)
                end

                vin = get_node!(ctx, :vin)
                out = get_node!(ctx, :out)

                # VA resistor with custom R
                dev = test_resistor_module.test_resistor(R=R_value)
                stamp!(dev, ctx, vin, out; _mna_t_=0.0, _mna_mode_=:dcop, _mna_x_=Float64[])

                # Voltage source
                stamp!(VoltageSource(1.0), ctx, vin, 0)

                # Fixed load
                stamp!(CedarSim.MNA.Resistor(1000.0), ctx, out, 0)

                return ctx
            end

            circuit = MNACircuit(va_divider_circuit)
            sol = dc!(circuit)

            # Build ctx to get node index
            ctx = va_divider_circuit((;), MNASpec())
            return sol.x[ctx.node_to_idx[:out]]
        end

        # Lower R in divider means lower output voltage
        v_100 = get_output_voltage(100.0)   # 1000/(100+1000) = 0.909
        v_1000 = get_output_voltage(1000.0)  # 1000/(1000+1000) = 0.5
        v_10000 = get_output_voltage(10000.0) # 1000/(10000+1000) = 0.091

        @test v_100 > v_1000 > v_10000
        @test v_1000 ≈ 0.5 atol=1e-6
    end

end

@testset "SPICE Circuit with VA Device" begin
    # This demonstrates the typical use case where:
    # 1. A VA device comes from a precompiled package (like BSIM4.jl)
    # 2. A SPICE netlist references that device by name
    # 3. The circuit is built using make_mna_circuit with imported_hdl_modules
    #
    # In a real PDK, the SPICE file would use:
    #   .hdl "jlpkg://BSIM4/bsim4.va"
    # to reference the package. Here we demonstrate the equivalent pattern
    # by passing the module via imported_hdl_modules.

    @testset "Build circuit from SPICE with VA device" begin
        # Parse the SPICE circuit
        using SpectreNetlistParser
        ast = SpectreNetlistParser.parsefile(circuit_with_va_path; implicit_title=true)
        @test !ast.ps.errored

        # Build the MNA circuit, passing the VA device module
        # This is equivalent to what happens when SPICE uses .hdl "jlpkg://..."
        builder_code = CedarSim.make_mna_circuit(ast;
            circuit_name=:va_circuit,
            imported_hdl_modules=[test_resistor_module])

        # Eval the builder
        builder = @eval $builder_code

        # Build and solve
        circuit = MNACircuit(builder)
        sol = dc!(circuit)

        # Build ctx to get node indices
        ctx = builder((;), MNASpec())
        out_idx = ctx.node_to_idx[:out]
        @test sol.x[out_idx] ≈ 0.5 atol=1e-6
    end

    @testset "Compare SPICE-defined vs Julia-defined circuit" begin
        # Build the same circuit in pure Julia for comparison
        function build_julia_circuit(params, spec::MNASpec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end

            vin = get_node!(ctx, :vin)
            out = get_node!(ctx, :out)

            # VA resistor
            dev = test_resistor_module.test_resistor(R=500.0)
            stamp!(dev, ctx, vin, out; _mna_t_=0.0, _mna_mode_=:dcop, _mna_x_=Float64[])

            # Regular resistor
            stamp!(CedarSim.MNA.Resistor(500.0), ctx, out, 0)

            # Voltage source
            stamp!(VoltageSource(1.0), ctx, vin, 0)

            return ctx
        end

        # Solve Julia-defined circuit
        circuit_julia = MNACircuit(build_julia_circuit)
        sol_julia = dc!(circuit_julia)
        ctx_julia = build_julia_circuit((;), MNASpec())

        # Solve SPICE-defined circuit
        using SpectreNetlistParser
        ast = SpectreNetlistParser.parsefile(circuit_with_va_path; implicit_title=true)
        builder_code = CedarSim.make_mna_circuit(ast;
            circuit_name=:va_circuit2,
            imported_hdl_modules=[test_resistor_module])
        builder = @eval $builder_code
        circuit_spice = MNACircuit(builder)
        sol_spice = dc!(circuit_spice)
        ctx_spice = builder((;), MNASpec())

        # Both should give same output voltage
        out_julia = sol_julia.x[ctx_julia.node_to_idx[:out]]
        out_spice = sol_spice.x[ctx_spice.node_to_idx[:out]]

        @test out_julia ≈ out_spice atol=1e-10
        @test out_julia ≈ 0.5 atol=1e-6
    end

end
