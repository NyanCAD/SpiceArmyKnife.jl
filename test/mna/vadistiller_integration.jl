#==============================================================================#
# MNA Phase 6+: VADistiller Integration Tests (Heavy Models)
#
# These tests load Verilog-A models from files. Due to memory constraints in CI,
# only a minimal representative set is tested here.
#
# Run with: julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'
# Or directly: julia --project=. test/mna/vadistiller_integration.jl
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: voltage, current
using CedarSim.MNA: VoltageSource, Resistor
using VerilogAParser

const deftol = 1e-6
isapprox_deftol(a, b) = isapprox(a, b; atol=deftol, rtol=deftol)

# Path to vadistiller models
const vadistiller_path = joinpath(@__DIR__, "..", "vadistiller", "models")

# Memory tracking helper
function mem_mb()
    GC.gc(true)  # Full GC
    return round(Sys.total_memory() - Sys.free_memory(); digits=0) / 1024 / 1024
end

@testset "VADistiller Integration Tests" begin

    @info "Starting integration tests" initial_mem_mb=mem_mb()

    #==========================================================================#
    # Minimal Model Tests - Representative subset to validate file-based loading
    # Tests: resistor (2-terminal), diode (2-terminal nonlinear), mos1 (4-terminal)
    #==========================================================================#

    @testset "File-based VA Model Loading" begin

        # Test 1: Resistor (simplest 2-terminal)
        @testset "sp_resistor from file" begin
            @info "Loading resistor.va" mem_mb=mem_mb()
            let
                va_code = read(joinpath(vadistiller_path, "resistor.va"), String)
                va = VerilogAParser.parse(IOBuffer(va_code))
                @test !va.ps.errored
                Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
            end
            GC.gc(true)

            function resistor_divider(params, spec, t::Real=0.0)
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                mid = get_node!(ctx, :mid)
                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(sp_resistor(resistance=1000.0), ctx, vcc, mid; _mna_spec_=spec, _mna_x_=Float64[])
                stamp!(sp_resistor(resistance=1000.0), ctx, mid, 0; _mna_spec_=spec, _mna_x_=Float64[])
                return ctx
            end

            ctx = resistor_divider((;), MNASpec())
            sys = assemble!(ctx)
            sol = solve_dc(sys)

            @test isapprox_deftol(voltage(sol, :vcc), 5.0)
            @test isapprox_deftol(voltage(sol, :mid), 2.5)
            @info "Resistor test passed" mem_mb=mem_mb()
        end

        GC.gc(true)

        # Test 2: Diode (2-terminal nonlinear)
        @testset "sp_diode from file" begin
            @info "Loading diode.va" mem_mb=mem_mb()
            let
                va_code = read(joinpath(vadistiller_path, "diode.va"), String)
                va = VerilogAParser.parse(IOBuffer(va_code))
                @test !va.ps.errored
                Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
            end
            GC.gc(true)

            function diode_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                diode_a = get_node!(ctx, :diode_a)
                stamp!(VoltageSource(1.0; name=:V1), ctx, vcc, 0)
                stamp!(Resistor(1000.0; name=:R1), ctx, vcc, diode_a)
                stamp!(sp_diode(), ctx, diode_a, 0; _mna_spec_=spec, _mna_x_=x)
                return ctx
            end

            sol = solve_dc(diode_circuit, (;), MNASpec())
            v_diode = sol.x[2]
            @test v_diode > 0.5 && v_diode < 0.8  # Reasonable diode forward voltage
            @info "Diode test passed" mem_mb=mem_mb()
        end

        GC.gc(true)

        # Test 3: MOS1 (4-terminal)
        @testset "sp_mos1 from file" begin
            @info "Loading mos1.va" mem_mb=mem_mb()
            let
                va_code = read(joinpath(vadistiller_path, "mos1.va"), String)
                va = VerilogAParser.parse(IOBuffer(va_code))
                @test !va.ps.errored
                Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
            end
            GC.gc(true)

            function mos1_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                drain = get_node!(ctx, :drain)
                gate = get_node!(ctx, :gate)
                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                stamp!(sp_mos1(; l=1e-6, w=10e-6, vto=0.7, kp=1e-4), ctx, drain, gate, 0, 0; _mna_spec_=spec, _mna_x_=x)
                return ctx
            end

            sol = solve_dc(mos1_circuit, (;), MNASpec())
            @test voltage(sol, :drain) > 0.0 && voltage(sol, :drain) < 5.0
            @info "MOS1 test passed" mem_mb=mem_mb()
        end

        GC.gc(true)

    end

    @info "Integration tests completed" final_mem_mb=mem_mb()

end
