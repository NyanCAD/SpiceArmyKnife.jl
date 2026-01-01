#==============================================================================#
# MNA Phase 6+: VADistiller Integration Tests (Heavy Models)
#
# These tests load large Verilog-A models from files and are separated from
# the core vadistiller tests to reduce memory pressure during CI.
#
# Run with: julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'
# Or directly: julia --project=. test/mna/vadistiller_integration.jl
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: voltage, current, make_ode_problem
using CedarSim.MNA: VoltageSource, Resistor, Capacitor, CurrentSource
using CedarSim.MNA: MNACircuit, SinVoltageSource, MNASolutionAccessor
using ForwardDiff: Dual, value, partials
using OrdinaryDiffEq
using OrdinaryDiffEq: Rodas5P, QNDF
using VerilogAParser
using Sundials: IDA
using CedarSim: tran!, dc!

const deftol = 1e-6
isapprox_deftol(a, b) = isapprox(a, b; atol=deftol, rtol=deftol)

# Path to vadistiller models
const vadistiller_path = joinpath(@__DIR__, "..", "vadistiller", "models")

# Memory tracking helper for diagnosing OOM issues
function mem_mb()
    return round(Sys.total_memory() - Sys.free_memory(); digits=0) / 1024 / 1024
end

function with_memory_tracking(f, name::String)
    GC.gc()
    mem_before = mem_mb()
    t0 = time()
    result = f()
    elapsed = time() - t0
    GC.gc()
    mem_after = mem_mb()
    @info "$name" elapsed_s=round(elapsed; digits=2) mem_before_mb=round(mem_before; digits=0) mem_after_mb=round(mem_after; digits=0) mem_delta_mb=round(mem_after - mem_before; digits=0)
    return result
end

# Load a VA model file with memory tracking
function load_va_model(filename::String)
    filepath = joinpath(vadistiller_path, filename)
    with_memory_tracking("Loading $filename") do
        va_code = read(filepath, String)
        va = VerilogAParser.parse(IOBuffer(va_code))
        Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
        va = nothing  # Help GC
    end
    GC.gc()
end

@testset "VADistiller Integration Tests" begin

    #==========================================================================#
    # Tier 6: Full VADistiller Model Tests (from file)
    #
    # Each model is loaded in a let block to allow GC of AST after testing.
    # GC.gc() is called between groups to reduce peak memory usage.
    #==========================================================================#

    @testset "Tier 6: Full VADistiller Models (from file)" begin

        @info "Starting Tier 6 tests" initial_mem_mb=round(mem_mb(); digits=0)

        # Group 1: Simple passives (resistor, capacitor, inductor)
        @testset "VADistiller passives" begin
            load_va_model("resistor.va")

            @testset "sp_resistor divider" begin
                function sp_resistor_divider(params, spec, t::Real=0.0)
                    ctx = MNAContext()
                    vcc = get_node!(ctx, :vcc)
                    mid = get_node!(ctx, :mid)

                    stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                    stamp!(sp_resistor(resistance=1000.0), ctx, vcc, mid; _mna_spec_=spec, _mna_x_=Float64[])
                    stamp!(sp_resistor(resistance=1000.0), ctx, mid, 0; _mna_spec_=spec, _mna_x_=Float64[])

                    return ctx
                end

                ctx = sp_resistor_divider((;), MNASpec())
                sys = assemble!(ctx)
                sol = solve_dc(sys)

                @test isapprox_deftol(voltage(sol, :vcc), 5.0)
                @test isapprox_deftol(voltage(sol, :mid), 2.5)
            end

            load_va_model("capacitor.va")

            @testset "sp_capacitor DC" begin
                function sp_capacitor_circuit(params, spec, t::Real=0.0)
                    ctx = MNAContext()
                    vcc = get_node!(ctx, :vcc)
                    mid = get_node!(ctx, :mid)

                    stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                    stamp!(Resistor(1000.0; name=:R1), ctx, vcc, mid)
                    stamp!(sp_capacitor(capacitance=1e-6), ctx, mid, 0; _mna_spec_=spec, _mna_x_=Float64[])

                    return ctx
                end

                ctx = sp_capacitor_circuit((;), MNASpec())
                sys = assemble!(ctx)
                sol = solve_dc(sys)

                @test isapprox_deftol(voltage(sol, :vcc), 5.0)
                @test isapprox_deftol(voltage(sol, :mid), 5.0)  # Open circuit at DC
            end

            load_va_model("inductor.va")

            @testset "sp_inductor DC" begin
                function sp_inductor_circuit(params, spec, t::Real=0.0)
                    ctx = MNAContext()
                    vcc = get_node!(ctx, :vcc)
                    mid = get_node!(ctx, :mid)

                    stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                    stamp!(Resistor(1000.0; name=:R1), ctx, vcc, mid)
                    stamp!(sp_inductor(inductance=1e-3), ctx, mid, 0; _mna_spec_=spec, _mna_x_=Float64[])

                    return ctx
                end

                ctx = sp_inductor_circuit((;), MNASpec())
                sys = assemble!(ctx)
                sol = solve_dc(sys)
                @test isapprox(voltage(sol, :mid), 0.0; atol=0.01)
            end
        end

        GC.gc()

        # Group 2: Diode
        @testset "VADistiller sp_diode" begin
            load_va_model("diode.va")

            @testset "Basic diode circuit" begin
                function sp_diode_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vcc = get_node!(ctx, :vcc)
                    diode_a = get_node!(ctx, :diode_a)
                    stamp!(VoltageSource(1.0; name=:V1), ctx, vcc, 0)
                    stamp!(Resistor(1000.0; name=:R1), ctx, vcc, diode_a)
                    stamp!(sp_diode(), ctx, diode_a, 0; _mna_spec_=spec, _mna_x_=x)
                    return ctx
                end

                sol = solve_dc(sp_diode_circuit, (;), MNASpec())
                v_diode = sol.x[2]
                @test v_diode > 0.6 && v_diode < 0.7
            end

            @testset "Diode with series resistance" begin
                function sp_diode_rs_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vcc = get_node!(ctx, :vcc)
                    diode_a = get_node!(ctx, :diode_a)
                    stamp!(VoltageSource(1.0; name=:V1), ctx, vcc, 0)
                    stamp!(Resistor(1000.0; name=:R1), ctx, vcc, diode_a)
                    stamp!(sp_diode(; rs=10.0), ctx, diode_a, 0; _mna_spec_=spec, _mna_x_=x)
                    return ctx
                end

                sol = solve_dc(sp_diode_rs_circuit, (;), MNASpec())
                sys = assemble!(sp_diode_rs_circuit((;), MNASpec(); x=sol.x))
                has_internal_node = :sp_diode_a_int in sys.node_names

                v_diode_a = sol.x[2]
                v_a_int = sol.x[3]

                junction_ok = v_a_int > 0.6 && v_a_int < 0.7
                drop_ok = v_diode_a > v_a_int
                drop_reasonable = (v_diode_a - v_a_int) < 0.01

                @test has_internal_node && junction_ok && drop_ok && drop_reasonable
            end
        end

        GC.gc()

        # Group 3: BJT
        @testset "VADistiller sp_bjt" begin
            load_va_model("bjt.va")

            @testset "Basic BJT circuit" begin
                function sp_bjt_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vcc = get_node!(ctx, :vcc)
                    vb = get_node!(ctx, :vb)
                    collector = get_node!(ctx, :collector)
                    base = get_node!(ctx, :base)

                    stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                    stamp!(VoltageSource(0.7; name=:V2), ctx, vb, 0)
                    stamp!(Resistor(10000.0; name=:Rb), ctx, vb, base)
                    stamp!(Resistor(1000.0; name=:Rc), ctx, vcc, collector)
                    stamp!(sp_bjt(; bf=100.0, is=1e-15), ctx, collector, base, 0, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_bjt_circuit, (;), MNASpec())
                @test true  # If we get here, it worked
            end
        end

        GC.gc()

        # Group 4: JFETs and MESFETs
        @testset "VADistiller 3-terminal FETs" begin
            load_va_model("jfet1.va")

            @testset "sp_jfet1" begin
                function sp_jfet_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(10.0; name=:V1), ctx, vdd, 0)
                    stamp!(VoltageSource(0.0; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_jfet1(; vt0=-2.0, beta=1e-3), ctx, drain, gate, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_jfet_circuit, (;), MNASpec())
                @test true
            end

            load_va_model("mes1.va")

            @testset "sp_mes1" begin
                function sp_mes_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(5.0; name=:V1), ctx, vdd, 0)
                    stamp!(VoltageSource(0.0; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(500.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_mes1(; vt0=-1.0, beta=2.5e-3), ctx, drain, gate, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_mes_circuit, (;), MNASpec())
                @test true
            end

            load_va_model("jfet2.va")

            @testset "sp_jfet2" begin
                function sp_jfet2_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(10.0; name=:V1), ctx, vdd, 0)
                    stamp!(VoltageSource(0.0; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_jfet2(; vto=-2.0, beta=1e-3), ctx, drain, gate, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_jfet2_circuit, (;), MNASpec())
                @test true
            end
        end

        GC.gc()

        # Group 5: MOSFETs (mos1, mos2, mos3, mos6, mos9)
        @testset "VADistiller MOSFETs" begin
            load_va_model("mos1.va")

            @testset "sp_mos1" begin
                function sp_mos1_circuit(params, spec, t::Real=0.0; x=Float64[])
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

                sol = solve_dc(sp_mos1_circuit, (;), MNASpec())
                @test true
            end

            load_va_model("mos2.va")

            @testset "sp_mos2" begin
                function sp_mos2_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                    stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_mos2(; l=1e-6, w=10e-6, vto=0.7, kp=1e-4), ctx, drain, gate, 0, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_mos2_circuit, (;), MNASpec())
                @test true
            end

            load_va_model("mos3.va")

            @testset "sp_mos3" begin
                function sp_mos3_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                    stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_mos3(; l=1e-6, w=10e-6, vto=0.7, kp=1e-4), ctx, drain, gate, 0, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_mos3_circuit, (;), MNASpec())
                @test true
            end

            load_va_model("mos6.va")

            @testset "sp_mos6" begin
                function sp_mos6_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                    stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_mos6(; l=1e-6, w=10e-6, vto=0.7, u0=600.0, tox=10e-9), ctx, drain, gate, 0, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_mos6_circuit, (;), MNASpec())
                @test true
            end

            load_va_model("mos9.va")

            @testset "sp_mos9" begin
                function sp_mos9_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                    stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_mos9(; l=1e-6, w=10e-6, vto=0.7, kp=1e-4), ctx, drain, gate, 0, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_mos9_circuit, (;), MNASpec())
                @test true
            end
        end

        GC.gc()

        # Group 6: VDMOS (5-terminal with thermal nodes)
        @testset "VADistiller sp_vdmos" begin
            load_va_model("vdmos.va")

            @testset "sp_vdmos circuit" begin
                function sp_vdmos_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(10.0; name=:Vdd), ctx, vdd, 0)
                    stamp!(VoltageSource(5.0; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(100.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_vdmos(; vto=2.0, kp=0.5), ctx, drain, gate, 0, 0, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_vdmos_circuit, (;), MNASpec())
                @test 0.0 < voltage(sol, :drain) < 10.0
            end
        end

        GC.gc()

        # Group 7: BSIM3v3 (large model - 4K lines)
        @testset "VADistiller sp_bsim3v3" begin
            load_va_model("bsim3v3.va")

            @testset "sp_bsim3v3 circuit" begin
                function sp_bsim3v3_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(1.8; name=:Vdd), ctx, vdd, 0)
                    stamp!(VoltageSource(1.0; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_bsim3v3(; l=100e-9, w=1e-6), ctx, drain, gate, 0, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_bsim3v3_circuit, (;), MNASpec())
                @test true
            end
        end

        GC.gc()

        # Group 8: BSIM4v8 (largest model - 10K lines)
        @testset "VADistiller sp_bsim4v8" begin
            load_va_model("bsim4v8.va")

            @testset "sp_bsim4v8 string parameters" begin
                @test fieldtype(sp_bsim4v8, :version) == CedarSim.DefaultOr{String}

                dev = sp_bsim4v8()
                @test dev.version.val == "4.8.3" && dev.version.is_default

                dev2 = sp_bsim4v8(version="4.8.0")
                @test dev2.version.val == "4.8.0" && !dev2.version.is_default
            end

            @testset "sp_bsim4v8 circuit" begin
                function sp_bsim4v8_circuit(params, spec, t::Real=0.0; x=Float64[])
                    ctx = MNAContext()
                    vdd = get_node!(ctx, :vdd)
                    drain = get_node!(ctx, :drain)
                    gate = get_node!(ctx, :gate)

                    stamp!(VoltageSource(1.0; name=:Vdd), ctx, vdd, 0)
                    stamp!(VoltageSource(0.5; name=:Vg), ctx, gate, 0)
                    stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                    stamp!(sp_bsim4v8(; l=100e-9, w=1e-6), ctx, drain, gate, 0, 0; _mna_spec_=spec, _mna_x_=x)

                    return ctx
                end

                sol = solve_dc(sp_bsim4v8_circuit, (;), MNASpec())
                @test 0.9 < voltage(sol, :drain) < 1.0
            end
        end

        GC.gc()

        @info "Completed Tier 6 tests" final_mem_mb=round(mem_mb(); digits=0)

    end  # Tier 6

    #==========================================================================#
    # Tier 7: Transient Circuits with VADistiller Models
    #==========================================================================#

    @info "Starting Tier 7 transient tests" mem_mb=round(mem_mb(); digits=0)

    @testset "Tier 7: Transient Circuits" begin

        @testset "RC circuit with VADistiller components" begin
            function build_rc(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                v = get_node!(ctx, :v)
                out = get_node!(ctx, :out)

                stamp!(VoltageSource(params.V), ctx, v, 0)
                stamp!(sp_resistor(; r=params.R), ctx, v, out; _mna_spec_=spec, _mna_x_=x)
                stamp!(sp_capacitor(; c=params.C), ctx, out, 0; _mna_spec_=spec, _mna_x_=x)

                return ctx
            end

            circuit = MNACircuit(build_rc; V=5.0, R=1000.0, C=1e-6)
            tspan = (0.0, 5e-3)

            sol_rodas = tran!(circuit, tspan; solver=Rodas5P(), abstol=1e-10, reltol=1e-8)
            @test sol_rodas.retcode == SciMLBase.ReturnCode.Success

            sol_qndf = tran!(circuit, tspan; solver=QNDF(), abstol=1e-10, reltol=1e-8)
            @test sol_qndf.retcode == SciMLBase.ReturnCode.Success

            T = tspan[2]
            @test sol_rodas(T)[2] > 4.5
            @test isapprox(sol_rodas(T)[2], sol_qndf(T)[2]; rtol=0.01)
        end

        @testset "Diode rectifier transient" begin
            function build_rectifier(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vin = get_node!(ctx, :vin)
                vout = get_node!(ctx, :vout)

                stamp!(VoltageSource(params.V), ctx, vin, 0)
                stamp!(sp_diode(; is=1e-14), ctx, vin, vout; _mna_x_=x, _mna_spec_=spec)
                stamp!(sp_resistor(; r=params.R), ctx, vout, 0; _mna_spec_=spec, _mna_x_=x)

                return ctx
            end

            circuit = MNACircuit(build_rectifier; V=1.0, R=1000.0)
            tspan = (0.0, 1e-3)

            sol_rodas = tran!(circuit, tspan; solver=Rodas5P(), abstol=1e-8, reltol=1e-6)
            @test sol_rodas.retcode == SciMLBase.ReturnCode.Success

            sol_qndf = tran!(circuit, tspan; solver=QNDF(), abstol=1e-8, reltol=1e-6)
            @test sol_qndf.retcode == SciMLBase.ReturnCode.Success

            T = 0.5e-3
            @test sol_rodas(T)[2] > 0.3 && sol_rodas(T)[2] < 0.8
            @test isapprox(sol_rodas(T)[2], sol_qndf(T)[2]; rtol=0.01)
        end

        @testset "MOSFET CS amplifier transient" begin
            function build_cs_amp(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                vgate = get_node!(ctx, :vgate)
                vdrain = get_node!(ctx, :vdrain)

                stamp!(VoltageSource(params.Vdd; name=:Vdd), ctx, vdd, 0)
                stamp!(SinVoltageSource(params.Vbias, params.Vac, params.freq; name=:Vg),
                       ctx, vgate, 0; t=t, _sim_mode_=spec.mode)
                stamp!(sp_resistor(; r=params.Rd), ctx, vdd, vdrain; _mna_spec_=spec, _mna_x_=x)
                stamp!(sp_mos1(; vto=1.0, kp=1e-4),
                       ctx, vdrain, vgate, 0, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            circuit = MNACircuit(build_cs_amp;
                                 Vdd=5.0, Vbias=1.5, Vac=0.1, freq=1000.0, Rd=2000.0)
            tspan = (0.0, 2e-3)

            sol_rodas = tran!(circuit, tspan; solver=Rodas5P(), abstol=1e-8, reltol=1e-6)
            @test sol_rodas.retcode == SciMLBase.ReturnCode.Success

            sol_qndf = tran!(circuit, tspan; solver=QNDF(), abstol=1e-8, reltol=1e-6)
            @test sol_qndf.retcode == SciMLBase.ReturnCode.Success

            T = 1e-3
            vd_rodas = sol_rodas(T)[3]
            vd_qndf = sol_qndf(T)[3]

            @test vd_rodas > 0.0 && vd_rodas < 5.5
            @test vd_qndf > 0.0 && vd_qndf < 5.5
            @test isapprox(vd_rodas, vd_qndf; rtol=0.05)
        end

        @testset "BJT CE amplifier transient" begin
            function build_ce_amp(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                vbase = get_node!(ctx, :vbase)
                vcollector = get_node!(ctx, :vcollector)

                stamp!(VoltageSource(params.Vcc; name=:Vcc), ctx, vcc, 0)
                stamp!(SinVoltageSource(params.Vbias, params.Vac, params.freq; name=:Vb),
                       ctx, vbase, 0; t=t, _sim_mode_=spec.mode)
                stamp!(sp_resistor(; r=params.Rc), ctx, vcc, vcollector; _mna_spec_=spec, _mna_x_=x)
                stamp!(sp_bjt(; bf=100.0),
                       ctx, vcollector, vbase, 0, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            circuit = MNACircuit(build_ce_amp;
                                 Vcc=12.0, Vbias=0.6, Vac=0.01, freq=1000.0, Rc=1000.0)
            tspan = (0.0, 2e-3)

            sol = tran!(circuit, tspan; abstol=1e-8, reltol=1e-6, explicit_jacobian=false)
            @test sol.retcode == SciMLBase.ReturnCode.Success

            T = 1e-3
            vc_t0 = sol(T)[3]
            vc_pos = sol(T + T/4)[3]
            vc_neg = sol(T + 3T/4)[3]

            @test !isnan(vc_t0) && !isnan(vc_pos) && !isnan(vc_neg)
            @test vc_t0 > 0.0 && vc_t0 < 12.5
            @test vc_pos < vc_neg + 0.5
        end

    end  # Tier 7

    GC.gc()

    @info "Completed Tier 7, starting Tier 8" mem_mb=round(mem_mb(); digits=0)

    #==========================================================================#
    # Tier 8: DAE Transient with Voltage-Dependent Capacitors
    #==========================================================================#

    @testset "Tier 8: DAE Transient with Voltage-Dependent Capacitors" begin

        @testset "Diode junction capacitance (sp_diode with IDA)" begin
            function diode_rectifier_with_cap(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vin = get_node!(ctx, :vin)
                vout = get_node!(ctx, :vout)

                stamp!(SinVoltageSource(0.0, params.Vamp, params.freq; name=:Vin),
                       ctx, vin, 0; t=t, _sim_mode_=spec.mode)
                stamp!(sp_diode(; is=1e-14, cjo=10e-12, m=0.5, vj=0.7),
                       ctx, vin, vout; _mna_x_=x, _mna_spec_=spec)
                stamp!(Resistor(params.R), ctx, vout, 0)

                return ctx
            end

            circuit = MNACircuit(diode_rectifier_with_cap;
                                 Vamp=5.0, freq=1000.0, R=1000.0)
            tspan = (0.0, 3e-3)

            sol_ida = tran!(circuit, tspan; abstol=1e-8, reltol=1e-6)
            @test sol_ida.retcode == SciMLBase.ReturnCode.Success

            sys = assemble!(circuit)
            acc = MNASolutionAccessor(sol_ida, sys)

            T = 1e-3
            @test length(sol_ida.t) > 50
            @test length(sol_ida.t) < 100000

            for i in 1:length(sol_ida.t)
                vin = sol_ida.u[i][1]
                vout = sol_ida.u[i][2]
                @test !isnan(vin) && !isnan(vout)
                @test abs(vin) < 10.0
                @test abs(vout) < 10.0
            end

            vin_pos = voltage(acc, :vin, T/4)
            vout_pos = voltage(acc, :vout, T/4)
            @test vin_pos > 4.5
            @test vout_pos > 0.0

            vin_neg = voltage(acc, :vin, 3T/4)
            vout_neg = voltage(acc, :vout, 3T/4)
            @test vin_neg < -4.5
            @test vout_neg >= -0.1
        end

        @testset "MOSFET gate capacitance (sp_mos1 with IDA)" begin
            function cs_amp_with_cap(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                vgate = get_node!(ctx, :vgate)
                vdrain = get_node!(ctx, :vdrain)

                stamp!(VoltageSource(params.Vdd; name=:Vdd), ctx, vdd, 0)
                stamp!(SinVoltageSource(params.Vbias, params.Vac, params.freq; name=:Vg),
                       ctx, vgate, 0; t=t, _sim_mode_=spec.mode)
                stamp!(Resistor(params.Rd), ctx, vdd, vdrain)
                stamp!(sp_mos1(; l=1e-6, w=10e-6, vto=0.7, kp=1e-4,
                               cgso=1e-12, cgdo=0.5e-12),
                       ctx, vdrain, vgate, 0, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            circuit = MNACircuit(cs_amp_with_cap;
                                 Vdd=5.0, Vbias=1.5, Vac=0.1, freq=1000.0, Rd=2000.0)
            tspan = (0.0, 2e-3)

            sol_ida = tran!(circuit, tspan; abstol=1e-8, reltol=1e-6)
            @test sol_ida.retcode == SciMLBase.ReturnCode.Success

            sys = assemble!(circuit)
            acc = MNASolutionAccessor(sol_ida, sys)

            T = 1e-3

            for i in 1:length(sol_ida.t)
                vdd = sol_ida.u[i][1]
                vgate = sol_ida.u[i][2]
                vdrain = sol_ida.u[i][3]
                @test !isnan(vdd) && !isnan(vgate) && !isnan(vdrain)
                @test abs(vdd - 5.0) < 0.01
                @test vgate > 1.0 && vgate < 2.0
                @test vdrain > 0.0 && vdrain < 5.5
            end

            vd_t0 = voltage(acc, :vdrain, T)
            vd_pos = voltage(acc, :vdrain, T + T/4)
            vd_neg = voltage(acc, :vdrain, T + 3T/4)

            @test !isnan(vd_t0) && !isnan(vd_pos) && !isnan(vd_neg)
            @test vd_pos < vd_neg
        end

        @testset "Half-wave rectifier with filter cap (sp_diode with IDA)" begin
            function halfwave_rectifier(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vin = get_node!(ctx, :vin)
                vout = get_node!(ctx, :vout)

                stamp!(SinVoltageSource(0.0, params.Vamp, params.freq; name=:Vin),
                       ctx, vin, 0; t=t, _sim_mode_=spec.mode)
                stamp!(sp_diode(; is=1e-14, cjo=5e-12, m=0.5),
                       ctx, vin, vout; _mna_x_=x, _mna_spec_=spec)
                stamp!(Capacitor(params.C), ctx, vout, 0)
                stamp!(Resistor(params.R), ctx, vout, 0)

                return ctx
            end

            circuit = MNACircuit(halfwave_rectifier;
                                 Vamp=10.0, freq=50.0, R=1000.0, C=100e-6)
            tspan = (0.0, 40e-3)

            sol_ida = tran!(circuit, tspan; abstol=1e-6, reltol=1e-4)
            @test sol_ida.retcode == SciMLBase.ReturnCode.Success

            sys = assemble!(circuit)

            @test length(sol_ida.t) > 50
            @test length(sol_ida.t) < 100000

            for i in 1:length(sol_ida.t)
                vin = sol_ida.u[i][1]
                vout = sol_ida.u[i][2]
                @test !isnan(vin) && !isnan(vout)
                @test abs(vin) < 15.0
                @test abs(vout) < 15.0
            end

            vout_idx = findfirst(n -> n == :vout, sys.node_names)
            if vout_idx !== nothing
                v_final = sol_ida(40e-3)[vout_idx]
                @test v_final > 5.0
                @test v_final < 12.0
            end
        end

    end  # Tier 8

    GC.gc()

    @info "All integration tests completed" final_mem_mb=round(mem_mb(); digits=0)

end  # VADistiller Integration Tests
