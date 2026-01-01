#==============================================================================#
# MNA Phase 6: VADistiller Model Integration Tests
#
# Tests for compatibility with models from https://codeberg.org/arpadbuermen/VADistiller
# These are SPICE3 models converted to Verilog-A.
#
# Approach:
# 1. Start with simplified versions that strip unsupported constructs
# 2. Gradually add support for missing VA features
# 3. Progress from simple (resistor) to complex (MOSFET) models
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: voltage, current, make_ode_problem
using CedarSim.MNA: va_ddt, stamp_current_contribution!, evaluate_contribution
using CedarSim.MNA: VoltageSource, Resistor, Capacitor, CurrentSource
using ForwardDiff: Dual, value, partials
using OrdinaryDiffEq
using VerilogAParser

const deftol = 1e-6
isapprox_deftol(a, b) = isapprox(a, b; atol=deftol, rtol=deftol)

@testset "VADistiller Models" begin

    #==========================================================================#
    # Tier 1: Simplified Linear Passives (2-terminal, no special constructs)
    #==========================================================================#

    @testset "Tier 1: Simplified Passives" begin

        @testset "Simple resistor" begin
            # Minimal resistor: just resistance parameter and I = V/R
            # Use unique module name to avoid conflicts
            va"""
            module VADResistor(pos, neg);
                parameter real resistance = 1000.0;
                inout pos, neg;
                electrical pos, neg;
                analog I(pos,neg) <+ V(pos,neg)/resistance;
            endmodule
            """

            # Test in voltage divider
            function resistor_divider(params, spec, t::Real=0.0)
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                mid = get_node!(ctx, :mid)

                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VADResistor(resistance=1000.0), ctx, vcc, mid)
                stamp!(VADResistor(resistance=1000.0), ctx, mid, 0)

                return ctx
            end

            ctx = resistor_divider((;), MNASpec())
            sys = assemble!(ctx)
            sol = solve_dc(sys)

            @test isapprox_deftol(voltage(sol, :vcc), 5.0)
            @test isapprox_deftol(voltage(sol, :mid), 2.5)  # Voltage divider
            @test isapprox(current(sol, :I_V1), -0.0025; atol=1e-5)  # 5V / 2kΩ
        end

        @testset "Simple capacitor" begin
            # Minimal capacitor: just capacitance parameter and I = C*dV/dt
            va"""
            module VADCapacitor(pos, neg);
                parameter real capacitance = 1e-6;
                inout pos, neg;
                electrical pos, neg;
                analog I(pos,neg) <+ capacitance*ddt(V(pos,neg));
            endmodule
            """

            # Test in RC circuit transient
            R_val = 1000.0
            C_val = 1e-6
            V_val = 5.0

            function rc_circuit(params, spec, t::Real=0.0)
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                cap = get_node!(ctx, :cap)

                stamp!(VoltageSource(V_val; name=:V1), ctx, vcc, 0)
                stamp!(Resistor(R_val; name=:R1), ctx, vcc, cap)
                stamp!(VADCapacitor(capacitance=C_val), ctx, cap, 0)

                return ctx
            end

            ctx = rc_circuit((;), MNASpec(mode=:tran))
            sys = assemble!(ctx)

            # Set up transient
            tau = R_val * C_val
            tspan = (0.0, 5 * tau)
            prob_data = make_ode_problem(sys, tspan)

            u0 = copy(prob_data.u0)
            cap_idx = findfirst(n -> n == :cap, sys.node_names)
            u0[cap_idx] = 0.0

            f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                            jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
            prob = ODEProblem(f, u0, prob_data.tspan)
            sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-8)

            # Check RC charging: V_cap(t) = V * (1 - e^(-t/τ))
            @test isapprox(sol.u[1][cap_idx], 0.0; atol=1e-6)
            expected_final = V_val * (1 - exp(-5))
            @test isapprox(sol.u[end][cap_idx], expected_final; rtol=0.01)
        end

    end

    #==========================================================================#
    # Tier 2: Diode (nonlinear, 2-terminal)
    #==========================================================================#

    @testset "Tier 2: Simple Diode" begin

        @testset "Ideal diode (exponential)" begin
            # Minimal Shockley diode: I = Is*(exp(V/Vt) - 1)
            va"""
            module VADDiode(a, c);
                parameter real Is = 1e-14;
                parameter real N = 1.0;
                inout a, c;
                electrical a, c;
                analog begin
                    I(a,c) <+ Is*(exp(V(a,c)/(N*0.02585)) - 1.0);
                end
            endmodule
            """

            # Test forward bias: V=0.6V should give I ≈ 1e-4 A
            # Use Newton iteration DC solver for nonlinear devices
            function diode_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                anode = get_node!(ctx, :anode)

                stamp!(VoltageSource(0.6; name=:V1), ctx, anode, 0)
                stamp!(VADDiode(Is=1e-14, N=1.0), ctx, anode, 0; _mna_x_=x)

                return ctx
            end

            # Use builder-based solve_dc which does Newton iteration
            sol = solve_dc(diode_circuit, (;), MNASpec())

            # Expected: I = 1e-14 * (exp(0.6/0.02585) - 1) ≈ 1.05e-4 A
            Vt = 0.02585
            expected_I = 1e-14 * (exp(0.6/Vt) - 1)
            actual_I = -current(sol, :I_V1)  # Negative because V1 sources current
            @test isapprox(actual_I, expected_I; rtol=0.01)
        end

        @testset "Diode with series resistance" begin
            # Diode with Rs for more realistic behavior
            # This tests internal nodes which require phase 6 features
            va"""
            module VADDiodeRs(a, c);
                parameter real Is = 1e-14;
                parameter real N = 1.0;
                parameter real Rs = 10.0;
                inout a, c;
                electrical a, c, a_int;
                analog begin
                    I(a, a_int) <+ V(a, a_int) / Rs;
                    I(a_int, c) <+ Is*(exp(V(a_int,c)/(N*0.02585)) - 1.0);
                end
            endmodule
            """

            # Forward bias test with internal node
            function diode_rs_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                anode = get_node!(ctx, :anode)

                stamp!(VoltageSource(0.7; name=:V1), ctx, anode, 0)
                stamp!(VADDiodeRs(Is=1e-14, N=1.0, Rs=10.0), ctx, anode, 0; _mna_x_=x)

                return ctx
            end

            # Solve with Newton iteration
            sol = solve_dc(diode_rs_circuit, (;), MNASpec())

            # With Rs=10Ω, at 0.7V forward bias:
            # V_internal ≈ 0.6V (drops ~0.1V across Rs)
            # I ≈ 10mA (rough estimate)
            # The exact current depends on Newton convergence
            actual_I = -current(sol, :I_V1)
            @test actual_I > 0  # Current should flow
            @test actual_I < 0.1  # Should be less than 100mA
        end

    end

    #==========================================================================#
    # Tier 3: 3-terminal devices (BJT, JFET)
    #==========================================================================#

    @testset "Tier 3: Simple 3-Terminal Devices" begin

        @testset "Simple VCCS (JFET approximation)" begin
            # Linear VCCS: Ids = gm * Vgs
            va"""
            module VADVCCS(d, g, s);
                parameter real gm = 1e-3;
                inout d, g, s;
                electrical d, g, s;
                analog begin
                    I(d,s) <+ gm * V(g,s);
                end
            endmodule
            """

            # Test: Vgs controls Ids
            function vccs_circuit(params, spec, t::Real=0.0)
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                # Power supply
                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                # Gate bias
                stamp!(VoltageSource(1.0; name=:Vg), ctx, gate, 0)
                # Load resistor
                stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                # VCCS: Ids = gm * Vgs
                stamp!(VADVCCS(gm=1e-3), ctx, drain, gate, 0)

                return ctx
            end

            ctx = vccs_circuit((;), MNASpec())
            sys = assemble!(ctx)
            sol = solve_dc(sys)

            # Ids = gm * Vgs = 1e-3 * 1.0 = 1mA
            # Vdrain = Vdd - Ids * Rd = 5 - 1e-3 * 1000 = 4V
            @test isapprox(voltage(sol, :drain), 4.0; atol=0.01)
        end

        @testset "Simple square-law MOSFET" begin
            # Level 1 MOSFET in saturation: Ids = (K/2)(Vgs - Vth)²
            # Simplified: always-on for testing (no threshold cutoff)
            va"""
            module VADMOS(d, g, s);
                parameter real Kp = 1e-4;
                parameter real Vth = 0.5;
                inout d, g, s;
                electrical d, g, s;
                analog I(d,s) <+ (Kp/2.0) * (V(g,s) - Vth) * (V(g,s) - Vth);
            endmodule
            """

            # Common source amplifier
            # Use Newton iteration DC solver for nonlinear devices
            function mos_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(1.5; name=:Vg), ctx, gate, 0)  # Vgs = 1.5V
                stamp!(Resistor(10000.0; name=:Rd), ctx, vdd, drain)  # 10kΩ load
                stamp!(VADMOS(Kp=1e-4, Vth=0.5), ctx, drain, gate, 0; _mna_x_=x)

                return ctx
            end

            # Use builder-based solve_dc which does Newton iteration
            sol = solve_dc(mos_circuit, (;), MNASpec())

            # Ids = (1e-4/2) * (1.5 - 0.5)² = 0.5e-4 * 1 = 50µA
            # Vdrain = 5 - 50e-6 * 10000 = 5 - 0.5 = 4.5V
            @test isapprox(voltage(sol, :drain), 4.5; atol=0.1)
        end

    end

    #==========================================================================#
    # Tier 4: N-type MOSFET with bulk terminal
    #==========================================================================#

    @testset "Tier 4: 4-Terminal Devices" begin

        @testset "Simple NMOS with bulk" begin
            # 4-terminal NMOS: d, g, s, b
            # Simplified square-law in saturation (no body effect for now)
            va"""
            module VADNMOS4(d, g, s, b);
                parameter real Kp = 1e-4;
                parameter real Vth = 0.5;
                inout d, g, s, b;
                electrical d, g, s, b;
                analog I(d,s) <+ (Kp/2.0) * (V(g,s) - Vth) * (V(g,s) - Vth);
            endmodule
            """

            # Common source amplifier with bulk tied to source
            function nmos4_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(1.5; name=:Vg), ctx, gate, 0)
                stamp!(Resistor(10000.0; name=:Rd), ctx, vdd, drain)
                # 4-terminal stamp: d=drain, g=gate, s=0, b=0 (bulk tied to source)
                stamp!(VADNMOS4(Kp=1e-4, Vth=0.5), ctx, drain, gate, 0, 0; _mna_x_=x)

                return ctx
            end

            # Use builder-based solve_dc
            sol = solve_dc(nmos4_circuit, (;), MNASpec())

            # Same result as 3-terminal MOSFET
            @test isapprox(voltage(sol, :drain), 4.5; atol=0.1)
        end

    end

    #==========================================================================#
    # Tier 5: MOSFET with Capacitances and Internal Nodes
    #==========================================================================#

    @testset "Tier 5: MOSFET with Capacitances" begin

        @testset "MOSFET with gate capacitances (DC)" begin
            # MOSFET with Cgs and Cgd - tests reactive contributions
            va"""
            module VAMOSCap(d, g, s);
                parameter real Kp = 1e-4;
                parameter real Vth = 0.5;
                parameter real Cgs = 1e-12;
                parameter real Cgd = 0.5e-12;
                inout d, g, s;
                electrical d, g, s;
                analog begin
                    // Square-law drain current
                    I(d,s) <+ (Kp/2.0) * (V(g,s) - Vth) * (V(g,s) - Vth);
                    // Gate capacitances
                    I(g,s) <+ Cgs * ddt(V(g,s));
                    I(g,d) <+ Cgd * ddt(V(g,d));
                end
            endmodule
            """

            # DC test - capacitors should not affect DC operating point
            function moscap_dc_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                stamp!(Resistor(10000.0; name=:Rd), ctx, vdd, drain)
                stamp!(VAMOSCap(Kp=1e-4, Vth=0.5, Cgs=1e-12, Cgd=0.5e-12), ctx, drain, gate, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            sol = solve_dc(moscap_dc_circuit, (;), MNASpec())

            # Ids = (1e-4/2) * (2.0 - 0.5)² = 0.5e-4 * 2.25 = 112.5µA
            # Vdrain = 5 - 112.5e-6 * 10000 = 5 - 1.125 = 3.875V
            @test isapprox(voltage(sol, :drain), 3.875; atol=0.1)
            @test isapprox(voltage(sol, :gate), 2.0; atol=1e-6)
        end

        @testset "MOSFET with gate capacitances (Transient)" begin
            using OrdinaryDiffEq

            # Transient test - verify C matrix is correctly stamped
            function moscap_tran_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                stamp!(Resistor(10000.0; name=:Rd), ctx, vdd, drain)
                stamp!(VAMOSCap(Kp=1e-4, Vth=0.5, Cgs=1e-12, Cgd=0.5e-12), ctx, drain, gate, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            ctx = moscap_tran_circuit((;), MNASpec(mode=:tran))
            sys = assemble!(ctx)

            # Check C matrix has correct entries
            C = Matrix(sys.C)
            gate_idx = findfirst(n -> n == :gate, sys.node_names)
            drain_idx = findfirst(n -> n == :drain, sys.node_names)

            # Cgs + Cgd at gate node diagonal
            @test C[gate_idx, gate_idx] ≈ 1.5e-12 atol=1e-15
            # -Cgd at gate-drain (Miller capacitance)
            @test C[gate_idx, drain_idx] ≈ -0.5e-12 atol=1e-15
            @test C[drain_idx, gate_idx] ≈ -0.5e-12 atol=1e-15
            # Cgd at drain diagonal
            @test C[drain_idx, drain_idx] ≈ 0.5e-12 atol=1e-15

            # Run transient simulation
            tspan = (0.0, 10e-9)  # 10ns
            prob_data = make_ode_problem(sys, tspan)

            f = ODEFunction(prob_data.f;
                mass_matrix=prob_data.mass_matrix,
                jac=prob_data.jac,
                jac_prototype=prob_data.jac_prototype)
            prob = ODEProblem(f, prob_data.u0, prob_data.tspan)
            sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-8)

            @test sol.retcode == SciMLBase.ReturnCode.Success
        end

        @testset "MOSFET with source/drain resistances (internal nodes)" begin
            # MOSFET with Rs and Rd - tests internal node handling
            va"""
            module VAMOSRsd(d, g, s);
                parameter real Kp = 1e-4;
                parameter real Vth = 0.5;
                parameter real Rs = 10.0;
                parameter real Rd = 10.0;
                inout d, g, s;
                electrical d, g, s, d_int, s_int;
                analog begin
                    // External drain to internal drain resistance
                    I(d, d_int) <+ V(d, d_int) / Rd;
                    // External source to internal source resistance
                    I(s, s_int) <+ V(s, s_int) / Rs;
                    // Intrinsic MOSFET between internal nodes
                    I(d_int, s_int) <+ (Kp/2.0) * (V(g,s_int) - Vth) * (V(g,s_int) - Vth);
                end
            endmodule
            """

            function mosrsd_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                stamp!(Resistor(10000.0; name=:Rd_load), ctx, vdd, drain)
                stamp!(VAMOSRsd(Kp=1e-4, Vth=0.5, Rs=10.0, Rd=10.0), ctx, drain, gate, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            sol = solve_dc(mosrsd_circuit, (;), MNASpec())

            # With small Rs/Rd, drain voltage should be close to ideal MOSFET
            # Ids ≈ 112.5µA, voltage drops: Ids*Rs ≈ 1.1mV, Ids*Rd ≈ 1.1mV
            # So drain voltage should be very close to 3.875V
            @test isapprox(voltage(sol, :drain), 3.875; atol=0.1)
        end

        @testset "MOSFET with capacitances AND internal nodes" begin
            using OrdinaryDiffEq

            # Full MOSFET model: Rs, Rd, Cgs, Cgd
            va"""
            module VAMOSFull(d, g, s);
                parameter real Kp = 1e-4;
                parameter real Vth = 0.5;
                parameter real Rs = 10.0;
                parameter real Rd = 10.0;
                parameter real Cgs = 1e-12;
                parameter real Cgd = 0.5e-12;
                inout d, g, s;
                electrical d, g, s, d_int, s_int;
                analog begin
                    // Series resistances
                    I(d, d_int) <+ V(d, d_int) / Rd;
                    I(s, s_int) <+ V(s, s_int) / Rs;
                    // Intrinsic MOSFET
                    I(d_int, s_int) <+ (Kp/2.0) * (V(g,s_int) - Vth) * (V(g,s_int) - Vth);
                    // Gate capacitances to internal nodes
                    I(g, s_int) <+ Cgs * ddt(V(g, s_int));
                    I(g, d_int) <+ Cgd * ddt(V(g, d_int));
                end
            endmodule
            """

            # DC test
            function mosfull_dc_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                stamp!(Resistor(10000.0; name=:Rd_load), ctx, vdd, drain)
                stamp!(VAMOSFull(Kp=1e-4, Vth=0.5, Rs=10.0, Rd=10.0, Cgs=1e-12, Cgd=0.5e-12),
                       ctx, drain, gate, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            sol_dc = solve_dc(mosfull_dc_circuit, (;), MNASpec())
            @test isapprox(voltage(sol_dc, :drain), 3.875; atol=0.1)

            # Transient test
            function mosfull_tran_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(2.0; name=:Vg), ctx, gate, 0)
                stamp!(Resistor(10000.0; name=:Rd_load), ctx, vdd, drain)
                stamp!(VAMOSFull(Kp=1e-4, Vth=0.5, Rs=10.0, Rd=10.0, Cgs=1e-12, Cgd=0.5e-12),
                       ctx, drain, gate, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            ctx = mosfull_tran_circuit((;), MNASpec(mode=:tran))
            sys = assemble!(ctx)

            # Verify internal nodes were allocated
            @test length(sys.node_names) >= 5  # vdd, gate, drain + 2 internal

            # C matrix should have capacitance entries
            C = Matrix(sys.C)
            @test count(!=(0), C) > 0  # Should have non-zero entries

            # Run transient
            tspan = (0.0, 10e-9)
            prob_data = make_ode_problem(sys, tspan)

            f = ODEFunction(prob_data.f;
                mass_matrix=prob_data.mass_matrix,
                jac=prob_data.jac,
                jac_prototype=prob_data.jac_prototype)
            prob = ODEProblem(f, prob_data.u0, prob_data.tspan)
            sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-8)

            @test sol.retcode == SciMLBase.ReturnCode.Success
        end

        @testset "MOSFET inverter DC and small-signal" begin
            # Test a simple inverter with capacitive load

            # DC operating point
            function inverter_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                vin = get_node!(ctx, :vin)
                vout = get_node!(ctx, :vout)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(2.5; name=:Vin), ctx, vin, 0)  # Mid-rail input
                stamp!(Resistor(10000.0; name=:Rd), ctx, vdd, vout)
                stamp!(VAMOSCap(Kp=1e-4, Vth=0.5, Cgs=1e-12, Cgd=0.5e-12), ctx, vout, vin, 0; _mna_x_=x, _mna_spec_=spec)
                # Load capacitance
                stamp!(Capacitor(1e-12; name=:Cload), ctx, vout, 0)

                return ctx
            end

            sol_dc = solve_dc(inverter_circuit, (;), MNASpec())

            # At Vgs=2.5V: Ids = (1e-4/2) * (2.5-0.5)² = 200µA
            # Vout = 5 - 200e-6 * 10000 = 3V
            @test isapprox(voltage(sol_dc, :vout), 3.0; atol=0.2)

            # Verify capacitances are present in the system matrix
            ctx = inverter_circuit((;), MNASpec(mode=:tran))
            sys = assemble!(ctx)

            C = Matrix(sys.C)
            vout_idx = findfirst(n -> n == :vout, sys.node_names)

            # Should have load capacitance + Cgd at vout
            # Cload = 1e-12, Cgd = 0.5e-12
            @test C[vout_idx, vout_idx] ≈ 1.5e-12 atol=1e-15
        end

    end

    #==========================================================================#
    # Feature Tests: VA System Functions
    #==========================================================================#

    @testset "Feature: VA System Functions" begin

        @testset "\$temperature access" begin
            # Test temperature-dependent resistor using $temperature
            va_code = raw"""
            module VATempResistor(p, n);
                parameter real R0 = 1000.0;
                parameter real TC = 0.004;
                inout p, n;
                electrical p, n;
                analog begin
                    I(p,n) <+ V(p,n) / (R0 * (1.0 + TC * ($temperature() - 300.0)));
                end
            endmodule
            """
            va = VerilogAParser.parse(IOBuffer(va_code))
            Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

            # Test at default temp (27C = 300.15K)
            function temp_resistor_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)

                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VATempResistor(R0=1000.0, TC=0.004), ctx, vcc, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            # At 27°C (300.15K), resistance should be ~R0
            spec = MNASpec(temp=27.0)
            sol = solve_dc(temp_resistor_circuit, (;), spec)
            # R = 1000 * (1 + 0.004 * (300.15 - 300)) = 1000 * 1.0006 ≈ 1000.6
            # I = 5V / 1000.6 ≈ 0.005
            @test isapprox(current(sol, :I_V1), -0.005; atol=1e-4)

            # At 100°C (373.15K), resistance should be higher
            spec_hot = MNASpec(temp=100.0)
            sol_hot = solve_dc(temp_resistor_circuit, (;), spec_hot)
            # R = 1000 * (1 + 0.004 * (373.15 - 300)) = 1000 * 1.293 ≈ 1292.6
            # I = 5V / 1292.6 ≈ 0.00387
            @test isapprox(current(sol_hot, :I_V1), -0.00387; atol=1e-4)
        end

        @testset "\$simparam access" begin
            # Test $simparam with default value
            va_code = raw"""
            module VAGminResistor(p, n);
                parameter real R = 1000.0;
                inout p, n;
                electrical p, n;
                analog begin
                    I(p,n) <+ V(p,n) / R + V(p,n) * $simparam("gmin", 1e-12);
                end
            endmodule
            """
            va = VerilogAParser.parse(IOBuffer(va_code))
            Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

            function gmin_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)

                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VAGminResistor(R=1000.0), ctx, vcc, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            # With default gmin=1e-12
            spec = MNASpec()
            sol = solve_dc(gmin_circuit, (;), spec)
            # I ≈ 5/1000 + 5*1e-12 ≈ 0.005
            @test isapprox(current(sol, :I_V1), -0.005; atol=1e-6)

            # With custom gmin=1e-3 (very high for testing)
            spec_gmin = MNASpec(gmin=1e-3)
            sol_gmin = solve_dc(gmin_circuit, (;), spec_gmin)
            # I = 5/1000 + 5*1e-3 = 0.005 + 0.005 = 0.01
            @test isapprox(current(sol_gmin, :I_V1), -0.01; atol=1e-6)
        end

        @testset "\$param_given check" begin
            # Test $param_given to check if parameter was explicitly set
            # Uses ternary expression since if/else with contributions needs more work
            va_code = raw"""
            module VAOptionalParam(p, n);
                parameter real R = 1000.0;
                parameter real Ralt = 500.0;
                inout p, n;
                electrical p, n;
                analog begin
                    I(p,n) <+ V(p,n) / ($param_given(R) ? R : Ralt);
                end
            endmodule
            """
            va = VerilogAParser.parse(IOBuffer(va_code))
            Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

            # With explicit R=2000 (R is "given")
            sol_explicit = solve_dc((p,s,t=0.0; x=Float64[]) -> begin
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VAOptionalParam(R=2000.0, Ralt=500.0), ctx, vcc, 0; _mna_x_=x, _mna_spec_=s)
                return ctx
            end, (;), MNASpec())
            # $param_given(R) is true, so uses R=2000
            # I = 5V / 2000Ω = 0.0025A
            @test isapprox(current(sol_explicit, :I_V1), -0.0025; atol=1e-6)

            # With only Ralt given (R is NOT "given")
            sol_default = solve_dc((p,s,t=0.0; x=Float64[]) -> begin
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VAOptionalParam(Ralt=500.0), ctx, vcc, 0; _mna_x_=x, _mna_spec_=s)
                return ctx
            end, (;), MNASpec())
            # $param_given(R) is false, so uses Ralt=500
            # I = 5V / 500Ω = 0.01A
            @test isapprox(current(sol_default, :I_V1), -0.01; atol=1e-6)
        end

    end

    #==========================================================================#
    # Feature Tests: Parser Extensions Still Needed
    #==========================================================================#

    @testset "Feature: aliasparam" begin
        # Test aliasparam declaration - allows parameter aliases
        va_code = raw"""
        module VAAliasTest(p, n);
            parameter real tnom = 27.0;
            aliasparam tref = tnom;
            inout p, n;
            electrical p, n;
            analog begin
                I(p,n) <+ V(p,n) / (1000.0 * (1.0 + 0.001 * (tnom - 27.0)));
            end
        endmodule
        """
        va = VerilogAParser.parse(IOBuffer(va_code))
        Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

        # Test 1: Using the real parameter name (tnom)
        sol_tnom = solve_dc((p,s,t=0.0; x=Float64[]) -> begin
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
            stamp!(VAAliasTest(tnom=100.0), ctx, vcc, 0; _mna_x_=x, _mna_spec_=s)
            return ctx
        end, (;), MNASpec())
        # R = 1000 * (1 + 0.001 * (100 - 27)) = 1000 * 1.073 = 1073
        # I = 5V / 1073Ω ≈ 0.00466A
        @test isapprox(current(sol_tnom, :I_V1), -0.00466; atol=1e-4)

        # Test 2: Using the alias (tref) - should have same effect as tnom
        sol_tref = solve_dc((p,s,t=0.0; x=Float64[]) -> begin
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
            stamp!(VAAliasTest(tref=100.0), ctx, vcc, 0; _mna_x_=x, _mna_spec_=s)
            return ctx
        end, (;), MNASpec())
        # Same result - alias forwards to tnom
        @test isapprox(current(sol_tref, :I_V1), -0.00466; atol=1e-4)

        # Test 3: Verify property access - dev.tref should return dev.tnom
        dev = VAAliasTest(tnom=50.0)
        @test dev.tnom == dev.tref  # Alias returns target value
    end

    @testset "Feature: Parser Extensions Needed" begin

        @testset "Real variable with initialization" begin
            # VADistiller uses `real G = 0.0;` at module scope
            # NOW WORKING: parser handles inline variable initialization

            # Test module with variable initialization
            va"""
            module VarInitResistor(p, n);
                parameter real R = 1000.0;
                real temp_var = 0.0;
                real scale = 1.0;
                inout p, n;
                electrical p, n;
                analog begin
                    temp_var = V(p, n) * scale;
                    I(p, n) <+ temp_var / R;
                end
            endmodule
            """

            # Test in voltage divider circuit
            function var_init_divider(params, spec, t::Real=0.0)
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                mid = get_node!(ctx, :mid)

                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VarInitResistor(R=1000.0), ctx, vcc, mid)
                stamp!(VarInitResistor(R=1000.0), ctx, mid, 0)

                return ctx
            end

            ctx = var_init_divider((;), MNASpec())
            sys = assemble!(ctx)
            sol = solve_dc(sys)

            @test isapprox_deftol(voltage(sol, :vcc), 5.0)
            @test isapprox_deftol(voltage(sol, :mid), 2.5)  # Voltage divider
            @test isapprox(current(sol, :I_V1), -0.0025; atol=1e-5)  # 5V / 2kΩ
        end

        @testset "Real variable with non-zero initialization" begin
            # Test that non-zero initial values work correctly
            va"""
            module VarInitNonZero(p, n);
                parameter real R = 1000.0;
                real offset = 0.5;  // Non-zero initial value
                inout p, n;
                electrical p, n;
                analog begin
                    I(p, n) <+ (V(p, n) + offset) / R;
                end
            endmodule
            """

            function var_init_offset(params, spec, t::Real=0.0)
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)

                stamp!(VoltageSource(4.5; name=:V1), ctx, vcc, 0)
                stamp!(VarInitNonZero(R=1000.0), ctx, vcc, 0)

                return ctx
            end

            ctx = var_init_offset((;), MNASpec())
            sys = assemble!(ctx)
            sol = solve_dc(sys)

            @test isapprox_deftol(voltage(sol, :vcc), 4.5)
            # Current should be (4.5 + 0.5) / 1000 = 5mA
            @test isapprox(current(sol, :I_V1), -0.005; atol=1e-5)
        end

        @testset "Internal nodes" begin
            # VADistiller models like BJT have internal nodes (b_int, c_int, etc.)
            # Phase 6.2: Internal node support is now implemented!

            # Test 1: Simple resistor with internal node (Rs in series)
            va"""
            module VAInternalResistor(p, n);
                parameter real R1 = 1000.0;
                parameter real R2 = 1000.0;
                inout p, n;
                electrical p, n, mid;
                analog begin
                    I(p, mid) <+ V(p, mid) / R1;
                    I(mid, n) <+ V(mid, n) / R2;
                end
            endmodule
            """

            # Voltage divider using internal node
            function internal_resistor_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)

                stamp!(VoltageSource(10.0; name=:V1), ctx, vcc, 0)
                stamp!(VAInternalResistor(R1=1000.0, R2=1000.0), ctx, vcc, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            sol = solve_dc(internal_resistor_circuit, (;), MNASpec())

            # With two 1kΩ in series across 10V:
            # Total R = 2kΩ, I = 10V / 2kΩ = 5mA
            @test isapprox(current(sol, :I_V1), -0.005; atol=1e-5)
            @test isapprox(voltage(sol, :vcc), 10.0; atol=1e-6)

            # Test 2: Multiple internal nodes (3 resistors in series)
            va"""
            module VAMultiInternal(p, n);
                parameter real R = 1000.0;
                inout p, n;
                electrical p, n, mid1, mid2;
                analog begin
                    I(p, mid1) <+ V(p, mid1) / R;
                    I(mid1, mid2) <+ V(mid1, mid2) / R;
                    I(mid2, n) <+ V(mid2, n) / R;
                end
            endmodule
            """

            function multi_internal_circuit(params, spec, t::Real=0.0; x=Float64[])
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)

                stamp!(VoltageSource(9.0; name=:V1), ctx, vcc, 0)
                stamp!(VAMultiInternal(R=1000.0), ctx, vcc, 0; _mna_x_=x, _mna_spec_=spec)

                return ctx
            end

            sol2 = solve_dc(multi_internal_circuit, (;), MNASpec())

            # With three 1kΩ in series across 9V:
            # Total R = 3kΩ, I = 9V / 3kΩ = 3mA
            @test isapprox(current(sol2, :I_V1), -0.003; atol=1e-5)
        end

    end

end  # VADistiller Models

#==============================================================================#
# Note: Tier 6+ tests (VADistiller file-based models, transient circuits) have
# been moved to vadistiller_integration.jl to reduce memory pressure during CI.
# Run integration tests with: Pkg.test(test_args=["integration"])
#==============================================================================#
