#==============================================================================#
# MNA Phase 6: Multi-Terminal VA Devices and MOSFET Tests
#
# Tests for Verilog-A devices with 3+ terminals, building up to MOSFET models.
# This file progressively tests:
# 1. Simple 3-terminal devices (VCCS-style)
# 2. Basic MOSFET with resistive channel
# 3. MOSFET with gate capacitances (Cgs, Cgd)
# 4. Nonlinear voltage-dependent capacitors
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: voltage, current, make_ode_problem
using CedarSim.MNA: va_ddt, stamp_current_contribution!, evaluate_contribution
using CedarSim.MNA: VoltageSource, CurrentSource, Resistor
using ForwardDiff: Dual, value, partials
using OrdinaryDiffEq

const deftol = 1e-6
isapprox_deftol(a, b) = isapprox(a, b; atol=deftol, rtol=deftol)

@testset "MNA VA Multi-Terminal Devices (Phase 6)" begin

    #==========================================================================#
    # Test 1: Simple 3-terminal VCCS (Voltage Controlled Current Source)
    #
    # This tests the n-terminal code generation path with a device where
    # output current depends on input voltage (different terminals).
    #==========================================================================#
    @testset "3-terminal VCCS" begin
        # VCCS: Output current proportional to input voltage
        # I(out_p, out_n) = gm * V(in_p, in_n)
        # For 3 terminals: s is shared ground
        # I(d, s) = gm * V(g, s)
        va"""
        module VA_VCCS(d, g, s);
            parameter real gm = 0.001;
            inout d, g, s;
            electrical d, g, s;
            analog I(d, s) <+ gm * V(g, s);
        endmodule
        """

        # Build test circuit - use builder pattern for Newton iteration
        function vccs_test_circuit(params, spec; x=Float64[])
            ctx = MNAContext()
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)

            stamp!(VoltageSource(1.0; name=:Vg), ctx, g, 0)
            stamp!(VoltageSource(5.0; name=:Vdd), ctx, d, 0)
            stamp!(VA_VCCS(gm=0.002), ctx, d, g, 0; _mna_x_=x)

            return ctx
        end

        sol = solve_dc(vccs_test_circuit, (;), MNASpec())

        # Check voltages
        @test isapprox(voltage(sol, :g), 1.0; atol=1e-6)
        @test isapprox(voltage(sol, :d), 5.0; atol=1e-6)

        # Check current: I_Vdd should supply the VCCS current
        # Id = gm * Vgs = 0.002 * 1.0 = 2mA
        I_Vdd = current(sol, :I_Vdd)
        @test isapprox(I_Vdd, -0.002; atol=1e-5)  # Negative because sourcing
    end

    #==========================================================================#
    # Test 2: Simple Linear MOSFET (resistive channel only)
    #
    # Shichman-Hodges style model in linear region:
    # Ids = K * ((Vgs - Vth)*Vds - Vds^2/2) for Vds < Vgs-Vth
    # Ids = K/2 * (Vgs - Vth)^2 for Vds >= Vgs-Vth (saturation)
    #==========================================================================#
    @testset "Simple Linear MOSFET (resistive only)" begin
        va"""
        module SimpleMOS(d, g, s);
            parameter real K = 1e-3;
            parameter real Vth = 0.5;
            inout d, g, s;
            electrical d, g, s;
            real Vgs, Vds, Vov, Ids;
            analog begin
                Vgs = V(g, s);
                Vds = V(d, s);
                Vov = Vgs - Vth;

                if (Vov <= 0) begin
                    // Cutoff
                    Ids = 0;
                end else if (Vds < Vov) begin
                    // Linear region
                    Ids = K * (Vov * Vds - Vds * Vds / 2);
                end else begin
                    // Saturation
                    Ids = K / 2 * Vov * Vov;
                end

                I(d, s) <+ Ids;
            end
        endmodule
        """

        # Test in saturation: Vgs = 1.5V, Vds = 2V
        # Vov = 1.5 - 0.5 = 1V
        # Ids = K/2 * Vov^2 = 0.001/2 * 1 = 0.5mA
        function simple_mos_circuit(params, spec; x=Float64[])
            ctx = MNAContext()
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)

            stamp!(VoltageSource(1.5; name=:Vg), ctx, g, 0)
            stamp!(VoltageSource(2.0; name=:Vd), ctx, d, 0)
            stamp!(SimpleMOS(K=1e-3, Vth=0.5), ctx, d, g, 0; _mna_x_=x)

            return ctx
        end

        sol = solve_dc(simple_mos_circuit, (;), MNASpec())

        @test isapprox(voltage(sol, :g), 1.5; atol=1e-6)
        @test isapprox(voltage(sol, :d), 2.0; atol=1e-6)

        # Drain current: Ids = K/2 * (1.5 - 0.5)^2 = 0.5mA
        I_Vd = current(sol, :I_Vd)
        @test isapprox(I_Vd, -0.0005; atol=1e-5)  # Negative = sourcing
    end

    @testset "SimpleMOS in linear region" begin
        # Test in linear region: Vgs = 1.5V, Vds = 0.3V
        # Vov = 1.5 - 0.5 = 1V, Vds = 0.3 < Vov
        # Ids = K * (Vov * Vds - Vds^2/2) = 0.001 * (1*0.3 - 0.045) = 0.255mA
        function simple_mos_linear_circuit(params, spec; x=Float64[])
            ctx = MNAContext()
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)

            stamp!(VoltageSource(1.5; name=:Vg), ctx, g, 0)
            stamp!(VoltageSource(0.3; name=:Vd), ctx, d, 0)
            stamp!(SimpleMOS(K=1e-3, Vth=0.5), ctx, d, g, 0; _mna_x_=x)

            return ctx
        end

        sol = solve_dc(simple_mos_linear_circuit, (;), MNASpec())

        @test isapprox(voltage(sol, :d), 0.3; atol=1e-6)

        # Ids = K * (Vov * Vds - Vds^2/2) = 0.001 * (1.0 * 0.3 - 0.09/2) = 0.000255
        I_Vd = current(sol, :I_Vd)
        @test isapprox(I_Vd, -0.000255; atol=1e-6)
    end

    @testset "SimpleMOS in cutoff" begin
        # Test in cutoff: Vgs = 0.3V < Vth = 0.5V
        function simple_mos_cutoff_circuit(params, spec; x=Float64[])
            ctx = MNAContext()
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)

            stamp!(VoltageSource(0.3; name=:Vg), ctx, g, 0)
            stamp!(VoltageSource(2.0; name=:Vd), ctx, d, 0)
            stamp!(SimpleMOS(K=1e-3, Vth=0.5), ctx, d, g, 0; _mna_x_=x)

            return ctx
        end

        sol = solve_dc(simple_mos_cutoff_circuit, (;), MNASpec())

        # No current in cutoff
        I_Vd = current(sol, :I_Vd)
        @test isapprox(I_Vd, 0.0; atol=1e-8)
    end

    #==========================================================================#
    # Test 3: MOSFET with Gate Capacitance
    #
    # Add Cgs and Cgd capacitances to the simple MOSFET
    #==========================================================================#
    @testset "MOSFET with gate capacitance" begin
        va"""
        module CapMOS(d, g, s);
            parameter real K = 1e-3;
            parameter real Vth = 0.5;
            parameter real Cgs = 10e-15;
            parameter real Cgd = 5e-15;
            inout d, g, s;
            electrical d, g, s;
            real Vgs, Vds, Vov, Ids;
            analog begin
                Vgs = V(g, s);
                Vds = V(d, s);
                Vov = Vgs - Vth;

                // Resistive channel current
                if (Vov <= 0) begin
                    Ids = 0;
                end else if (Vds < Vov) begin
                    Ids = K * (Vov * Vds - Vds * Vds / 2);
                end else begin
                    Ids = K / 2 * Vov * Vov;
                end

                I(d, s) <+ Ids;

                // Gate capacitances
                I(g, s) <+ Cgs * ddt(V(g, s));
                I(g, d) <+ Cgd * ddt(V(g, d));
            end
        endmodule
        """

        # DC test - capacitors should have no effect on DC
        function cap_mos_dc_circuit(params, spec; x=Float64[])
            ctx = MNAContext()
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)

            stamp!(VoltageSource(1.5; name=:Vg), ctx, g, 0)
            stamp!(VoltageSource(2.0; name=:Vd), ctx, d, 0)
            stamp!(CapMOS(K=1e-3, Vth=0.5, Cgs=10e-15, Cgd=5e-15), ctx, d, g, 0; _mna_x_=x)

            return ctx
        end

        sol = solve_dc(cap_mos_dc_circuit, (;), MNASpec())

        # Same as SimpleMOS in saturation
        I_Vd = current(sol, :I_Vd)
        @test isapprox(I_Vd, -0.0005; atol=1e-5)

        # Check that capacitances are stamped into C matrix
        ctx = cap_mos_dc_circuit((;), MNASpec(mode=:tran); x=sol.x)
        sys = assemble!(ctx)
        @test !iszero(sys.C)
    end

    @testset "CapMOS transient - gate charging" begin
        # Transient test: charging gate through resistor
        # Pulse Vg from 0 to 2V, observe gate charging
        R_gate = 1e3
        Cgs = 10e-12  # 10pF for visible time constant
        Cgd = 5e-12

        function cap_mos_tran_circuit(params, spec; x=Float64[])
            ctx = MNAContext()
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)
            vin = get_node!(ctx, :vin)

            # Input voltage through gate resistor
            stamp!(VoltageSource(2.0; name=:Vin), ctx, vin, 0)
            stamp!(Resistor(R_gate; name=:Rg), ctx, vin, g)

            # Drain tied to high voltage
            stamp!(VoltageSource(3.0; name=:Vd), ctx, d, 0)

            # MOSFET with capacitances
            stamp!(CapMOS(K=1e-3, Vth=0.5, Cgs=Cgs, Cgd=Cgd), ctx, d, g, 0; _mna_x_=x)

            return ctx
        end

        ctx = cap_mos_tran_circuit((;), MNASpec(mode=:tran))
        sys = assemble!(ctx)

        # Time constant for gate charging
        # Effective capacitance at gate: Cgs + Cgd = 15pF
        # tau = Rg * (Cgs + Cgd) = 1000 * 15e-12 = 15ns
        tau = R_gate * (Cgs + Cgd)
        tspan = (0.0, 5 * tau)

        prob_data = make_ode_problem(sys, tspan)

        # Start with gate at 0V
        u0 = copy(prob_data.u0)
        g_idx = findfirst(n -> n == :g, sys.node_names)
        u0[g_idx] = 0.0

        f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                        jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
        prob = ODEProblem(f, u0, prob_data.tspan)
        sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-9)

        # Check that gate charges to ~2V with RC time constant behavior
        # At t=0: Vg = 0
        # At t=5tau: Vg ≈ 2 * (1 - exp(-5)) ≈ 1.986V
        @test isapprox(sol.u[1][g_idx], 0.0; atol=0.1)
        expected_final = 2.0 * (1 - exp(-5))
        @test isapprox(sol.u[end][g_idx], expected_final; rtol=0.05)
    end

    #==========================================================================#
    # Test 4: CMOS Inverter (NMOS + PMOS)
    #
    # This is the key test for MOSFET circuits
    #==========================================================================#
    @testset "CMOS Inverter DC" begin
        # Define NMOS
        va"""
        module NMOS(d, g, s);
            parameter real K = 1e-3;
            parameter real Vth = 0.5;
            inout d, g, s;
            electrical d, g, s;
            real Vgs, Vds, Vov, Ids;
            analog begin
                Vgs = V(g, s);
                Vds = V(d, s);
                Vov = Vgs - Vth;

                if (Vov <= 0) begin
                    Ids = 0;
                end else if (Vds < Vov) begin
                    Ids = K * (Vov * Vds - Vds * Vds / 2);
                end else begin
                    Ids = K / 2 * Vov * Vov;
                end

                I(d, s) <+ Ids;
            end
        endmodule
        """

        # Define PMOS (source at top, Vth negative relative to source)
        va"""
        module PMOS(d, g, s);
            parameter real K = 1e-3;
            parameter real Vth = 0.5;
            inout d, g, s;
            electrical d, g, s;
            real Vsg, Vsd, Vov, Ids;
            analog begin
                Vsg = V(s, g);
                Vsd = V(s, d);
                Vov = Vsg - Vth;

                if (Vov <= 0) begin
                    Ids = 0;
                end else if (Vsd < Vov) begin
                    Ids = K * (Vov * Vsd - Vsd * Vsd / 2);
                end else begin
                    Ids = K / 2 * Vov * Vov;
                end

                // Current flows from source to drain (out of source into drain)
                I(s, d) <+ Ids;
            end
        endmodule
        """

        Vdd = 3.0

        # Test with input LOW (0V) - output should be HIGH (Vdd)
        function inverter_low_circuit(params, spec; x=Float64[])
            ctx = MNAContext()
            vdd = get_node!(ctx, :vdd)
            out = get_node!(ctx, :out)
            inp = get_node!(ctx, :inp)

            stamp!(VoltageSource(Vdd; name=:Vdd), ctx, vdd, 0)
            stamp!(VoltageSource(0.0; name=:Vin), ctx, inp, 0)

            # PMOS: source=vdd, gate=inp, drain=out
            stamp!(PMOS(K=1e-3, Vth=0.5), ctx, out, inp, vdd; _mna_x_=x)

            # NMOS: drain=out, gate=inp, source=gnd
            stamp!(NMOS(K=1e-3, Vth=0.5), ctx, out, inp, 0; _mna_x_=x)

            # Small load to observe output
            stamp!(Resistor(1e6; name=:Rload), ctx, out, 0)

            return ctx
        end

        sol = solve_dc(inverter_low_circuit, (;), MNASpec())

        # With input LOW:
        # - PMOS on (Vsg = Vdd - 0 = 3V > Vth)
        # - NMOS off (Vgs = 0 < Vth)
        # - Output pulled to Vdd
        @test isapprox(voltage(sol, :out), Vdd; atol=0.01)
    end

    @testset "CMOS Inverter HIGH input" begin
        Vdd = 3.0

        # Test with input HIGH (Vdd) - output should be LOW (0V)
        function inverter_high_circuit(params, spec; x=Float64[])
            ctx = MNAContext()
            vdd = get_node!(ctx, :vdd)
            out = get_node!(ctx, :out)
            inp = get_node!(ctx, :inp)

            stamp!(VoltageSource(Vdd; name=:Vdd), ctx, vdd, 0)
            stamp!(VoltageSource(Vdd; name=:Vin), ctx, inp, 0)

            # PMOS: source=vdd, gate=inp, drain=out
            stamp!(PMOS(K=1e-3, Vth=0.5), ctx, out, inp, vdd; _mna_x_=x)

            # NMOS: drain=out, gate=inp, source=gnd
            stamp!(NMOS(K=1e-3, Vth=0.5), ctx, out, inp, 0; _mna_x_=x)

            # Small load
            stamp!(Resistor(1e6; name=:Rload), ctx, out, 0)

            return ctx
        end

        sol = solve_dc(inverter_high_circuit, (;), MNASpec())

        # With input HIGH:
        # - PMOS off (Vsg = 0 < Vth)
        # - NMOS on (Vgs = Vdd > Vth)
        # - Output pulled to 0V
        @test isapprox(voltage(sol, :out), 0.0; atol=0.01)
    end

    #==========================================================================#
    # Test 5: Nonlinear (voltage-dependent) capacitor
    #
    # Junction capacitance: C(V) = Cj0 / (1 - V/phi)^m
    #==========================================================================#
    @testset "Nonlinear junction capacitor" begin
        va"""
        module JunctionCap(p, n);
            parameter real Cj0 = 1e-12;
            parameter real phi = 0.8;
            parameter real m = 0.5;
            inout p, n;
            electrical p, n;
            real V, C;
            analog begin
                V = V(p, n);
                // Guard against V >= phi
                if (V < 0.5 * phi) begin
                    C = Cj0 / $pow(1 - V/phi, m);
                end else begin
                    C = Cj0 / $pow(0.5, m);
                end
                I(p, n) <+ C * ddt(V(p, n));
            end
        endmodule
        """

        # Test that capacitance varies with voltage
        function junction_cap_circuit(params, spec; x=Float64[])
            ctx = MNAContext()
            p = get_node!(ctx, :p)

            stamp!(VoltageSource(0.4; name=:V1), ctx, p, 0)
            stamp!(JunctionCap(Cj0=1e-12, phi=0.8, m=0.5), ctx, p, 0; _mna_x_=x)

            return ctx
        end

        ctx = junction_cap_circuit((;), MNASpec(mode=:tran); x=[0.4, 0.0])
        sys = assemble!(ctx)

        # At V=0.4V, C = Cj0 / (1 - 0.4/0.8)^0.5 = Cj0 / 0.5^0.5 = Cj0 * sqrt(2)
        # Expected C ≈ 1.414e-12
        # C matrix should have this capacitance
        p_idx = findfirst(n -> n == :p, sys.node_names)
        expected_C = 1e-12 / sqrt(0.5)
        @test isapprox(sys.C[p_idx, p_idx], expected_C; rtol=0.01)
    end

end  # testset
