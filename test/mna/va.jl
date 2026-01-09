#==============================================================================#
# MNA Phase 5: Verilog-A Integration Tests
#
# Tests for VA contribution functions with s-dual approach.
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!
using CedarSim.MNA: voltage, current, make_ode_problem, MNACircuit
using CedarSim: dc!
using CedarSim.MNA: va_ddt, stamp_current_contribution!, evaluate_contribution, ContributionTag, JacobianTag
using CedarSim.MNA: VoltageSource, Resistor  # Use MNA versions explicitly
using ForwardDiff: Dual, value, partials
using OrdinaryDiffEq

const deftol = 1e-6
isapprox_deftol(a, b) = isapprox(a, b; atol=deftol, rtol=deftol)

@testset "MNA VA Integration (Phase 5)" begin

    @testset "va_ddt s-dual basics" begin
        # Test that va_ddt creates proper s-dual structure
        x = 5.0
        result = va_ddt(x)
        @test result isa Dual{ContributionTag}
        @test value(result) == 0.0  # s*x at s=0
        @test partials(result, 1) == 5.0  # coefficient of s

        # Test with Dual input (for nested differentiation)
        x_dual = Dual{JacobianTag}(3.0, 1.0)
        result2 = va_ddt(x_dual)
        @test result2 isa Dual{ContributionTag}
        @test value(result2) == 0.0  # s*x at s=0
    end

    @testset "evaluate_contribution - resistor" begin
        # I(p,n) <+ V(p,n)/R  where R = 1000
        R = 1000.0
        contrib_fn(V) = V / R

        result = evaluate_contribution(contrib_fn, 5.0, 0.0)

        # Current at Vpn = 5V: I = 5/1000 = 5mA
        @test isapprox_deftol(result.I, 0.005)
        # dI/dVp = 1/R = 0.001, dI/dVn = -1/R = -0.001
        @test isapprox_deftol(result.dI_dVp, 0.001)
        @test isapprox_deftol(result.dI_dVn, -0.001)
        # No reactive part
        @test isapprox_deftol(result.q, 0.0)
        @test isapprox_deftol(result.dq_dVp, 0.0)
        @test isapprox_deftol(result.dq_dVn, 0.0)
    end

    @testset "evaluate_contribution - capacitor" begin
        # I(p,n) <+ C*ddt(V(p,n))  where C = 1e-6
        C = 1e-6
        contrib_fn(V) = C * va_ddt(V)

        result = evaluate_contribution(contrib_fn, 5.0, 0.0)

        # No DC current through capacitor
        @test isapprox_deftol(result.I, 0.0)
        @test isapprox_deftol(result.dI_dVp, 0.0)
        @test isapprox_deftol(result.dI_dVn, 0.0)
        # Charge = C * V = 1e-6 * 5 = 5e-6
        @test isapprox_deftol(result.q, 5e-6)
        # dq/dVp = C = 1e-6, dq/dVn = -C = -1e-6
        @test isapprox_deftol(result.dq_dVp, 1e-6)
        @test isapprox_deftol(result.dq_dVn, -1e-6)
    end

    @testset "evaluate_contribution - parallel RC" begin
        # I(p,n) <+ V(p,n)/R + C*ddt(V(p,n))
        R = 1000.0
        C = 1e-6
        contrib_fn(V) = V / R + C * va_ddt(V)

        result = evaluate_contribution(contrib_fn, 5.0, 0.0)

        # Resistive current: I = V/R = 5/1000 = 5mA
        @test isapprox_deftol(result.I, 0.005)
        # dI/dVp = 1/R = 0.001
        @test isapprox_deftol(result.dI_dVp, 0.001)
        @test isapprox_deftol(result.dI_dVn, -0.001)
        # Charge = C * V = 5e-6
        @test isapprox_deftol(result.q, 5e-6)
        # dq/dVp = C = 1e-6
        @test isapprox_deftol(result.dq_dVp, 1e-6)
    end

    @testset "evaluate_contribution - diode" begin
        # I(p,n) <+ Is*(exp(V(p,n)/Vt) - 1)
        Is = 1e-14
        Vt = 0.026  # ~26mV at room temp
        contrib_fn(V) = Is * (exp(V / Vt) - 1)

        # Test at V = 0.6V (typical forward bias)
        result = evaluate_contribution(contrib_fn, 0.6, 0.0)

        # Expected current: Is*(exp(0.6/0.026) - 1) ≈ Is*exp(23.08) ≈ 1e-4 A
        expected_I = Is * (exp(0.6 / Vt) - 1)
        @test isapprox(result.I, expected_I; rtol=1e-6)

        # Expected conductance: dI/dV = Is/Vt * exp(V/Vt) ≈ I/Vt
        expected_G = Is / Vt * exp(0.6 / Vt)
        @test isapprox(result.dI_dVp, expected_G; rtol=1e-6)

        # No reactive part
        @test isapprox_deftol(result.q, 0.0)
    end

    @testset "stamp_current_contribution! - resistor" begin
        # Create context and stamp a resistor via contribution function
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)

        R = 1000.0
        contrib_fn(V) = V / R

        # Stamp at Vp=5, Vn=0
        x = [5.0, 0.0]
        stamp_current_contribution!(ctx, p, n, contrib_fn, x)

        sys = assemble!(ctx)

        # G matrix should have conductance pattern:
        # G[p,p] = 1/R = 0.001
        # G[p,n] = -1/R = -0.001
        # G[n,p] = -1/R = -0.001
        # G[n,n] = 1/R = 0.001
        @test isapprox_deftol(sys.G[p, p], 0.001)
        @test isapprox_deftol(sys.G[p, n], -0.001)
        @test isapprox_deftol(sys.G[n, p], -0.001)
        @test isapprox_deftol(sys.G[n, n], 0.001)
    end

    @testset "stamp_current_contribution! - capacitor" begin
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)

        C = 1e-6
        contrib_fn(V) = C * va_ddt(V)

        x = [5.0, 0.0]
        stamp_current_contribution!(ctx, p, n, contrib_fn, x)

        sys = assemble!(ctx)

        # C matrix should have capacitance pattern:
        # C[p,p] = C = 1e-6
        # C[p,n] = -C = -1e-6
        # C[n,p] = -C = -1e-6
        # C[n,n] = C = 1e-6
        @test isapprox_deftol(sys.C[p, p], 1e-6)
        @test isapprox_deftol(sys.C[p, n], -1e-6)
        @test isapprox_deftol(sys.C[n, p], -1e-6)
        @test isapprox_deftol(sys.C[n, n], 1e-6)

        # G matrix should be zero (no resistive part)
        @test isapprox_deftol(sys.G[p, p], 0.0)
    end

    @testset "stamp_current_contribution! - parallel RC" begin
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)

        R = 1000.0
        C = 1e-6
        contrib_fn(V) = V / R + C * va_ddt(V)

        x = [5.0, 0.0]
        stamp_current_contribution!(ctx, p, n, contrib_fn, x)

        sys = assemble!(ctx)

        # G matrix should have 1/R
        @test isapprox_deftol(sys.G[p, p], 0.001)
        @test isapprox_deftol(sys.G[p, n], -0.001)

        # C matrix should have C
        @test isapprox_deftol(sys.C[p, p], 1e-6)
        @test isapprox_deftol(sys.C[p, n], -1e-6)
    end

    @testset "stamp_current_contribution! - nonlinear diode" begin
        # Test that nonlinear contributions properly stamp based on operating point
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)

        Is = 1e-14
        Vt = 0.026
        contrib_fn(V) = Is * (exp(V / Vt) - 1)

        # Stamp at forward bias (Vp=0.6V, Vn=0V)
        x = [0.6, 0.0]
        stamp_current_contribution!(ctx, p, n, contrib_fn, x)

        sys = assemble!(ctx)

        # Conductance should be derivative at operating point
        expected_G = Is / Vt * exp(0.6 / Vt)
        @test isapprox(sys.G[p, p], expected_G; rtol=1e-6)
        @test isapprox(sys.G[p, n], -expected_G; rtol=1e-6)

        # RHS uses Newton companion model: b = I - G*V at operating point
        # This ensures Newton iteration converges correctly for nonlinear elements
        expected_I = Is * (exp(0.6 / Vt) - 1)
        Vp, Vn = 0.6, 0.0
        dI_dVp, dI_dVn = expected_G, -expected_G
        expected_b = expected_I - dI_dVp * Vp - dI_dVn * Vn
        @test isapprox(sys.b[p], -expected_b; rtol=1e-6)
    end

    # VA→MNA integration tests using va_str macro
    # Note: Don't use `include "disciplines.vams"` - disciplines are implicit
    # and the include causes a parser bug with IOBuffer sources.

    @testset "VA resistor via va_str" begin
        # Define a simple VA resistor
        va"""
        module VAResistor(p, n);
            parameter real R = 1000.0;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/R;
        endmodule
        """

        # Build circuit: VoltageSource + VA Resistor
        function va_resistor_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)

            stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
            stamp!(VAResistor(R=2000.0), ctx, vcc, 0)

            return ctx
        end

        circuit = MNACircuit(va_resistor_circuit)
        sol = dc!(circuit)

        # V = 5V, R = 2000Ω, I = 2.5mA
        # Voltage source current = -2.5mA (negative = sourcing)
        @test isapprox_deftol(voltage(sol, :vcc), 5.0)
        @test isapprox(current(sol, :I_V1), -0.0025; atol=1e-5)
    end

    @testset "VA capacitor transient" begin
        # Define a VA capacitor
        va"""
        module VACapacitor(p, n);
            parameter real C = 1e-6;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ C*ddt(V(p,n));
        endmodule
        """

        # RC circuit: V -> R -> C -> GND
        R_val = 1000.0
        C_val = 1e-6
        V_val = 5.0

        function va_rc_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            cap = get_node!(ctx, :cap)

            stamp!(VoltageSource(V_val; name=:V1), ctx, vcc, 0)
            stamp!(Resistor(R_val; name=:R1), ctx, vcc, cap)
            stamp!(VACapacitor(C=C_val), ctx, cap, 0)

            return ctx
        end

        ctx = va_rc_circuit((;), MNASpec(mode=:tran))
        sys = assemble!(ctx)

        # Set up transient simulation
        tau = R_val * C_val  # Time constant
        tspan = (0.0, 5 * tau)
        prob_data = make_ode_problem(sys, tspan)

        # Start with capacitor uncharged
        u0 = copy(prob_data.u0)
        cap_idx = findfirst(n -> n == :cap, sys.node_names)
        u0[cap_idx] = 0.0

        f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                        jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
        prob = ODEProblem(f, u0, prob_data.tspan)
        sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-8)

        # Check RC charging behavior
        # At t=0: V_cap = 0
        # At t=5τ: V_cap ≈ V_val * (1 - exp(-5)) ≈ 0.993 * V_val
        @test isapprox(sol.u[1][cap_idx], 0.0; atol=1e-6)
        expected_final = V_val * (1 - exp(-5))
        @test isapprox(sol.u[end][cap_idx], expected_final; rtol=0.01)
    end

    @testset "VA parallel RC" begin
        # Test a device with both resistive and reactive parts
        va"""
        module VAParallelRC(p, n);
            parameter real R = 1000.0;
            parameter real C = 1e-6;
            inout p, n;
            electrical p, n;
            analog I(p,n) <+ V(p,n)/R + C*ddt(V(p,n));
        endmodule
        """

        function va_parallel_rc_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                CedarSim.MNA.reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)

            stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
            stamp!(VAParallelRC(R=1000.0, C=1e-6), ctx, vcc, 0)

            return ctx
        end

        ctx = va_parallel_rc_circuit((;), MNASpec())
        sys = assemble!(ctx)

        # G matrix should have 1/R = 0.001
        vcc_idx = findfirst(n -> n == :vcc, sys.node_names)
        @test isapprox(sys.G[vcc_idx, vcc_idx], 0.001; atol=1e-6)

        # C matrix should have C = 1e-6
        @test isapprox(sys.C[vcc_idx, vcc_idx], 1e-6; atol=1e-9)

        # DC solution: I = V/R = 5/1000 = 5mA
        circuit = MNACircuit(va_parallel_rc_circuit)
        sol = dc!(circuit)
        @test isapprox_deftol(voltage(sol, :vcc), 5.0)
        @test isapprox(current(sol, :I_V1), -0.005; atol=1e-5)
    end

end
