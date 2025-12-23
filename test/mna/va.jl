#==============================================================================#
# MNA Phase 5: Verilog-A Integration Tests
#
# Tests for VA contribution functions with s-dual approach.
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: voltage, current, make_ode_problem
using CedarSim.MNA: va_ddt, stamp_current_contribution!, evaluate_contribution, ContributionTag
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
        x_dual = Dual{Nothing}(3.0, 1.0)
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

        # Current stamped as -I into b[p] (current leaving node p)
        expected_I = Is * (exp(0.6 / Vt) - 1)
        @test isapprox(sys.b[p], -expected_I; rtol=1e-6)
    end

    # Note: The va_str integration tests require the VerilogA preprocessor to
    # resolve `include "disciplines.vams"`. This works when loading .va files
    # but not with inline va_str from IOBuffer. Those tests are skipped here
    # but should be tested via loading actual .va files.
    #
    # TODO: Test full VA→MNA pipeline by loading test .va files instead of
    # inline va_str strings.

end
