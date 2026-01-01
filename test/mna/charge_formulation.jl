#==============================================================================#
# Tests for Voltage-Dependent Capacitor Detection and Charge Formulation
#
# See doc/voltage_dependent_capacitors.md for design details.
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: va_ddt, is_voltage_dependent_charge, CapacitanceDerivTag
using CedarSim.MNA: ContributionTag, JacobianTag
using ForwardDiff: Dual, value, partials

@testset "Voltage-Dependent Capacitor Detection" begin

    @testset "Detection basics - constant capacitors" begin
        # Linear capacitor: I = C * ddt(V)
        # Q(V) = C * V, so ∂²Q/∂V² = 0 -> constant capacitance
        C0 = 1e-12
        linear_cap(V) = C0 * va_ddt(V)

        @test !is_voltage_dependent_charge(linear_cap, 0.0, 0.0)
        @test !is_voltage_dependent_charge(linear_cap, 1.0, 0.0)
        @test !is_voltage_dependent_charge(linear_cap, 5.0, 2.0)

        # Parallel RC: I = V/R + C * ddt(V)
        # Still constant capacitance
        R = 1000.0
        parallel_rc(V) = V/R + C0 * va_ddt(V)

        @test !is_voltage_dependent_charge(parallel_rc, 1.0, 0.0)
    end

    @testset "Detection basics - voltage-dependent capacitors" begin
        # Quadratic charge: Q(V) = C0 * V^2
        # C(V) = dQ/dV = 2*C0*V -> varies with V
        # ∂C/∂V = 2*C0 ≠ 0
        C0 = 1e-12
        quadratic_cap(V) = va_ddt(C0 * V^2)

        @test is_voltage_dependent_charge(quadratic_cap, 1.0, 0.0)
        @test is_voltage_dependent_charge(quadratic_cap, 0.0, 0.0)  # Even at V=0, ∂C/∂V ≠ 0

        # Cubic charge: Q(V) = C0 * V^3
        # C(V) = 3*C0*V^2 -> varies with V
        # ∂²Q/∂V² = 6*C0*V = 0 at V=0, but perturbation detects it
        cubic_cap(V) = va_ddt(C0 * V^3)

        @test is_voltage_dependent_charge(cubic_cap, 1.0, 0.0)
        @test is_voltage_dependent_charge(cubic_cap, 0.0, 0.0)  # Detected via perturbation
    end

    @testset "Detection - junction capacitance model" begin
        # Standard junction capacitance: Q(V) = Cj0 * φ * (1 - (1 - V/φ)^(1-m))
        # This is the integral of C(V) = Cj0 * (1 - V/φ)^(-m)
        Cj0 = 1e-12
        phi = 0.7
        m = 0.5

        function junction_charge(V)
            # Avoid singularity at V = φ
            V_clamped = min(V, 0.95 * phi)
            Q = Cj0 * phi * (1 - (1 - V_clamped/phi)^(1-m)) / (1-m)
            return va_ddt(Q)
        end

        @test is_voltage_dependent_charge(junction_charge, 0.0, 0.0)
        @test is_voltage_dependent_charge(junction_charge, 0.3, 0.0)
        @test is_voltage_dependent_charge(junction_charge, 0.5, 0.0)
    end

    @testset "Detection - pure resistive (no capacitance)" begin
        # Resistor: I = V/R
        R = 1000.0
        resistor(V) = V / R

        @test !is_voltage_dependent_charge(resistor, 1.0, 0.0)

        # Diode: I = Is*(exp(V/Vt) - 1)
        Is = 1e-14
        Vt = 0.026
        diode(V) = Is * (exp(V / Vt) - 1)

        @test !is_voltage_dependent_charge(diode, 0.6, 0.0)
    end

    @testset "Detection - mixed resistive and constant capacitive" begin
        # Diode with constant junction capacitance
        # I = Is*(exp(V/Vt) - 1) + C * ddt(V)
        Is = 1e-14
        Vt = 0.026
        Cj = 1e-12

        diode_with_cap(V) = Is * (exp(V / Vt) - 1) + Cj * va_ddt(V)

        # Capacitance is constant, so should return false
        @test !is_voltage_dependent_charge(diode_with_cap, 0.6, 0.0)
    end

    @testset "Detection - mixed resistive and voltage-dependent capacitive" begin
        # Diode with voltage-dependent junction capacitance
        Is = 1e-14
        Vt = 0.026
        Cj0 = 1e-12
        phi = 0.7
        m = 0.5

        function diode_with_vdep_cap(V)
            I_res = Is * (exp(V / Vt) - 1)
            V_clamped = min(V, 0.95 * phi)
            Q = Cj0 * phi * (1 - (1 - V_clamped/phi)^(1-m)) / (1-m)
            return I_res + va_ddt(Q)
        end

        @test is_voltage_dependent_charge(diode_with_vdep_cap, 0.3, 0.0)
    end

end

@testset "Charge State Variable Allocation" begin
    using CedarSim.MNA: MNAContext, get_node!, alloc_charge!, has_charge, get_charge_idx
    using CedarSim.MNA: system_size, n_charges

    ctx = MNAContext()
    a = get_node!(ctx, :a)
    c = get_node!(ctx, :c)

    @test n_charges(ctx) == 0
    @test system_size(ctx) == 2  # Just nodes

    # Allocate a charge variable
    q_idx = alloc_charge!(ctx, :Q_test, a, c)

    @test n_charges(ctx) == 1
    @test system_size(ctx) == 3  # nodes + charge
    @test has_charge(ctx, :Q_test)
    @test get_charge_idx(ctx, :Q_test) == q_idx
    @test q_idx == 3  # After 2 nodes

    # Allocate another
    b = get_node!(ctx, :b)
    q_idx2 = alloc_charge!(ctx, :Q_test2, b, c)

    @test n_charges(ctx) == 2
    @test system_size(ctx) == 5  # 3 nodes + 2 charges
    @test q_idx2 == 5
end

@testset "stamp_charge_state! basic" begin
    using CedarSim.MNA: MNAContext, MNASystem, get_node!, stamp!, assemble!
    using CedarSim.MNA: stamp_charge_state!, VoltageSource

    # Test: voltage source + nonlinear capacitor to ground
    ctx = MNAContext()
    vcc = get_node!(ctx, :vcc)

    # Voltage source
    stamp!(VoltageSource(1.0; name=:V1), ctx, vcc, 0)

    # Quadratic charge: Q(V) = C0 * V^2, so C(V) = 2*C0*V
    C0 = 1e-12
    q_fn(V) = C0 * V^2

    # Need to provide x vector for operating point
    x = [1.0, 0.0]  # [Vcc, I_V1]

    q_idx = stamp_charge_state!(ctx, vcc, 0, q_fn, x, :Q_cap)

    @test q_idx == 3  # After node + current

    sys = assemble!(ctx)

    # Check C matrix has charge formulation entries
    # The charge constraint row is ALGEBRAIC (no C[q_idx, q_idx])
    # The dq/dt terms appear in KCL rows via C[vcc, q_idx]
    @test sys.C[q_idx, q_idx] ≈ 0.0  # Algebraic constraint - no mass

    # C[vcc, q_idx] = 1 (current flows out of vcc)
    @test sys.C[vcc, q_idx] ≈ 1.0

    # G matrix has constraint entries
    # G[q_idx, q_idx] = 1
    @test sys.G[q_idx, q_idx] ≈ 1.0

    # G[q_idx, vcc] = -dQ/dVp = -2*C0*V = -2e-12 at V=1
    @test sys.G[q_idx, vcc] ≈ -2e-12
end

@testset "stamp_charge_state! with both nodes" begin
    using CedarSim.MNA: MNAContext, get_node!, assemble!
    using CedarSim.MNA: stamp_charge_state!

    ctx = MNAContext()
    p = get_node!(ctx, :p)
    n = get_node!(ctx, :n)

    # Linear charge for simpler verification: Q(V) = C * V
    C = 1e-9
    q_fn(V) = C * V

    x = [2.0, 1.0]  # Vp=2, Vn=1, so Vpn=1

    q_idx = stamp_charge_state!(ctx, p, n, q_fn, x, :Q_C)

    sys = assemble!(ctx)

    # KCL coupling
    @test sys.C[p, q_idx] ≈ 1.0   # Current leaves p
    @test sys.C[n, q_idx] ≈ -1.0  # Current enters n

    # Constraint Jacobian
    @test sys.G[q_idx, q_idx] ≈ 1.0
    @test sys.G[q_idx, p] ≈ -C     # -dQ/dVp
    @test sys.G[q_idx, n] ≈ C      # -dQ/dVn = -(-C) = C
end

@testset "stamp_reactive_with_detection!" begin
    using CedarSim.MNA: MNAContext, get_node!, assemble!
    using CedarSim.MNA: stamp_reactive_with_detection!, va_ddt, n_charges

    @testset "Linear capacitor -> C matrix stamping" begin
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)

        C = 1e-12
        contrib_fn(V) = C * va_ddt(V)  # Linear: Q = C*V

        x = [1.0, 0.0]
        used_charge = stamp_reactive_with_detection!(ctx, p, n, contrib_fn, x, :Q_test)

        @test !used_charge  # Should use C matrix, not charge variable
        @test n_charges(ctx) == 0

        sys = assemble!(ctx)

        # C matrix should have standard capacitance pattern
        @test sys.C[p, p] ≈ C
        @test sys.C[p, n] ≈ -C
        @test sys.C[n, p] ≈ -C
        @test sys.C[n, n] ≈ C
    end

    @testset "Nonlinear capacitor -> charge formulation" begin
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)

        C0 = 1e-12
        contrib_fn(V) = va_ddt(C0 * V^2)  # Nonlinear: Q = C0*V^2, C = 2*C0*V

        x = [1.0, 0.0]
        used_charge = stamp_reactive_with_detection!(ctx, p, n, contrib_fn, x, :Q_test)

        @test used_charge  # Should use charge formulation
        @test n_charges(ctx) == 1

        sys = assemble!(ctx)

        # Should have charge variable
        q_idx = 3  # After 2 nodes

        # Charge formulation entries
        # The constraint row is ALGEBRAIC (no C[q_idx, q_idx])
        @test sys.C[q_idx, q_idx] ≈ 0.0  # Algebraic constraint
        @test sys.C[p, q_idx] ≈ 1.0      # KCL coupling
        @test sys.C[n, q_idx] ≈ -1.0

        # Constraint entries in G
        @test sys.G[q_idx, q_idx] ≈ 1.0  # ∂/∂q = 1
    end

    @testset "Mixed resistive + linear capacitive" begin
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)

        R = 1000.0
        C = 1e-12
        contrib_fn(V) = V/R + C * va_ddt(V)  # Resistive + linear capacitive

        x = [1.0, 0.0]
        used_charge = stamp_reactive_with_detection!(ctx, p, n, contrib_fn, x, :Q_test)

        @test !used_charge  # Linear capacitor, use C matrix
        @test n_charges(ctx) == 0
    end
end

@testset "Charge formulation solver integration" begin
    using CedarSim.MNA: MNAContext, get_node!, stamp!, VoltageSource, assemble!
    using CedarSim.MNA: stamp_charge_state!, detect_differential_vars

    @testset "Charge constraint is algebraic" begin
        # Create a simple circuit with a nonlinear capacitor
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)

        # Voltage source to provide DC bias
        stamp!(VoltageSource(1.0; name=:V1), ctx, vcc, 0)

        # Nonlinear capacitor: Q(V) = C0 * V^2
        C0 = 1e-12
        q_fn(V) = C0 * V^2

        x = [1.0, 0.0]  # [Vcc, I_V1]
        q_idx = stamp_charge_state!(ctx, vcc, 0, q_fn, x, :Q_cap)

        sys = assemble!(ctx)

        # Detect which variables are differential
        diff_vars = detect_differential_vars(sys)

        # vcc node should be differential (has KCL with dq/dt term)
        @test diff_vars[vcc] == true

        # Charge variable should be ALGEBRAIC (constraint q = Q(V))
        @test diff_vars[q_idx] == false

        # Current variable (voltage source) should be algebraic
        I_idx = 2  # After vcc node
        @test diff_vars[I_idx] == false
    end

    @testset "KCL has dq/dt terms" begin
        # Create circuit with nonlinear cap between two nodes
        ctx = MNAContext()
        p = get_node!(ctx, :p)
        n = get_node!(ctx, :n)

        C0 = 1e-12
        q_fn(V) = C0 * V^2

        x = [1.0, 0.5]  # [Vp, Vn]
        q_idx = stamp_charge_state!(ctx, p, n, q_fn, x, :Q_cap)

        sys = assemble!(ctx)

        # KCL at p includes +dq/dt (current leaves)
        @test sys.C[p, q_idx] ≈ 1.0

        # KCL at n includes -dq/dt (current enters)
        @test sys.C[n, q_idx] ≈ -1.0

        # Constraint row has no C entries (algebraic)
        @test sys.C[q_idx, q_idx] ≈ 0.0
        @test sys.C[q_idx, p] ≈ 0.0
        @test sys.C[q_idx, n] ≈ 0.0
    end
end
