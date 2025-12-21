#==============================================================================#
# MNA Phase 1: Core Tests
#
# This file contains comprehensive tests for the MNA core functionality:
# - Context and node allocation
# - Stamping primitives
# - Matrix assembly
# - Basic device stamps
# - DC and AC analysis
#
# Tests are validated against analytical solutions.
#==============================================================================#

using Test
using LinearAlgebra
using SparseArrays

# Import MNA module - use explicit imports to avoid conflicts with CedarSim types
using CedarSim.MNA: MNAContext, MNASystem, get_node!, alloc_current!
using CedarSim.MNA: stamp_G!, stamp_C!, stamp_b!, stamp_conductance!, stamp_capacitance!
using CedarSim.MNA: stamp!, system_size
using CedarSim.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using CedarSim.MNA: VCVS, VCCS, CCVS, CCCS
using CedarSim.MNA: assemble!, assemble_G, assemble_C, get_rhs
using CedarSim.MNA: DCSolution, ACSolution, solve_dc, solve_ac
using CedarSim.MNA: voltage, current, magnitude_db, phase_deg
using CedarSim.MNA: make_ode_problem, make_ode_function

@testset "MNA Core Tests" begin

    #==========================================================================#
    # Context and Node Allocation
    #==========================================================================#

    @testset "MNAContext basics" begin
        ctx = MNAContext()

        # Initially empty
        @test ctx.n_nodes == 0
        @test ctx.n_currents == 0
        @test system_size(ctx) == 0

        # Allocate nodes
        n1 = get_node!(ctx, :n1)
        @test n1 == 1
        @test ctx.n_nodes == 1

        n2 = get_node!(ctx, :n2)
        @test n2 == 2
        @test ctx.n_nodes == 2

        # Ground is always 0
        gnd = get_node!(ctx, :gnd)
        @test gnd == 0
        @test ctx.n_nodes == 2  # Ground doesn't add to node count

        gnd2 = get_node!(ctx, Symbol("0"))
        @test gnd2 == 0

        # Existing nodes return same index
        n1_again = get_node!(ctx, :n1)
        @test n1_again == 1
        @test ctx.n_nodes == 2

        # Current variables
        i1 = alloc_current!(ctx, :I_V1)
        @test i1 == 3  # n_nodes + 1
        @test ctx.n_currents == 1
        @test system_size(ctx) == 3
    end

    @testset "Stamping primitives" begin
        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)

        # Stamp G matrix
        stamp_G!(ctx, n1, n1, 1.0)
        stamp_G!(ctx, n1, n2, -1.0)
        stamp_G!(ctx, n2, n1, -1.0)
        stamp_G!(ctx, n2, n2, 1.0)

        @test length(ctx.G_V) == 4
        @test ctx.G_I == [1, 1, 2, 2]
        @test ctx.G_J == [1, 2, 1, 2]
        @test ctx.G_V == [1.0, -1.0, -1.0, 1.0]

        # Stamp C matrix
        stamp_C!(ctx, n1, n1, 1e-6)
        @test length(ctx.C_V) == 1

        # Stamp RHS
        stamp_b!(ctx, n1, 5.0)
        @test ctx.b[n1] == 5.0

        stamp_b!(ctx, n1, 3.0)  # Accumulates
        @test ctx.b[n1] == 8.0

        # Ground stamps are ignored
        stamp_G!(ctx, 0, n1, 1.0)
        @test length(ctx.G_V) == 4  # No change
        stamp_G!(ctx, n1, 0, 1.0)
        @test length(ctx.G_V) == 4  # No change
        stamp_b!(ctx, 0, 5.0)
        # No error, just ignored
    end

    #==========================================================================#
    # Matrix Assembly
    #==========================================================================#

    @testset "Matrix assembly" begin
        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)

        # Resistor pattern
        G_val = 1.0 / 1000.0  # 1k ohm
        stamp_conductance!(ctx, n1, n2, G_val)

        sys = assemble!(ctx)

        @test size(sys.G) == (2, 2)
        @test nnz(sys.G) == 4

        # Check matrix structure
        G_dense = Matrix(sys.G)
        @test G_dense[1, 1] ≈ G_val
        @test G_dense[1, 2] ≈ -G_val
        @test G_dense[2, 1] ≈ -G_val
        @test G_dense[2, 2] ≈ G_val
    end

    #==========================================================================#
    # Basic Device Stamps
    #==========================================================================#

    @testset "Resistor stamp" begin
        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)

        R = Resistor(1000.0)
        stamp!(R, ctx, n1, n2)

        sys = assemble!(ctx)
        G = Matrix(sys.G)

        expected_G = 1.0 / 1000.0
        @test G[1, 1] ≈ expected_G
        @test G[1, 2] ≈ -expected_G
        @test G[2, 1] ≈ -expected_G
        @test G[2, 2] ≈ expected_G
    end

    @testset "Capacitor stamp" begin
        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)

        C = Capacitor(1e-6)
        stamp!(C, ctx, n1, n2)

        sys = assemble!(ctx)

        # G should be empty
        @test nnz(sys.G) == 0

        # C should have capacitor pattern
        C_mat = Matrix(sys.C)
        @test C_mat[1, 1] ≈ 1e-6
        @test C_mat[1, 2] ≈ -1e-6
        @test C_mat[2, 1] ≈ -1e-6
        @test C_mat[2, 2] ≈ 1e-6
    end

    @testset "Voltage source stamp" begin
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)

        V = VoltageSource(5.0; name=:V1)
        I_idx = stamp!(V, ctx, vcc, 0)  # vcc to ground

        @test I_idx == 2  # n_nodes + 1

        sys = assemble!(ctx)
        @test size(sys.G) == (2, 2)

        G = Matrix(sys.G)
        # [vcc, I_V1] x [vcc, I_V1]
        # KCL at vcc: +1 * I_V1
        @test G[1, 2] ≈ 1.0
        # Voltage equation: Vcc - 0 = 5
        @test G[2, 1] ≈ 1.0

        # RHS
        @test sys.b[2] ≈ 5.0
    end

    @testset "Current source stamp" begin
        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)

        I = CurrentSource(0.001)  # 1mA
        stamp!(I, ctx, n1, 0)  # Into n1 from ground

        sys = assemble!(ctx)

        # G should be empty (no conductance)
        @test nnz(sys.G) == 0

        # RHS should have current
        @test sys.b[1] ≈ 0.001
    end

    @testset "Inductor stamp" begin
        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)

        L = Inductor(1e-3; name=:L1)  # 1mH
        I_idx = stamp!(L, ctx, n1, n2)

        @test I_idx == 3  # After 2 nodes

        sys = assemble!(ctx)
        @test size(sys.G) == (3, 3)
        @test size(sys.C) == (3, 3)

        G = Matrix(sys.G)
        C_mat = Matrix(sys.C)

        # KCL at n1: +I_L
        @test G[1, 3] ≈ 1.0
        # KCL at n2: -I_L
        @test G[2, 3] ≈ -1.0
        # Voltage equation: V(n1) - V(n2) = L*dI/dt
        @test G[3, 1] ≈ 1.0
        @test G[3, 2] ≈ -1.0
        # C matrix: -L on diagonal for I_L
        @test C_mat[3, 3] ≈ -1e-3
    end

    #==========================================================================#
    # Controlled Sources
    #==========================================================================#

    @testset "VCCS stamp" begin
        ctx = MNAContext()
        out_p = get_node!(ctx, :out_p)
        out_n = get_node!(ctx, :out_n)
        in_p = get_node!(ctx, :in_p)
        in_n = get_node!(ctx, :in_n)

        G_dev = VCCS(0.01)  # gm = 10mS
        stamp!(G_dev, ctx, out_p, out_n, in_p, in_n)

        sys = assemble!(ctx)
        G = Matrix(sys.G)

        # I(out) = gm * V(in), but MNA uses "current leaving is positive"
        # So current INTO out_p means negative contribution to G row for out_p
        @test G[1, 3] ≈ -0.01  # out_p, in_p: -gm (current enters out_p)
        @test G[1, 4] ≈ 0.01   # out_p, in_n: +gm
        @test G[2, 3] ≈ 0.01   # out_n, in_p: +gm (current leaves out_n)
        @test G[2, 4] ≈ -0.01  # out_n, in_n: -gm
    end

    @testset "VCVS stamp" begin
        ctx = MNAContext()
        out_p = get_node!(ctx, :out_p)
        out_n = get_node!(ctx, :out_n)
        in_p = get_node!(ctx, :in_p)
        in_n = get_node!(ctx, :in_n)

        E = VCVS(10.0; name=:E1)  # Gain = 10
        I_idx = stamp!(E, ctx, out_p, out_n, in_p, in_n)

        @test I_idx == 5  # After 4 nodes

        sys = assemble!(ctx)
        @test size(sys.G) == (5, 5)

        G = Matrix(sys.G)

        # KCL at output nodes
        @test G[1, 5] ≈ 1.0   # out_p
        @test G[2, 5] ≈ -1.0  # out_n

        # Voltage equation: Vout - gain*Vin = 0
        @test G[5, 1] ≈ 1.0    # Vout_p
        @test G[5, 2] ≈ -1.0   # Vout_n
        @test G[5, 3] ≈ -10.0  # -gain * Vin_p
        @test G[5, 4] ≈ 10.0   # -gain * (-Vin_n)
    end

    #==========================================================================#
    # DC Analysis - Analytical Validation
    #==========================================================================#

    @testset "DC: Voltage divider" begin
        # Classic voltage divider: 5V source, 1k/1k resistors
        # Expected: Vout = 5 * 1k/(1k+1k) = 2.5V

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
        stamp!(Resistor(1000.0), ctx, vcc, out)
        stamp!(Resistor(1000.0), ctx, out, 0)

        sol = solve_dc(ctx)

        @test voltage(sol, :vcc) ≈ 5.0
        @test voltage(sol, :out) ≈ 2.5 atol=1e-10
    end

    @testset "DC: Voltage divider (unequal)" begin
        # 5V source, 2k/1k resistors
        # Expected: Vout = 5 * 1k/(2k+1k) = 5/3 ≈ 1.6667V

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(5.0), ctx, vcc, 0)
        stamp!(Resistor(2000.0), ctx, vcc, out)
        stamp!(Resistor(1000.0), ctx, out, 0)

        sol = solve_dc(ctx)

        @test voltage(sol, :out) ≈ 5.0 / 3.0 atol=1e-10
    end

    @testset "DC: Current source into resistor" begin
        # 1mA current source into 1k resistor
        # Expected: V = I * R = 0.001 * 1000 = 1V

        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)

        stamp!(CurrentSource(0.001), ctx, n1, 0)  # 1mA into n1
        stamp!(Resistor(1000.0), ctx, n1, 0)

        sol = solve_dc(ctx)

        @test voltage(sol, :n1) ≈ 1.0 atol=1e-10
    end

    @testset "DC: Two voltage sources" begin
        # V1 = 5V at vcc, V2 = 3V at mid
        # R1 between vcc and mid, R2 between mid and gnd
        # Current through R1: (5-3)/R1, through R2: 3/R2

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        mid = get_node!(ctx, :mid)

        stamp!(VoltageSource(5.0), ctx, vcc, 0)
        stamp!(VoltageSource(3.0), ctx, mid, 0)
        stamp!(Resistor(1000.0), ctx, vcc, mid)
        stamp!(Resistor(1000.0), ctx, mid, 0)

        sol = solve_dc(ctx)

        @test voltage(sol, :vcc) ≈ 5.0
        @test voltage(sol, :mid) ≈ 3.0

        # Check currents through sources
        I_V1 = current(sol, :I_V)  # Current through V1
        I_V2 = current(sol, Symbol("I_V"))  # Current through V2

        # Actually need to track by name...
        # For now, just verify voltages are correct
    end

    @testset "DC: VCCS amplifier" begin
        # Input: 1V source, gm = 10mS
        # Output: into 1k resistor
        # Expected: Iout = gm * Vin = 0.01 * 1 = 10mA
        # Vout = Iout * R = 0.01 * 1000 = 10V

        ctx = MNAContext()
        inp = get_node!(ctx, :inp)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(1.0), ctx, inp, 0)
        stamp!(VCCS(0.01), ctx, out, 0, inp, 0)
        stamp!(Resistor(1000.0), ctx, out, 0)

        sol = solve_dc(ctx)

        @test voltage(sol, :inp) ≈ 1.0
        @test voltage(sol, :out) ≈ 10.0 atol=1e-10
    end

    @testset "DC: Inverting amplifier with VCVS" begin
        # Simple inverting amp model with gain = -10
        # Vin = 0.5V, Vout should be -5V

        ctx = MNAContext()
        inp = get_node!(ctx, :inp)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(0.5), ctx, inp, 0)
        stamp!(VCVS(-10.0), ctx, out, 0, inp, 0)

        sol = solve_dc(ctx)

        @test voltage(sol, :inp) ≈ 0.5
        @test voltage(sol, :out) ≈ -5.0 atol=1e-10
    end

    @testset "DC: Multi-node network" begin
        # Star network: center node connected to 3 voltage sources via resistors
        # V1 = 3V (R=1k), V2 = 6V (R=2k), V3 = 9V (R=3k)
        # Center voltage by superposition:
        # Vcenter = (V1/R1 + V2/R2 + V3/R3) / (1/R1 + 1/R2 + 1/R3)
        #         = (3/1 + 6/2 + 9/3) / (1/1 + 1/2 + 1/3)
        #         = (3 + 3 + 3) / (1 + 0.5 + 0.333...)
        #         = 9 / 1.8333... ≈ 4.909V

        ctx = MNAContext()
        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)
        n3 = get_node!(ctx, :n3)
        center = get_node!(ctx, :center)

        stamp!(VoltageSource(3.0), ctx, n1, 0)
        stamp!(VoltageSource(6.0), ctx, n2, 0)
        stamp!(VoltageSource(9.0), ctx, n3, 0)

        stamp!(Resistor(1000.0), ctx, n1, center)
        stamp!(Resistor(2000.0), ctx, n2, center)
        stamp!(Resistor(3000.0), ctx, n3, center)

        sol = solve_dc(ctx)

        # Analytical: 9 / (11/6) = 54/11 ≈ 4.909
        expected = 9.0 / (1.0 + 0.5 + 1.0/3.0)
        @test voltage(sol, :center) ≈ expected atol=1e-10
    end

    #==========================================================================#
    # AC Analysis
    #==========================================================================#

    @testset "AC: RC low-pass filter" begin
        # R = 1k, C = 1uF
        # fc = 1/(2*pi*R*C) ≈ 159.15 Hz
        # At f = fc: |H| = 1/sqrt(2) ≈ 0.707 (-3dB)

        ctx = MNAContext()
        inp = get_node!(ctx, :inp)
        out = get_node!(ctx, :out)

        R = 1000.0
        C_val = 1e-6
        fc = 1.0 / (2π * R * C_val)

        stamp!(VoltageSource(1.0), ctx, inp, 0)  # 1V AC source
        stamp!(Resistor(R), ctx, inp, out)
        stamp!(Capacitor(C_val), ctx, out, 0)

        sys = assemble!(ctx)

        # Test at various frequencies
        freqs = [fc/10, fc, fc*10]
        ac_sol = solve_ac(sys, freqs)

        # At low frequency: Vout ≈ Vin
        Vout_low = abs(voltage(ac_sol, :out, 1))
        @test Vout_low > 0.99

        # At cutoff: |Vout/Vin| ≈ 0.707
        Vout_fc = abs(voltage(ac_sol, :out, 2))
        @test Vout_fc ≈ 1.0/sqrt(2) atol=0.01

        # At high frequency: Vout << Vin
        Vout_high = abs(voltage(ac_sol, :out, 3))
        @test Vout_high < 0.15
    end

    @testset "AC: RL high-pass filter" begin
        # For high-pass: R in series, L to ground
        # Vout/Vin = jωL / (R + jωL)
        # R = 1k, L = 1H
        # fc = R/(2*pi*L) ≈ 159.15 Hz

        ctx = MNAContext()
        inp = get_node!(ctx, :inp)
        out = get_node!(ctx, :out)

        R = 1000.0
        L_val = 1.0
        fc = R / (2π * L_val)

        stamp!(VoltageSource(1.0), ctx, inp, 0)
        stamp!(Resistor(R), ctx, inp, out)  # R in series
        stamp!(Inductor(L_val), ctx, out, 0)  # L to ground

        sys = assemble!(ctx)

        freqs = [fc/10, fc, fc*10]
        ac_sol = solve_ac(sys, freqs)

        # At low frequency: Vout << Vin (inductor is short)
        Vout_low = abs(voltage(ac_sol, :out, 1))
        @test Vout_low < 0.15

        # At cutoff: |Vout/Vin| ≈ 0.707
        Vout_fc = abs(voltage(ac_sol, :out, 2))
        @test Vout_fc ≈ 1.0/sqrt(2) atol=0.01

        # At high frequency: Vout ≈ Vin (inductor is open)
        Vout_high = abs(voltage(ac_sol, :out, 3))
        @test Vout_high > 0.99
    end

    #==========================================================================#
    # Transient Analysis (ODE Problem Setup)
    #==========================================================================#

    @testset "Transient: ODE problem creation" begin
        # Simple RC circuit
        ctx = MNAContext()
        inp = get_node!(ctx, :inp)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(5.0), ctx, inp, 0)
        stamp!(Resistor(1000.0), ctx, inp, out)
        stamp!(Capacitor(1e-6), ctx, out, 0)

        sys = assemble!(ctx)

        # Create ODE problem data
        prob = make_ode_problem(sys, (0.0, 1e-3))

        @test prob.tspan == (0.0, 1e-3)
        @test length(prob.u0) == system_size(sys)

        # Initial condition should be DC solution
        dc_sol = solve_dc(sys)
        @test prob.u0 ≈ dc_sol.x

        # Mass matrix should be the C matrix
        @test prob.mass_matrix == sys.C
    end

    #==========================================================================#
    # Edge Cases
    #==========================================================================#

    @testset "Edge cases" begin
        # Empty context
        ctx = MNAContext()
        sys = assemble!(ctx)
        @test system_size(sys) == 0

        # Single node (floating)
        ctx2 = MNAContext()
        n1 = get_node!(ctx2, :n1)
        stamp!(Resistor(1000.0), ctx2, n1, 0)
        sys2 = assemble!(ctx2)
        @test size(sys2.G) == (1, 1)

        # Very large resistance (should still work)
        ctx3 = MNAContext()
        n1 = get_node!(ctx3, :n1)
        stamp!(VoltageSource(1.0), ctx3, n1, 0)
        stamp!(Resistor(1e12), ctx3, n1, 0)  # 1 TΩ
        sol3 = solve_dc(ctx3)
        @test voltage(sol3, :n1) ≈ 1.0

        # Very small resistance
        ctx4 = MNAContext()
        n1 = get_node!(ctx4, :n1)
        n2 = get_node!(ctx4, :n2)
        stamp!(VoltageSource(5.0), ctx4, n1, 0)
        stamp!(Resistor(1e-6), ctx4, n1, n2)  # 1 μΩ
        stamp!(Resistor(1e-6), ctx4, n2, 0)
        sol4 = solve_dc(ctx4)
        @test voltage(sol4, :n2) ≈ 2.5 atol=1e-6
    end

    #==========================================================================#
    # Pretty Printing (no crashes)
    #==========================================================================#

    @testset "Display functions" begin
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(5.0), ctx, vcc, 0)
        stamp!(Resistor(1000.0), ctx, vcc, out)
        stamp!(Resistor(1000.0), ctx, out, 0)

        # Should not error
        io = IOBuffer()
        show(io, ctx)
        show(io, MIME"text/plain"(), ctx)

        sys = assemble!(ctx)
        show(io, sys)
        show(io, MIME"text/plain"(), sys)

        sol = solve_dc(sys)
        show(io, sol)
        show(io, MIME"text/plain"(), sol)

        # Just check no errors
        @test true
    end

end  # @testset "MNA Core Tests"
