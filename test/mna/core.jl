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
using SciMLBase: ReturnCode

# Import MNA module - use explicit imports to avoid conflicts with CedarSim types
using CedarSim.MNA: MNAContext, MNASystem, get_node!, alloc_current!
using CedarSim.MNA: stamp_G!, stamp_C!, stamp_b!, stamp_conductance!, stamp_capacitance!
using CedarSim.MNA: stamp!, system_size
using CedarSim.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using CedarSim.MNA: TimeDependentVoltageSource, PWLVoltageSource, get_source_value, pwl_value
using CedarSim.MNA: VCVS, VCCS, CCVS, CCCS
using CedarSim.MNA: assemble!, assemble_G, assemble_C, get_rhs
using CedarSim.MNA: DCSolution, ACSolution, solve_dc, solve_ac
using CedarSim.MNA: voltage, current, magnitude_db, phase_deg
using CedarSim.MNA: make_ode_problem, make_ode_function
using CedarSim.MNA: make_dae_problem, make_dae_function

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

    @testset "CCVS stamp" begin
        # CCVS (transresistance): Vout = rm * I_in
        # Needs current variables for both input sensing and output
        ctx = MNAContext()
        out_p = get_node!(ctx, :out_p)
        out_n = get_node!(ctx, :out_n)
        in_p = get_node!(ctx, :in_p)
        in_n = get_node!(ctx, :in_n)

        H = CCVS(1000.0; name=:H1)  # rm = 1kΩ
        (I_out_idx, I_in_idx) = stamp!(H, ctx, out_p, out_n, in_p, in_n)

        # I_in allocated first (index 5), I_out second (index 6)
        @test I_in_idx == 5   # First current variable (after 4 nodes)
        @test I_out_idx == 6  # Second current variable

        sys = assemble!(ctx)
        @test size(sys.G) == (6, 6)

        G = Matrix(sys.G)

        # Input sensing branch (zero-volt source) - uses I_in_idx = 5
        @test G[3, 5] ≈ 1.0   # in_p
        @test G[4, 5] ≈ -1.0  # in_n
        @test G[5, 3] ≈ 1.0   # V(in_p)
        @test G[5, 4] ≈ -1.0  # V(in_n)

        # Output branch - uses I_out_idx = 6
        @test G[1, 6] ≈ 1.0   # out_p
        @test G[2, 6] ≈ -1.0  # out_n

        # Voltage equation: V(out) = rm * I_in  - row I_out_idx = 6
        # V(out_p) - V(out_n) - rm * I_in = 0
        @test G[6, 1] ≈ 1.0     # Vout_p
        @test G[6, 2] ≈ -1.0    # Vout_n
        @test G[6, 5] ≈ -1000.0 # -rm * I_in
    end

    @testset "CCCS stamp" begin
        # CCCS: I_out = gain * I_in
        # Needs current variable for input sensing
        ctx = MNAContext()
        out_p = get_node!(ctx, :out_p)
        out_n = get_node!(ctx, :out_n)
        in_p = get_node!(ctx, :in_p)
        in_n = get_node!(ctx, :in_n)

        F = CCCS(2.0; name=:F1)  # Current gain = 2
        I_in_idx = stamp!(F, ctx, out_p, out_n, in_p, in_n)

        @test I_in_idx == 5  # After 4 nodes

        sys = assemble!(ctx)
        @test size(sys.G) == (5, 5)

        G = Matrix(sys.G)

        # Input sensing branch (zero-volt source)
        @test G[3, 5] ≈ 1.0   # in_p
        @test G[4, 5] ≈ -1.0  # in_n
        @test G[5, 3] ≈ 1.0   # V(in_p)
        @test G[5, 4] ≈ -1.0  # V(in_n)

        # Output current: gain * I_in flows into out_p (out of out_n)
        @test G[1, 5] ≈ -2.0   # out_p: -gain (current entering)
        @test G[2, 5] ≈ 2.0    # out_n: +gain (current leaving)
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

    @testset "DC: Transresistance amplifier with CCVS" begin
        # CCVS: Vout = rm * I_in
        # Current source I = 1mA through sensing branch (in_p -> in_n)
        # rm = 1000 Ω
        # Expected: Vout = 1000 * 1e-3 = 1V

        ctx = MNAContext()
        inp = get_node!(ctx, :inp)
        out = get_node!(ctx, :out)

        # Drive the sensing branch with a current source
        stamp!(CurrentSource(1e-3), ctx, inp, 0)  # 1mA into inp

        # CCVS senses current in (inp, 0) and outputs voltage at (out, 0)
        stamp!(CCVS(1000.0), ctx, out, 0, inp, 0)

        # Need a load to close the output circuit
        stamp!(Resistor(1e6), ctx, out, 0)  # High-impedance load

        sol = solve_dc(ctx)

        @test voltage(sol, :out) ≈ 1.0 atol=1e-6
    end

    @testset "DC: Current mirror with CCCS" begin
        # CCCS: I_out = gain * I_in
        # Current source I = 1mA through sensing branch
        # Gain = 2
        # Output into 1k resistor
        # Expected: I_out = 2mA, V_out = 2mA * 1kΩ = 2V

        ctx = MNAContext()
        inp = get_node!(ctx, :inp)
        out = get_node!(ctx, :out)

        # Drive the sensing branch with a current source
        stamp!(CurrentSource(1e-3), ctx, inp, 0)  # 1mA into inp

        # CCCS senses current in (inp, 0) and outputs current at (out, 0)
        stamp!(CCCS(2.0), ctx, out, 0, inp, 0)

        # Load resistor
        stamp!(Resistor(1000.0), ctx, out, 0)

        sol = solve_dc(ctx)

        @test voltage(sol, :out) ≈ 2.0 atol=1e-10
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

    #==========================================================================#
    # Transient Analysis - Actual ODE/DAE Solving
    #==========================================================================#

    @testset "Transient - RC Charging (Mass Matrix ODE)" begin
        using OrdinaryDiffEq

        # RC circuit: V -> R -> C -> GND
        #
        #  Vcc──┬──R──┬──out
        #       │     │
        #      [V]    C
        #       │     │
        #      GND   GND
        #
        # Time constant τ = R*C
        # V(t) = Vcc * (1 - exp(-t/τ))

        Vcc = 5.0
        R_val = 1000.0   # 1 kΩ
        C_val = 1e-6     # 1 μF
        τ = R_val * C_val  # 1 ms

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(Vcc), ctx, vcc, 0)
        stamp!(Resistor(R_val), ctx, vcc, out)
        stamp!(Capacitor(C_val), ctx, out, 0)

        sys = assemble!(ctx)

        # Initial condition: capacitor starts at 0V
        # The voltage source forces vcc = Vcc, so we set u0 manually
        n = system_size(sys)
        u0 = zeros(n)
        u0[1] = Vcc  # vcc node
        u0[2] = 0.0  # out node (capacitor voltage)
        # Current through voltage source will be solved

        # Solve for 5 time constants (>99% charged)
        tspan = (0.0, 5.0 * τ)

        # Get ODE problem data
        prob_data = make_ode_problem(sys, tspan; u0=u0)

        # Create ODEFunction with mass matrix
        f = ODEFunction(prob_data.f;
                        mass_matrix = prob_data.mass_matrix,
                        jac = prob_data.jac,
                        jac_prototype = prob_data.jac_prototype)
        prob = ODEProblem(f, prob_data.u0, prob_data.tspan)

        # Solve with stiff solver (Rodas5 handles mass matrices well)
        sol = solve(prob, Rodas5P(); reltol=1e-8, abstol=1e-10)

        @test sol.retcode == ReturnCode.Success

        # Analytical solution: V_out(t) = Vcc * (1 - exp(-t/τ))
        V_analytical(t) = Vcc * (1.0 - exp(-t / τ))

        # Test at multiple time points
        test_times = [0.0, τ/2, τ, 2τ, 3τ, 5τ]
        for t in test_times
            V_sim = sol(t)[2]  # out node is index 2
            V_exact = V_analytical(t)
            @test isapprox(V_sim, V_exact; rtol=1e-4) || (t, V_sim, V_exact)
        end

        # At t=0, V_out should be 0
        @test isapprox(sol(0.0)[2], 0.0; atol=1e-10)

        # At t=5τ, V_out should be ~99.3% of Vcc
        @test isapprox(sol(5τ)[2], Vcc * (1 - exp(-5)); rtol=1e-4)
    end

    @testset "Transient - RC Charging (Implicit DAE)" begin
        using Sundials
        using DiffEqBase: BrownFullBasicInit

        # Same RC circuit as above, but using DAE formulation
        Vcc = 5.0
        R_val = 1000.0
        C_val = 1e-6
        τ = R_val * C_val

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(Vcc), ctx, vcc, 0)
        stamp!(Resistor(R_val), ctx, vcc, out)
        stamp!(Capacitor(C_val), ctx, out, 0)

        sys = assemble!(ctx)

        # Initial condition: need consistent ICs for DAE
        # For RC circuit at t=0: V_cap = 0, so I = Vcc/R = 5mA
        n = system_size(sys)
        u0 = zeros(n)
        u0[1] = Vcc        # vcc = 5V
        u0[2] = 0.0        # out = 0V (cap initially uncharged)
        u0[3] = Vcc/R_val  # I_V = Vcc/R = 5mA (current through voltage source)

        tspan = (0.0, 5.0 * τ)

        # Get DAE problem data
        dae_data = make_dae_problem(sys, tspan; u0=u0)

        # Create DAEProblem
        prob = DAEProblem(dae_data.f!, dae_data.du0, dae_data.u0, dae_data.tspan;
                          differential_vars = dae_data.differential_vars)

        # Solve with IDA using Brown's initialization for consistent ICs
        sol = solve(prob, IDA(); reltol=1e-8, abstol=1e-10,
                    initializealg = BrownFullBasicInit())

        @test sol.retcode == ReturnCode.Success

        # Analytical solution
        V_analytical(t) = Vcc * (1.0 - exp(-t / τ))

        # Test at multiple time points
        test_times = [0.0, τ/2, τ, 2τ, 3τ, 5τ]
        for t in test_times
            V_sim = sol(t)[2]
            V_exact = V_analytical(t)
            @test isapprox(V_sim, V_exact; rtol=1e-3) || (t, V_sim, V_exact)
        end
    end

    @testset "Transient - RL Circuit (Inductor Current)" begin
        using OrdinaryDiffEq

        # RL circuit: V -> R -> L -> GND
        #
        # The inductor current rises as: I(t) = (V/R) * (1 - exp(-t*R/L))
        # Time constant τ = L/R

        Vcc = 10.0
        R_val = 100.0    # 100 Ω
        L_val = 0.01     # 10 mH
        τ = L_val / R_val  # 100 μs

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        mid = get_node!(ctx, :mid)

        stamp!(VoltageSource(Vcc), ctx, vcc, 0)
        stamp!(Resistor(R_val), ctx, vcc, mid)
        I_L = stamp!(Inductor(L_val), ctx, mid, 0)  # Returns current index

        sys = assemble!(ctx)

        # Initial condition: inductor starts with 0 current
        n = system_size(sys)
        u0 = zeros(n)
        u0[1] = Vcc  # vcc
        u0[2] = Vcc  # mid (initially at Vcc since no current flows)
        # Current variables start at 0

        tspan = (0.0, 5.0 * τ)

        prob_data = make_ode_problem(sys, tspan; u0=u0)

        f = ODEFunction(prob_data.f;
                        mass_matrix = prob_data.mass_matrix,
                        jac = prob_data.jac,
                        jac_prototype = prob_data.jac_prototype)
        prob = ODEProblem(f, prob_data.u0, prob_data.tspan)

        sol = solve(prob, Rodas5P(); reltol=1e-8, abstol=1e-10)

        @test sol.retcode == ReturnCode.Success

        # Steady state current: I_ss = V/R
        I_ss = Vcc / R_val

        # Analytical: I(t) = I_ss * (1 - exp(-t/τ))
        I_analytical(t) = I_ss * (1.0 - exp(-t / τ))

        # Test current at various times
        # The inductor current is at index I_L in the solution
        for t in [0.0, τ/2, τ, 2τ, 5τ]
            I_sim = sol(t)[I_L]
            I_exact = I_analytical(t)
            @test isapprox(I_sim, I_exact; rtol=1e-3) || (t, I_sim, I_exact)
        end

        # At t=0, current should be 0
        @test isapprox(sol(0.0)[I_L], 0.0; atol=1e-8)

        # At t=5τ, current should be ~99.3% of I_ss
        @test isapprox(sol(5τ)[I_L], I_ss * (1 - exp(-5)); rtol=1e-3)
    end

    @testset "Transient - RLC Oscillator" begin
        using OrdinaryDiffEq

        # Underdamped RLC circuit
        # V -> R -> L ─┬─ C -> GND
        #              │
        #             out
        #
        # For underdamped: R < 2*sqrt(L/C)
        # Natural frequency: ω₀ = 1/sqrt(LC)
        # Damping factor: α = R/(2L)
        # Damped frequency: ωd = sqrt(ω₀² - α²)

        Vcc = 5.0
        R_val = 10.0      # 10 Ω (small for underdamping)
        L_val = 0.001     # 1 mH
        C_val = 1e-6      # 1 μF

        ω0 = 1.0 / sqrt(L_val * C_val)  # ~31.6 krad/s
        α = R_val / (2 * L_val)          # 5000 rad/s
        ωd = sqrt(ω0^2 - α^2)            # Damped frequency

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        mid = get_node!(ctx, :mid)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(Vcc), ctx, vcc, 0)
        stamp!(Resistor(R_val), ctx, vcc, mid)
        I_L = stamp!(Inductor(L_val), ctx, mid, out)
        stamp!(Capacitor(C_val), ctx, out, 0)

        sys = assemble!(ctx)

        # Initial condition: everything at 0 except voltage source
        n = system_size(sys)
        u0 = zeros(n)
        u0[1] = Vcc  # vcc

        # Simulate for several oscillation periods
        period = 2π / ωd
        tspan = (0.0, 5.0 * period)

        prob_data = make_ode_problem(sys, tspan; u0=u0)

        f = ODEFunction(prob_data.f;
                        mass_matrix = prob_data.mass_matrix,
                        jac = prob_data.jac,
                        jac_prototype = prob_data.jac_prototype)
        prob = ODEProblem(f, prob_data.u0, prob_data.tspan)

        sol = solve(prob, Rodas5P(); reltol=1e-8, abstol=1e-10)

        @test sol.retcode == ReturnCode.Success

        # The capacitor voltage should oscillate and eventually settle to Vcc
        # At late times, it should approach Vcc
        V_final = sol(tspan[2])[3]  # out node
        @test isapprox(V_final, Vcc; rtol=0.01)

        # Check that oscillation occurred by looking for overshoot
        # In an underdamped system, the voltage should exceed Vcc at some point
        times = range(0, tspan[2], length=1000)
        V_out = [sol(t)[3] for t in times]
        @test maximum(V_out) > Vcc * 1.01  # Should overshoot by at least 1%
    end

    @testset "Transient - Comparison ODE vs DAE" begin
        using OrdinaryDiffEq
        using Sundials
        using DiffEqBase: BrownFullBasicInit

        # Solve the same RC circuit with both methods and compare
        Vcc = 3.3
        R_val = 2200.0
        C_val = 100e-9
        τ = R_val * C_val

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(Vcc), ctx, vcc, 0)
        stamp!(Resistor(R_val), ctx, vcc, out)
        stamp!(Capacitor(C_val), ctx, out, 0)

        sys = assemble!(ctx)

        # Consistent initial conditions for DAE
        n = system_size(sys)
        u0 = zeros(n)
        u0[1] = Vcc        # vcc = 3.3V
        u0[2] = 0.0        # out = 0V
        u0[3] = Vcc/R_val  # I_V = Vcc/R
        tspan = (0.0, 10.0 * τ)

        # Solve with mass matrix ODE
        prob_ode = make_ode_problem(sys, tspan; u0=u0)
        f_ode = ODEFunction(prob_ode.f;
                            mass_matrix = prob_ode.mass_matrix,
                            jac = prob_ode.jac,
                            jac_prototype = prob_ode.jac_prototype)
        ode_prob = ODEProblem(f_ode, prob_ode.u0, prob_ode.tspan)
        sol_ode = solve(ode_prob, Rodas5P(); reltol=1e-10, abstol=1e-12)

        # Solve with DAE using Brown's initialization
        dae_data = make_dae_problem(sys, tspan; u0=u0)
        dae_prob = DAEProblem(dae_data.f!, dae_data.du0, dae_data.u0, dae_data.tspan;
                              differential_vars = dae_data.differential_vars)
        sol_dae = solve(dae_prob, IDA(); reltol=1e-10, abstol=1e-12,
                        initializealg = BrownFullBasicInit())

        @test sol_ode.retcode == ReturnCode.Success
        @test sol_dae.retcode == ReturnCode.Success

        # Solutions should match within tolerance
        test_times = range(0, tspan[2], length=20)
        for t in test_times
            V_ode = sol_ode(t)[2]
            V_dae = sol_dae(t)[2]
            @test isapprox(V_ode, V_dae; rtol=1e-4) || (t, V_ode, V_dae)
        end
    end

    #==========================================================================#
    # MNASim: Parameterized Circuit Wrapper
    #==========================================================================#

    # Import MNASim and related exports
    using CedarSim.MNA: MNASim, MNASpec, alter, with_mode, with_spec, with_temp, eval_circuit
    using CedarSim.MNA: MNACircuit
    using SciMLBase: ODEProblem as SciMLODEProblem

    @testset "MNASim basics" begin
        # Define a parameterized circuit builder (new API: params, spec)
        function build_voltage_divider(params, spec)
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            stamp!(Resistor(params.R1), ctx, vcc, out)
            stamp!(Resistor(params.R2), ctx, out, 0)

            return ctx
        end

        # Create sim with parameters
        sim = MNASim(build_voltage_divider; Vcc=10.0, R1=1000.0, R2=1000.0)

        @test sim.spec.mode == :tran
        @test sim.params.Vcc == 10.0
        @test sim.params.R1 == 1000.0

        # Build and solve
        sol = solve_dc(sim)
        @test voltage(sol, :out) ≈ 5.0  # Voltage divider: Vcc * R2/(R1+R2)

        # Alter parameters
        sim2 = alter(sim; R2=3000.0)
        @test sim2.params.R2 == 3000.0
        @test sim2.params.R1 == 1000.0  # Unchanged

        sol2 = solve_dc(sim2)
        @test voltage(sol2, :out) ≈ 7.5  # 10 * 3000/4000

        # Change mode
        sim3 = with_mode(sim, :dcop)
        @test sim3.spec.mode == :dcop
        @test sim3.params.Vcc == 10.0  # Params preserved

        # Test out-of-place eval_circuit
        sys = eval_circuit(sim)
        @test sys isa MNASystem
    end

    @testset "MNASim with RC circuit" begin
        function build_rc(params, spec)
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        sim = MNASim(build_rc; Vcc=5.0, R=1000.0, C=1e-6)

        # DC solution (capacitor is open circuit)
        sol = solve_dc(sim)
        @test voltage(sol, :out) ≈ 5.0  # At DC, capacitor is open

        # AC sweep - cutoff fc = 1/(2πRC) ≈ 159Hz for R=1kΩ, C=1μF
        # At 10Hz (well below cutoff), gain should be ~1
        freqs = [10.0, 100.0, 1000.0]
        ac_sol = solve_ac(sim, freqs)

        # At 10Hz (f << fc), gain ≈ 1 (within 1%)
        @test abs(voltage(ac_sol, :out)[1]) > 0.99 * sim.params.Vcc
        # At 1000Hz (f >> fc), gain should be < 0.2 (rolloff)
        @test abs(voltage(ac_sol, :out)[3]) < 0.2 * sim.params.Vcc
    end

    @testset "DC-initialized transient (ODE)" begin
        using OrdinaryDiffEq

        # Build RC circuit and use DC-initialized transient
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        Vcc = 5.0
        R = 1000.0
        C = 1e-6
        τ = R * C

        stamp!(VoltageSource(Vcc), ctx, vcc, 0)
        stamp!(Resistor(R), ctx, vcc, out)
        stamp!(Capacitor(C), ctx, out, 0)

        sys = assemble!(ctx)
        tspan = (0.0, 5τ)

        # Use make_ode_problem which does DC initialization by default
        ode_data = make_ode_problem(sys, tspan)

        # DC initialization means V_out starts at Vcc (steady state)
        @test ode_data.u0[2] ≈ Vcc

        f = ODEFunction(ode_data.f;
                        mass_matrix = ode_data.mass_matrix,
                        jac = ode_data.jac,
                        jac_prototype = ode_data.jac_prototype)
        prob = ODEProblem(f, ode_data.u0, ode_data.tspan)
        sol = solve(prob, Rodas5P(); reltol=1e-8, abstol=1e-10)

        @test sol.retcode == ReturnCode.Success

        # Starting from steady state, should stay at Vcc
        @test sol(0.0)[2] ≈ Vcc
        @test sol(5τ)[2] ≈ Vcc
    end

    @testset "DC-initialized transient (DAE)" begin
        using Sundials
        using DiffEqBase: BrownFullBasicInit

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        Vcc = 3.3
        R = 2200.0
        C = 100e-9
        τ = R * C

        stamp!(VoltageSource(Vcc), ctx, vcc, 0)
        stamp!(Resistor(R), ctx, vcc, out)
        stamp!(Capacitor(C), ctx, out, 0)

        sys = assemble!(ctx)
        tspan = (0.0, 5τ)

        # Use make_dae_problem which does DC initialization by default
        dae_data = make_dae_problem(sys, tspan)

        # DC initialization means consistent ICs
        @test dae_data.u0[1] ≈ Vcc  # vcc
        @test dae_data.u0[2] ≈ Vcc  # out (DC steady state)

        prob = DAEProblem(dae_data.f!, dae_data.du0, dae_data.u0, dae_data.tspan;
                          differential_vars = dae_data.differential_vars)
        sol = solve(prob, IDA(); reltol=1e-8, abstol=1e-10,
                    initializealg = BrownFullBasicInit())

        @test sol.retcode == ReturnCode.Success

        # Starting from steady state, should stay there
        @test sol(0.0)[2] ≈ Vcc rtol=1e-3
        @test sol(5τ)[2] ≈ Vcc rtol=1e-3
    end

    @testset "Parameterized circuit with mode switching" begin
        using OrdinaryDiffEq
        using DiffEqBase: BrownFullBasicInit

        # Build a parameterized RC circuit with a step voltage source
        # In :dcop mode, source returns dc_value
        # In :tran mode, source returns time-dependent value
        function build_rc_step(params, spec)
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            # Time-dependent voltage: step from 0 to Vcc at t=0
            # In :dcop mode, use 0V (pre-step state) or Vcc (post-step state)
            if spec.mode == :dcop
                # DC operating point at t=0+ (after step)
                stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            else
                # For transient, use full voltage (step already occurred at t=0)
                stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            end

            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        # Create parameterized simulation
        sim = MNASim(build_rc_step; Vcc=5.0, R=1000.0, C=1e-6)

        # Test 1: DC analysis with default mode (:tran)
        sol_dc = solve_dc(sim)
        @test voltage(sol_dc, :vcc) ≈ 5.0
        @test voltage(sol_dc, :out) ≈ 5.0  # steady state

        # Test 2: Change parameters using alter()
        sim_modified = alter(sim; R=2000.0, C=2e-6)
        sol_mod = solve_dc(sim_modified)
        @test voltage(sol_mod, :vcc) ≈ 5.0
        @test voltage(sol_mod, :out) ≈ 5.0  # same DC, different dynamics

        # Verify time constant changed: τ = R*C
        τ_original = 1000.0 * 1e-6   # 1ms
        τ_modified = 2000.0 * 2e-6   # 4ms
        @test τ_modified == 4 * τ_original

        # Test 3: Change Vcc parameter and verify DC changes
        sim_highv = alter(sim; Vcc=10.0)
        sol_highv = solve_dc(sim_highv)
        @test voltage(sol_highv, :out) ≈ 10.0

        # Test 4: Mode switching with with_mode()
        sim_dcop = with_mode(sim, :dcop)
        @test sim_dcop.spec.mode == :dcop
        sol_dcop = solve_dc(sim_dcop)
        @test voltage(sol_dcop, :out) ≈ 5.0

        # Test 5: Transient simulation with different parameters
        # Compare τ=1ms circuit vs τ=4ms circuit
        function run_transient(sim_instance)
            sys = assemble!(sim_instance)
            R = sim_instance.params.R
            C = sim_instance.params.C
            τ = R * C
            tspan = (0.0, 5τ)

            ode_data = make_ode_problem(sys, tspan)  # DC-initialized by default

            M = ode_data.mass_matrix
            f = ode_data.f
            u0 = ode_data.u0
            jac = ode_data.jac
            jac_proto = ode_data.jac_prototype

            ode_fn = ODEFunction(f; mass_matrix=M, jac=jac, jac_prototype=jac_proto)
            prob = ODEProblem(ode_fn, u0, tspan)
            sol = solve(prob, Rodas5P(); reltol=1e-8, abstol=1e-10)

            return sol, τ
        end

        sol_fast, τ_fast = run_transient(sim)  # τ=1ms
        sol_slow, τ_slow = run_transient(sim_modified)  # τ=4ms

        # Both should start at steady state (DC-initialized)
        @test sol_fast(0.0)[2] ≈ 5.0 rtol=1e-3
        @test sol_slow(0.0)[2] ≈ 5.0 rtol=1e-3

        # Both should end at steady state
        @test sol_fast(5*τ_fast)[2] ≈ 5.0 rtol=1e-3
        @test sol_slow(5*τ_slow)[2] ≈ 5.0 rtol=1e-3

        # Test 6: Chain multiple alterations
        sim_chain = alter(alter(sim; Vcc=3.3); R=4700.0)
        sol_chain = solve_dc(sim_chain)
        @test voltage(sol_chain, :vcc) ≈ 3.3
        @test voltage(sol_chain, :out) ≈ 3.3
        @test sim_chain.params.R == 4700.0
        @test sim_chain.params.Vcc == 3.3
        @test sim_chain.params.C == 1e-6  # unchanged from original
    end

    @testset "Time-dependent source with mode" begin
        using OrdinaryDiffEq

        # Test TimeDependentVoltageSource behavior
        pulse_src = TimeDependentVoltageSource(
            t -> t < 1e-3 ? 0.0 : 5.0;  # Step at t=1ms
            dc_value = 2.5,  # DC value for :dcop mode
            name = :Vpulse
        )

        # In :dcop mode, should return dc_value
        @test get_source_value(pulse_src, 0.0, :dcop) == 2.5
        @test get_source_value(pulse_src, 1e-3, :dcop) == 2.5
        @test get_source_value(pulse_src, 5e-3, :dcop) == 2.5

        # In :tran mode, should return time-dependent value
        @test get_source_value(pulse_src, 0.0, :tran) == 0.0
        @test get_source_value(pulse_src, 0.5e-3, :tran) == 0.0
        @test get_source_value(pulse_src, 1.5e-3, :tran) == 5.0

        # Test PWLVoltageSource behavior
        pwl_src = PWLVoltageSource(
            [0.0, 1e-3, 2e-3],  # times
            [0.0, 5.0, 5.0];    # values (ramp then hold)
            name = :Vramp
        )

        @test pwl_value(pwl_src, 0.0) == 0.0
        @test pwl_value(pwl_src, 0.5e-3) ≈ 2.5  # midpoint of ramp
        @test pwl_value(pwl_src, 1e-3) == 5.0
        @test pwl_value(pwl_src, 1.5e-3) == 5.0  # hold
        @test pwl_value(pwl_src, 5e-3) == 5.0   # after last point
    end

    @testset "Mode-aware parameterized simulation" begin
        using OrdinaryDiffEq

        # Build RC circuit with time-dependent source that respects mode
        function build_mode_aware_rc(params, spec)
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            # Use mode to determine source value
            # :dcop -> use steady-state value (params.Vdc)
            # :tranop -> use t=0 transient value
            # :tran -> (for stamping, use DC value; time-dependence handled in ODE)
            source_voltage = if spec.mode == :dcop
                params.Vdc  # DC operating point value
            else
                params.Vss  # Steady-state transient value
            end

            stamp!(VoltageSource(source_voltage), ctx, vcc, 0)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        # Create sim with both DC and steady-state parameters
        sim = MNASim(build_mode_aware_rc; Vdc=0.0, Vss=5.0, R=1000.0, C=1e-6)

        # Test DC mode gives Vdc
        sim_dcop = with_mode(sim, :dcop)
        sol_dcop = solve_dc(sim_dcop)
        @test voltage(sol_dcop, :vcc) ≈ 0.0
        @test voltage(sol_dcop, :out) ≈ 0.0

        # Test transient mode gives Vss
        sim_tran = with_mode(sim, :tran)
        sol_tran = solve_dc(sim_tran)
        @test voltage(sol_tran, :vcc) ≈ 5.0
        @test voltage(sol_tran, :out) ≈ 5.0

        # Run transient from DC operating point
        # This simulates the typical SPICE flow:
        # 1. Compute DC operating point (with sources at DC values)
        # 2. Run transient (with sources at transient values)

        # First, get DC operating point (cap starts at 0V)
        dcop_sys = assemble!(sim_dcop)
        dc_sol = solve_dc(dcop_sys)
        V_cap_dc = voltage(dc_sol, :out)
        @test V_cap_dc ≈ 0.0  # Cap at 0V in DC mode

        # Now run transient with source at 5V, cap starting at 0V
        tran_sys = assemble!(sim_tran)
        τ = 1000.0 * 1e-6  # 1ms
        tspan = (0.0, 5τ)

        # Create ODE problem with initial condition from DC
        ode_data = make_ode_problem(tran_sys, tspan)

        # Override u0 with DC solution (cap at 0V)
        u0 = copy(ode_data.u0)
        u0[1] = 5.0  # vcc at Vss
        u0[2] = 0.0  # out starts at DC value (0V)
        u0[3] = 5.0 / 1000.0  # initial current from DC

        M = ode_data.mass_matrix
        f! = ode_data.f
        jac! = ode_data.jac
        jac_proto = ode_data.jac_prototype

        ode_fn = ODEFunction(f!; mass_matrix=M, jac=jac!, jac_prototype=jac_proto)
        prob = ODEProblem(ode_fn, u0, tspan)
        sol = solve(prob, Rodas5P(); reltol=1e-8, abstol=1e-10)

        @test sol.retcode == ReturnCode.Success

        # Should see exponential charging from 0V to 5V
        @test sol(0.0)[2] ≈ 0.0 rtol=1e-3
        @test sol(τ)[2] ≈ 5.0 * (1 - exp(-1)) rtol=1e-2  # ~3.16V at t=τ
        @test sol(5τ)[2] ≈ 5.0 rtol=1e-2  # ~5V at t=5τ
    end

    @testset "ParamLens with unmodified parameters" begin
        # Import CedarSim's actual ParamLens infrastructure
        using CedarSim: ParamLens, IdentityLens

        # Test 1: IdentityLens returns defaults unchanged
        ident = IdentityLens()
        defaults = ident(; R=1000.0, C=1e-6, Vcc=5.0)
        @test defaults.R == 1000.0
        @test defaults.C == 1e-6
        @test defaults.Vcc == 5.0

        # Test 2: ParamLens with no overrides (empty) acts like IdentityLens
        empty_lens = ParamLens()
        defaults2 = empty_lens(; R=1000.0, C=1e-6)
        @test defaults2.R == 1000.0
        @test defaults2.C == 1e-6

        # Test 3: ParamLens with partial overrides - unmodified params use defaults
        # This is the key test: only override R, leave C at default
        partial_lens = ParamLens((params=(R=2000.0,),))
        merged = partial_lens(; R=1000.0, C=1e-6)
        @test merged.R == 2000.0  # Overridden
        @test merged.C == 1e-6    # Uses default (unmodified)

        # Test 4: Use ParamLens in a circuit builder
        function build_with_lens(lens, spec)
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            # lens(; defaults...) returns params with lens overrides merged
            p = lens(; Vcc=5.0, R1=1000.0, R2=1000.0)

            stamp!(VoltageSource(p.Vcc), ctx, vcc, 0)
            stamp!(Resistor(p.R1), ctx, vcc, out)
            stamp!(Resistor(p.R2), ctx, out, 0)

            return ctx
        end

        # With identity lens: all defaults
        ctx_default = build_with_lens(IdentityLens(), MNASpec())
        sys_default = assemble!(ctx_default)
        sol_default = solve_dc(sys_default)
        @test voltage(sol_default, :out) ≈ 2.5  # 5V * 1k/(1k+1k)

        # With partial override: only R1 changed
        override_lens = ParamLens((params=(R1=3000.0,),))
        ctx_override = build_with_lens(override_lens, MNASpec())
        sys_override = assemble!(ctx_override)
        sol_override = solve_dc(sys_override)
        # Vout = 5V * R2/(R1+R2) = 5 * 1000/4000 = 1.25V
        @test voltage(sol_override, :out) ≈ 1.25 rtol=1e-6

        # Verify R2 and Vcc still at defaults (unmodified)
        # We can check by computing what voltage would be with different values
        # If R2=2000: Vout = 5 * 2000/5000 = 2.0V (not what we see)
        # So R2 must be at default 1000
        @test voltage(sol_override, :vcc) ≈ 5.0  # Vcc at default

        # Test 5: Hierarchical lens traversal
        # lens.subcircuit returns a new lens for that subcircuit
        hier_lens = ParamLens((sub1=(params=(R=500.0,),),))
        sub_lens = getproperty(hier_lens, :sub1)
        sub_params = sub_lens(; R=1000.0, C=1e-6)
        @test sub_params.R == 500.0  # Override from sub1
        @test sub_params.C == 1e-6   # Default (unmodified)

        # Accessing undefined subcircuit returns IdentityLens
        other_lens = getproperty(hier_lens, :sub2)
        @test other_lens isa IdentityLens
        other_params = other_lens(; R=1000.0)
        @test other_params.R == 1000.0  # All defaults
    end

    @testset "MNASpec and temperature" begin
        # Test MNASpec creation and manipulation
        spec = MNASpec()
        @test spec.temp == 27.0
        @test spec.mode == :tran

        spec2 = MNASpec(temp=50.0, mode=:dcop)
        @test spec2.temp == 50.0
        @test spec2.mode == :dcop

        # Test with_temp and with_mode on MNASpec
        spec3 = with_temp(spec, 100.0)
        @test spec3.temp == 100.0
        @test spec3.mode == :tran  # unchanged

        spec4 = with_mode(spec, :ac)
        @test spec4.temp == 27.0  # unchanged
        @test spec4.mode == :ac

        # Test temperature-dependent circuit
        function build_temp_dependent(params, spec)
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            # Temperature-dependent resistance: R(T) = R0 * (1 + tc*(T-Tnom))
            R_temp = params.R0 * (1 + params.tc * (spec.temp - 27.0))

            stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            stamp!(Resistor(R_temp), ctx, vcc, out)
            stamp!(Resistor(params.R2), ctx, out, 0)

            return ctx
        end

        # At 27°C (nominal), R_temp = R0
        sim_27 = MNASim(build_temp_dependent;
                        spec=MNASpec(temp=27.0),
                        Vcc=10.0, R0=1000.0, tc=0.004, R2=1000.0)
        sol_27 = solve_dc(sim_27)
        @test voltage(sol_27, :out) ≈ 5.0  # Equal resistors

        # At 127°C, R_temp = 1000 * (1 + 0.004*100) = 1400 Ω
        sim_127 = with_temp(sim_27, 127.0)
        @test sim_127.spec.temp == 127.0
        sol_127 = solve_dc(sim_127)
        # Voltage divider: Vout = Vcc * R2/(R_temp + R2) = 10 * 1000/2400 ≈ 4.167V
        @test voltage(sol_127, :out) ≈ 10.0 * 1000.0 / 2400.0 rtol=1e-6

        # At -73°C, R_temp = 1000 * (1 + 0.004*(-100)) = 600 Ω
        sim_m73 = with_temp(sim_27, -73.0)
        sol_m73 = solve_dc(sim_m73)
        # Voltage divider: Vout = 10 * 1000/1600 = 6.25V
        @test voltage(sol_m73, :out) ≈ 10.0 * 1000.0 / 1600.0 rtol=1e-6

        # Test with_spec
        new_spec = MNASpec(temp=85.0, mode=:dcop)
        sim_85 = with_spec(sim_27, new_spec)
        @test sim_85.spec.temp == 85.0
        @test sim_85.spec.mode == :dcop

        # Test eval_circuit directly
        sys = eval_circuit(build_temp_dependent,
                          (Vcc=10.0, R0=1000.0, tc=0.004, R2=1000.0),
                          MNASpec(temp=27.0))
        @test sys isa MNASystem
        @test sys.n_nodes == 2
    end

    #==========================================================================#
    # High-Level dc!/tran! API (integration with CedarSim sweep API)
    #==========================================================================#

    # dc!/tran! are exported from CedarSim (via sweeps.jl), not MNA module
    using CedarSim: dc!, tran!
    using CedarSim.MNA: MNASolutionAccessor, scope, NodeRef, ScopedSystem

    @testset "dc! and tran! API" begin
        # Define a simple RC circuit
        function build_rc_simple(params, spec)
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        sim = MNASim(build_rc_simple; Vcc=5.0, R=1000.0, C=1e-6)
        τ = 1000.0 * 1e-6  # 1ms

        # Test dc!(sim) - matches CedarSim API
        dc_sol = dc!(sim)
        @test dc_sol isa DCSolution
        @test voltage(dc_sol, :out) ≈ 5.0  # DC steady state

        # Test tran!(sim, tspan) - matches CedarSim API
        ode_sol = tran!(sim, (0.0, 5τ))
        @test ode_sol.retcode == ReturnCode.Success

        # Wrap for symbolic access
        sys = assemble!(sim)
        acc = MNASolutionAccessor(ode_sol, sys)

        # Test voltage access by name
        v_out_final = voltage(acc, :out, 5τ)
        @test v_out_final ≈ 5.0 rtol=1e-2  # Should stay at DC steady state

        # Test voltage trajectory
        v_trajectory = voltage(acc, :out)
        @test length(v_trajectory) == length(acc.t)

        # Test acc[:name] syntax
        v_via_getindex = acc[:out]
        @test v_via_getindex == v_trajectory

        # Test acc.t access
        @test acc.t[1] == 0.0
        @test acc.t[end] ≈ 5τ rtol=1e-6

        # Test acc(t) interpolation
        state_at_tau = acc(τ)
        @test length(state_at_tau) == 3  # vcc, out, I_V
    end

    @testset "Hierarchical scope access" begin
        # Build a circuit with hierarchical-like node names
        function build_hierarchical(params, spec)
            ctx = MNAContext()
            # Simulate subcircuit x1 with nodes x1_in and x1_out
            x1_in = get_node!(ctx, :x1_in)
            x1_out = get_node!(ctx, :x1_out)
            vcc = get_node!(ctx, :vcc)

            stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            stamp!(Resistor(params.R1), ctx, vcc, x1_in)
            stamp!(Resistor(params.R2), ctx, x1_in, x1_out)
            stamp!(Resistor(params.R3), ctx, x1_out, 0)

            return ctx
        end

        sim = MNASim(build_hierarchical; Vcc=10.0, R1=1000.0, R2=1000.0, R3=1000.0)
        sys = assemble!(sim)

        # Test ScopedSystem
        s = scope(sys)
        @test s isa ScopedSystem

        # Access hierarchical node
        node_ref = s.x1.out
        @test node_ref isa NodeRef
        @test node_ref.name == :out
        @test node_ref.path == [:x1]

        # Test with tran!
        ode_sol = tran!(sim, (0.0, 1e-3))
        acc = MNASolutionAccessor(ode_sol, sys)

        # Access via NodeRef
        v_x1_out = acc[s.x1.out]
        @test length(v_x1_out) == length(acc.t)

        # Should equal direct access
        v_direct = acc[:x1_out]
        @test v_x1_out == v_direct

        # Test voltage with NodeRef
        v_at_t = voltage(acc, s.x1.out, 0.0)
        @test v_at_t isa Float64
    end

    #==========================================================================#
    # PWL and SIN Time-Dependent Sources
    #==========================================================================#

    using CedarSim.MNA: SinVoltageSource, SinCurrentSource
    using CedarSim.MNA: PWLCurrentSource, sin_value

    @testset "SinVoltageSource evaluation" begin
        # SIN(vo, va, freq, [td, theta, phase])
        # V(t) = vo + va * sin(2π*freq*t + phase)  (simplified, no damping)

        # Simple sine: 1V offset, 2V amplitude, 1kHz
        sin_src = SinVoltageSource(1.0, 2.0, 1000.0; name=:Vsin)

        # At t=0 (phase=0), V = vo + va*sin(0) = 1 + 0 = 1
        @test sin_value(sin_src, 0.0) ≈ 1.0

        # At t=0.25ms (quarter period), V = 1 + 2*sin(90°) = 3
        @test sin_value(sin_src, 0.25e-3) ≈ 3.0 atol=1e-10

        # At t=0.5ms (half period), V = 1 + 2*sin(180°) = 1
        @test sin_value(sin_src, 0.5e-3) ≈ 1.0 atol=1e-10

        # At t=0.75ms (3/4 period), V = 1 + 2*sin(270°) = -1
        @test sin_value(sin_src, 0.75e-3) ≈ -1.0 atol=1e-10

        # Test with delay
        sin_delayed = SinVoltageSource(0.0, 1.0, 1000.0; td=1e-3, name=:Vdelay)
        # Before delay: V = vo + va*sin(phase) = 0
        @test sin_value(sin_delayed, 0.0) ≈ 0.0
        @test sin_value(sin_delayed, 0.5e-3) ≈ 0.0
        # After delay: starts oscillating
        @test sin_value(sin_delayed, 1.25e-3) ≈ 1.0 atol=1e-10  # peak at t=1.25ms

        # Test with phase
        sin_phase = SinVoltageSource(0.0, 1.0, 1000.0; phase=90.0, name=:Vphase)
        # At t=0, V = sin(90°) = 1
        @test sin_value(sin_phase, 0.0) ≈ 1.0 atol=1e-10

        # Test DC mode
        @test get_source_value(sin_src, 0.0, :dcop) ≈ 1.0  # vo + va*sin(phase) = 1 + 0
        @test get_source_value(sin_src, 0.5e-3, :dcop) ≈ 1.0  # Same DC regardless of t
    end

    @testset "PWLVoltageSource evaluation" begin
        # Ramp from 0 to 5V over 1ms, then hold
        pwl = PWLVoltageSource([0.0, 1e-3, 2e-3], [0.0, 5.0, 5.0]; name=:Vramp)

        @test pwl_value(pwl, 0.0) ≈ 0.0
        @test pwl_value(pwl, 0.5e-3) ≈ 2.5  # midpoint
        @test pwl_value(pwl, 1e-3) ≈ 5.0
        @test pwl_value(pwl, 1.5e-3) ≈ 5.0  # hold
        @test pwl_value(pwl, 5e-3) ≈ 5.0    # after last point

        # Before first point
        @test pwl_value(pwl, -1e-3) ≈ 0.0

        # DC mode: use value at t=0
        @test get_source_value(pwl, 0.0, :dcop) ≈ 0.0
        @test get_source_value(pwl, 1e-3, :dcop) ≈ 0.0  # t=0 value regardless of t

        # Tran mode: use actual value
        @test get_source_value(pwl, 0.0, :tran) ≈ 0.0
        @test get_source_value(pwl, 1e-3, :tran) ≈ 5.0
    end

    @testset "PWL/SIN stamp! methods" begin
        using CedarSim.MNA: MNASpec

        # Test PWL voltage source stamping
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)

        pwl = PWLVoltageSource([0.0, 1e-3], [0.0, 5.0]; name=:Vpwl)
        stamp!(pwl, ctx, vcc, 0; t=0.5e-3, mode=:tran)

        sys = assemble!(ctx)
        sol = solve_dc(sys)

        # At t=0.5ms, PWL value = 2.5V
        @test voltage(sol, :vcc) ≈ 2.5 atol=1e-10

        # Test SIN voltage source stamping
        ctx2 = MNAContext()
        vcc2 = get_node!(ctx2, :vcc)

        sin_src = SinVoltageSource(0.0, 5.0, 1000.0; name=:Vsin)
        # At t=0.25ms, sin = 5*sin(90°) = 5
        stamp!(sin_src, ctx2, vcc2, 0; t=0.25e-3, mode=:tran)

        sys2 = assemble!(ctx2)
        sol2 = solve_dc(sys2)
        @test voltage(sol2, :vcc) ≈ 5.0 atol=1e-10
    end

    @testset "Time-dependent ODE with builder" begin
        using OrdinaryDiffEq

        # Build RC circuit with PWL voltage source
        function build_pwl_rc(params, spec; x=Float64[])
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            # PWL: ramp from 0 to Vmax over ramp_time, then hold
            pwl = PWLVoltageSource(
                [0.0, params.ramp_time],
                [0.0, params.Vmax];
                name=:Vpwl
            )
            stamp!(pwl, ctx, vcc, 0; t=spec.time, mode=spec.mode)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        params = (Vmax=5.0, ramp_time=1e-3, R=1000.0, C=1e-6)
        τ = params.R * params.C  # 1ms

        # Create ODE problem using MNACircuit (time-dependent sources handled automatically)
        circuit = MNACircuit(build_pwl_rc, params, MNASpec(temp=27.0), (0.0, 5e-3))
        prob = SciMLODEProblem(circuit)

        sol = solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-8)

        @test sol.retcode == ReturnCode.Success

        # At t=0, source is 0V
        @test sol(0.0)[1] ≈ 0.0 atol=1e-6  # vcc

        # At t=5ms (after ramp), source should be at 5V
        # Capacitor should be nearly charged
        @test sol(5e-3)[1] ≈ 5.0 atol=0.1  # vcc
        @test sol(5e-3)[2] > 4.0  # out should be approaching 5V
    end

    @testset "SIN transient simulation" begin
        using OrdinaryDiffEq

        # RC circuit with sinusoidal source
        function build_sin_rc(params, spec; x=Float64[])
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            sin_src = SinVoltageSource(
                params.vo, params.va, params.freq;
                name=:Vsin
            )
            stamp!(sin_src, ctx, vcc, 0; t=spec.time, mode=spec.mode)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        # Low-pass RC filter with SIN source
        # R = 1k, C = 1μF, fc = 159Hz
        # Source: 1kHz (well above cutoff) - should be attenuated
        params = (vo=0.0, va=5.0, freq=1000.0, R=1000.0, C=1e-6)
        fc = 1.0 / (2π * params.R * params.C)  # ~159 Hz

        # Create ODE problem using MNACircuit
        circuit = MNACircuit(build_sin_rc, params, MNASpec(temp=27.0), (0.0, 5e-3))
        prob = SciMLODEProblem(circuit)

        sol = solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-8)

        @test sol.retcode == ReturnCode.Success

        # The output amplitude should be attenuated since freq >> fc
        # At steady state, |H(jω)| = 1/sqrt(1 + (f/fc)²) ≈ 0.157 for f=1000Hz, fc=159Hz
        # So output amplitude should be about 5 * 0.157 ≈ 0.79V

        # After initial transient (a few ms), check output amplitude
        times = 3e-3:0.01e-3:5e-3
        out_vals = [sol(t)[2] for t in times]
        amplitude = (maximum(out_vals) - minimum(out_vals)) / 2

        # Output should be attenuated (less than input amplitude of 5V)
        @test amplitude < 2.0
        @test amplitude > 0.3
    end

end  # @testset "MNA Core Tests"
