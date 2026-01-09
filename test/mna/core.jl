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
using CedarSim.MNA: MNAContext, MNASystem, get_node!, alloc_current!, resolve_index
using CedarSim.MNA: alloc_internal_node!, is_internal_node, n_internal_nodes
using CedarSim.MNA: stamp_G!, stamp_C!, stamp_b!, stamp_conductance!, stamp_capacitance!
using CedarSim.MNA: stamp!, system_size
using CedarSim.MNA: MNAIndex, NodeIndex, CurrentIndex, ChargeIndex
using CedarSim.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using CedarSim.MNA: TimeDependentVoltageSource, PWLVoltageSource, get_source_value, pwl_value
using CedarSim.MNA: VCVS, VCCS, CCVS, CCCS
using CedarSim.MNA: assemble!, assemble_G, assemble_C, get_rhs
using CedarSim.MNA: DCSolution, ACSolution, solve_dc, solve_ac
using CedarSim.MNA: voltage, current, magnitude_db, phase_deg
using CedarSim.MNA: make_ode_problem, make_ode_function
using CedarSim.MNA: make_dae_problem, make_dae_function
using CedarSim.MNA: reset_for_restamping!

# Import CedarSim for macros and dc!/tran!
using CedarSim
using CedarSim: dc!, tran!  # explicit import to avoid Julia 1.12 conflict
using CedarSim.MNA: MNACircuit, MNASpec
using OrdinaryDiffEq: Rodas5P, QNDF, FBDF
using VerilogAParser

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

        # Current variables return typed CurrentIndex
        i1 = alloc_current!(ctx, :I_V1)
        @test i1 isa CurrentIndex
        @test resolve_index(ctx, i1) == 3  # n_nodes + 1
        @test ctx.n_currents == 1
        @test system_size(ctx) == 3
    end

    @testset "Internal node allocation" begin
        ctx = MNAContext()

        # Initially no internal nodes
        @test n_internal_nodes(ctx) == 0

        # Regular nodes are not internal
        a = get_node!(ctx, :a)
        c = get_node!(ctx, :c)
        @test is_internal_node(ctx, a) == false
        @test is_internal_node(ctx, c) == false
        @test n_internal_nodes(ctx) == 0

        # Allocate internal node
        a_int = alloc_internal_node!(ctx, Symbol("D1.a_int"))
        @test a_int == 3  # After a and c
        @test ctx.n_nodes == 3
        @test is_internal_node(ctx, a_int) == true
        @test n_internal_nodes(ctx) == 1

        # Regular nodes still not internal
        @test is_internal_node(ctx, a) == false
        @test is_internal_node(ctx, c) == false

        # Allocate another internal node
        b_int = alloc_internal_node!(ctx, Symbol("D1.b_int"))
        @test b_int == 4
        @test n_internal_nodes(ctx) == 2

        # Ground is never internal
        @test is_internal_node(ctx, 0) == false

        # Invalid indices return false
        @test is_internal_node(ctx, -1) == false
        @test is_internal_node(ctx, 100) == false

        # String name works too
        c_int = alloc_internal_node!(ctx, "D2.c_int")
        @test c_int == 5
        @test n_internal_nodes(ctx) == 3

        # Existing internal node returns same index
        a_int_again = alloc_internal_node!(ctx, Symbol("D1.a_int"))
        @test a_int_again == a_int
        @test n_internal_nodes(ctx) == 3  # No change

        # Internal nodes participate in system size
        @test system_size(ctx) == 5  # 5 nodes, 0 currents

        # Can stamp to/from internal nodes
        stamp_conductance!(ctx, a, a_int, 0.001)  # R = 1k between a and a_int
        @test length(ctx.G_V) == 4

        # Current variables work alongside internal nodes (typed indices)
        i1 = alloc_current!(ctx, :I_V1)
        @test i1 isa CurrentIndex
        @test resolve_index(ctx, i1) == 6  # n_nodes + 1 = 5 + 1
        @test system_size(ctx) == 6
    end

    @testset "Internal node in circuit simulation" begin
        # Simulate a simple diode with series resistance pattern:
        # V1 --[Rs]-- a_int --[Rd]-- gnd
        # Where Rs is 10Ω and Rd is 1000Ω (representing junction conductance)
        #
        # This test specifically tests alloc_internal_node! and is_internal_node,
        # so it uses the builder pattern with direct stamping.

        function internal_node_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            anode = get_node!(ctx, :anode)
            a_int = alloc_internal_node!(ctx, Symbol("D1.a_int"))

            stamp!(VoltageSource(5.0; name=:V1), ctx, anode, 0)
            stamp!(Resistor(10.0), ctx, anode, a_int)
            stamp!(Resistor(1000.0), ctx, a_int, 0)
            return ctx
        end

        # Build once to get internal node info for assertions
        ctx = internal_node_circuit((;), MNASpec())
        a_int = 2  # Internal node is second node (anode=1, a_int=2)

        # Check internal node is marked correctly
        @test is_internal_node(ctx, a_int) == true
        @test is_internal_node(ctx, 1) == false  # anode

        # Solve using unified dc! path
        circuit = MNACircuit(internal_node_circuit)
        sol = dc!(circuit)

        # Verify voltages using voltage divider
        # V(anode) = 5.0 (forced by voltage source)
        # V(a_int) = 5.0 * 1000 / (10 + 1000) ≈ 4.9505
        @test voltage(sol, :anode) ≈ 5.0
        @test isapprox(sol.x[a_int], 5.0 * 1000 / 1010; rtol=1e-6)

        # Current through circuit
        # I = 5.0 / (10 + 1000) = 5.0 / 1010 ≈ 4.95mA
        @test isapprox(-current(sol, :I_V1), 5.0 / 1010; rtol=1e-6)
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
        @test ctx.G_I == [NodeIndex(1), NodeIndex(1), NodeIndex(2), NodeIndex(2)]
        @test ctx.G_J == [NodeIndex(1), NodeIndex(2), NodeIndex(1), NodeIndex(2)]
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

        # I_idx is now a typed CurrentIndex, resolved at assembly time
        @test I_idx isa CurrentIndex
        @test resolve_index(ctx, I_idx) == 2  # n_nodes + 1

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

        # I_idx is now a typed CurrentIndex
        @test I_idx isa CurrentIndex
        @test resolve_index(ctx, I_idx) == 3  # After 2 nodes

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

        # I_idx is now a typed CurrentIndex
        @test I_idx isa CurrentIndex
        @test resolve_index(ctx, I_idx) == 5  # After 4 nodes

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
        # Indices are now typed CurrentIndex
        @test I_in_idx isa CurrentIndex && I_out_idx isa CurrentIndex
        @test resolve_index(ctx, I_in_idx) == 5   # First current variable (after 4 nodes)
        @test resolve_index(ctx, I_out_idx) == 6  # Second current variable

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

        # Index is now typed CurrentIndex
        @test I_in_idx isa CurrentIndex
        @test resolve_index(ctx, I_in_idx) == 5  # After 4 nodes

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
    #
    # These tests use sp"..." SPICE macro to verify circuit behavior.
    # The sp"..." macro generates a builder function that is used with dc!().
    #==========================================================================#

    @testset "DC: Voltage divider" begin
        # Classic voltage divider: 5V source, 1k/1k resistors
        # Expected: Vout = 5 * 1k/(1k+1k) = 2.5V
        circuit = MNACircuit(sp"""
        V1 vcc 0 DC 5
        R1 vcc out 1k
        R2 out 0 1k
        """i)
        sol = dc!(circuit)
        @test voltage(sol, :vcc) ≈ 5.0
        @test voltage(sol, :out) ≈ 2.5 atol=1e-10
    end

    @testset "DC: Voltage divider (unequal)" begin
        # 5V source, 2k/1k resistors
        # Expected: Vout = 5 * 1k/(2k+1k) = 5/3 ≈ 1.6667V
        circuit = MNACircuit(sp"""
        V1 vcc 0 DC 5
        R1 vcc out 2k
        R2 out 0 1k
        """i)
        sol = dc!(circuit)
        @test voltage(sol, :out) ≈ 5.0 / 3.0 atol=1e-10
    end

    @testset "DC: Current source into resistor" begin
        # 1mA current source into 1k resistor
        # Expected: V = I * R = 0.001 * 1000 = 1V
        circuit = MNACircuit(sp"""
        I1 0 n1 DC 1m
        R1 n1 0 1k
        """i)
        sol = dc!(circuit)
        @test voltage(sol, :n1) ≈ 1.0 atol=1e-10
    end

    @testset "DC: Two voltage sources" begin
        # V1 = 5V at vcc, V2 = 3V at mid
        # R1 between vcc and mid, R2 between mid and gnd
        circuit = MNACircuit(sp"""
        V1 vcc 0 DC 5
        V2 mid 0 DC 3
        R1 vcc mid 1k
        R2 mid 0 1k
        """i)
        sol = dc!(circuit)
        @test voltage(sol, :vcc) ≈ 5.0
        @test voltage(sol, :mid) ≈ 3.0
    end

    @testset "DC: VCCS amplifier" begin
        # Input: 1V source, gm = 10mS
        # Output: into 1k resistor
        # Expected: Iout = gm * Vin = 0.01 * 1 = 10mA
        # Vout = Iout * R = 0.01 * 1000 = 10V
        circuit = MNACircuit(sp"""
        V1 inp 0 DC 1
        G1 out 0 inp 0 0.01
        R1 out 0 1k
        """i)
        sol = dc!(circuit)
        @test voltage(sol, :inp) ≈ 1.0
        @test voltage(sol, :out) ≈ 10.0 atol=1e-10
    end

    @testset "DC: Inverting amplifier with VCVS" begin
        # Simple inverting amp model with gain = -10
        # Vin = 0.5V, Vout should be -5V
        circuit = MNACircuit(sp"""
        V1 inp 0 DC 0.5
        E1 out 0 inp 0 -10.0
        """i)
        sol = dc!(circuit)
        @test voltage(sol, :inp) ≈ 0.5
        @test voltage(sol, :out) ≈ -5.0 atol=1e-10
    end

    @testset "DC: Transresistance amplifier with CCVS" begin
        # CCVS: Vout = rm * I_in
        # Current source I = 1mA through sensing branch
        # rm = 1000 Ω
        # Expected: Vout = 1000 * 1e-3 = 1V
        circuit = MNACircuit(sp"""
        I1 0 inp DC 1m
        H1 out 0 V_sense 1000.0
        V_sense inp 0 DC 0
        R1 out 0 1Meg
        """i)
        sol = dc!(circuit)
        @test voltage(sol, :out) ≈ 1.0 atol=1e-6
    end

    @testset "DC: Current mirror with CCCS" begin
        # CCCS: I_out = gain * I_in
        # Current source I = 1mA through sensing branch
        # Gain = 2
        # Output into 1k resistor
        # Expected: I_out = 2mA, V_out = 2mA * 1kΩ = 2V
        circuit = MNACircuit(sp"""
        I1 0 inp DC 1m
        F1 out 0 V_sense 2.0
        V_sense inp 0 DC 0
        R1 out 0 1k
        """i)
        sol = dc!(circuit)
        @test voltage(sol, :out) ≈ 2.0 atol=1e-10
    end

    @testset "DC: Multi-node network" begin
        # Star network: center node connected to 3 voltage sources via resistors
        # V1 = 3V (R=1k), V2 = 6V (R=2k), V3 = 9V (R=3k)
        # Vcenter = (V1/R1 + V2/R2 + V3/R3) / (1/R1 + 1/R2 + 1/R3)
        #         = 9 / 1.8333... ≈ 4.909V
        circuit = MNACircuit(sp"""
        V1 n1 0 DC 3
        V2 n2 0 DC 6
        V3 n3 0 DC 9
        R1 n1 center 1k
        R2 n2 center 2k
        R3 n3 center 3k
        """i)
        sol = dc!(circuit)
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

        # Initial condition should be DC solution (G\b)
        @test prob.u0 ≈ sys.G \ sys.b

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
        circuit3 = MNACircuit(sp"""
        V1 n1 0 DC 1
        R1 n1 0 1T
        """i)
        sol3 = dc!(circuit3)
        @test voltage(sol3, :n1) ≈ 1.0

        # Very small resistance
        circuit4 = MNACircuit(sp"""
        V1 n1 0 DC 5
        R1 n1 n2 1u
        R2 n2 0 1u
        """i)
        sol4 = dc!(circuit4)
        @test voltage(sol4, :n2) ≈ 2.5 atol=1e-6
    end

    #==========================================================================#
    # Pretty Printing (no crashes)
    #==========================================================================#

    @testset "Display functions" begin
        # Test display methods using sp"..." for circuit definition
        circuit = MNACircuit(sp"""
        V1 vcc 0 DC 5
        R1 vcc out 1k
        R2 out 0 1k
        """i)

        # Build ctx for context display test
        ctx = circuit.builder(circuit.params, circuit.spec, 0.0)

        # Should not error
        io = IOBuffer()
        show(io, ctx)
        show(io, MIME"text/plain"(), ctx)

        # Solve and test solution display
        sol = dc!(circuit)
        show(io, sol)
        show(io, MIME"text/plain"(), sol)

        # Also test MNASystem display
        ctx2 = circuit.builder(circuit.params, circuit.spec, 0.0)
        sys = assemble!(ctx2)
        show(io, sys)
        show(io, MIME"text/plain"(), sys)

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
        I_L_deferred = stamp!(Inductor(L_val), ctx, mid, 0)  # Returns deferred current index

        sys = assemble!(ctx)

        # Resolve the deferred index to actual system index
        I_L = resolve_index(ctx, I_L_deferred)

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
    # MNACircuit: Parameterized Circuit Wrapper
    #==========================================================================#

    # Import MNACircuit and related exports
    using CedarSim.MNA: MNASpec, alter, with_mode, with_spec, with_temp, eval_circuit
    using CedarSim.MNA: MNACircuit
    using SciMLBase: ODEProblem as SciMLODEProblem

    @testset "MNACircuit basics" begin
        # Define a parameterized circuit builder (new API: params, spec)
        function build_voltage_divider(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            stamp!(Resistor(params.R1), ctx, vcc, out)
            stamp!(Resistor(params.R2), ctx, out, 0)

            return ctx
        end

        # Create circuit with parameters
        circuit = MNACircuit(build_voltage_divider; Vcc=10.0, R1=1000.0, R2=1000.0)

        @test circuit.spec.mode == :tran
        @test circuit.params.Vcc == 10.0
        @test circuit.params.R1 == 1000.0

        # Build and solve
        sol = solve_dc(circuit)
        @test voltage(sol, :out) ≈ 5.0  # Voltage divider: Vcc * R2/(R1+R2)

        # Alter parameters
        circuit2 = alter(circuit; R2=3000.0)
        @test circuit2.params.R2 == 3000.0
        @test circuit2.params.R1 == 1000.0  # Unchanged

        sol2 = solve_dc(circuit2)
        @test voltage(sol2, :out) ≈ 7.5  # 10 * 3000/4000

        # Change mode
        circuit3 = with_mode(circuit, :dcop)
        @test circuit3.spec.mode == :dcop
        @test circuit3.params.Vcc == 10.0  # Params preserved

        # Test out-of-place eval_circuit
        sys = eval_circuit(circuit)
        @test sys isa MNASystem
    end

    @testset "MNACircuit with RC circuit" begin
        function build_rc(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        circuit = MNACircuit(build_rc; Vcc=5.0, R=1000.0, C=1e-6)

        # DC solution (capacitor is open circuit)
        sol = solve_dc(circuit)
        @test voltage(sol, :out) ≈ 5.0  # At DC, capacitor is open

        # AC sweep - cutoff fc = 1/(2πRC) ≈ 159Hz for R=1kΩ, C=1μF
        # At 10Hz (well below cutoff), gain should be ~1
        freqs = [10.0, 100.0, 1000.0]
        ac_sol = solve_ac(circuit, freqs)

        # At 10Hz (f << fc), gain ≈ 1 (within 1%)
        @test abs(voltage(ac_sol, :out)[1]) > 0.99 * circuit.params.Vcc
        # At 1000Hz (f >> fc), gain should be < 0.2 (rolloff)
        @test abs(voltage(ac_sol, :out)[3]) < 0.2 * circuit.params.Vcc
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
        function build_rc_step(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
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

        # Create parameterized circuit
        circuit = MNACircuit(build_rc_step; Vcc=5.0, R=1000.0, C=1e-6)

        # Test 1: DC analysis with default mode (:tran)
        sol_dc = solve_dc(circuit)
        @test voltage(sol_dc, :vcc) ≈ 5.0
        @test voltage(sol_dc, :out) ≈ 5.0  # steady state

        # Test 2: Change parameters using alter()
        circuit_modified = alter(circuit; R=2000.0, C=2e-6)
        sol_mod = solve_dc(circuit_modified)
        @test voltage(sol_mod, :vcc) ≈ 5.0
        @test voltage(sol_mod, :out) ≈ 5.0  # same DC, different dynamics

        # Verify time constant changed: τ = R*C
        τ_original = 1000.0 * 1e-6   # 1ms
        τ_modified = 2000.0 * 2e-6   # 4ms
        @test τ_modified == 4 * τ_original

        # Test 3: Change Vcc parameter and verify DC changes
        circuit_highv = alter(circuit; Vcc=10.0)
        sol_highv = solve_dc(circuit_highv)
        @test voltage(sol_highv, :out) ≈ 10.0

        # Test 4: Mode switching with with_mode()
        circuit_dcop = with_mode(circuit, :dcop)
        @test circuit_dcop.spec.mode == :dcop
        sol_dcop = solve_dc(circuit_dcop)
        @test voltage(sol_dcop, :out) ≈ 5.0

        # Test 5: Transient simulation with different parameters
        # Compare τ=1ms circuit vs τ=4ms circuit using high-level tran! API
        τ_fast = circuit.params.R * circuit.params.C  # 1ms
        τ_slow = circuit_modified.params.R * circuit_modified.params.C  # 4ms

        sol_fast = tran!(circuit, (0.0, 5τ_fast); solver=Rodas5P())
        sol_slow = tran!(circuit_modified, (0.0, 5τ_slow); solver=Rodas5P())

        # Both should start at steady state (DC-initialized)
        @test sol_fast(0.0)[2] ≈ 5.0 rtol=1e-3
        @test sol_slow(0.0)[2] ≈ 5.0 rtol=1e-3

        # Both should end at steady state
        @test sol_fast(5*τ_fast)[2] ≈ 5.0 rtol=1e-3
        @test sol_slow(5*τ_slow)[2] ≈ 5.0 rtol=1e-3

        # Test 6: Chain multiple alterations
        circuit_chain = alter(alter(circuit; Vcc=3.3); R=4700.0)
        sol_chain = solve_dc(circuit_chain)
        @test voltage(sol_chain, :vcc) ≈ 3.3
        @test voltage(sol_chain, :out) ≈ 3.3
        @test circuit_chain.params.R == 4700.0
        @test circuit_chain.params.Vcc == 3.3
        @test circuit_chain.params.C == 1e-6  # unchanged from original
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
        function build_mode_aware_rc(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
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

        # Create circuit with both DC and steady-state parameters
        circuit = MNACircuit(build_mode_aware_rc; Vdc=0.0, Vss=5.0, R=1000.0, C=1e-6)

        # Test DC mode gives Vdc
        circuit_dcop = with_mode(circuit, :dcop)
        sol_dcop = solve_dc(circuit_dcop)
        @test voltage(sol_dcop, :vcc) ≈ 0.0
        @test voltage(sol_dcop, :out) ≈ 0.0

        # Test transient mode gives Vss
        circuit_tran = with_mode(circuit, :tran)
        sol_tran = solve_dc(circuit_tran)
        @test voltage(sol_tran, :vcc) ≈ 5.0
        @test voltage(sol_tran, :out) ≈ 5.0

        # Run transient from DC operating point
        # This simulates the typical SPICE flow:
        # 1. Compute DC operating point (with sources at DC values)
        # 2. Run transient (with sources at transient values)

        # First, get DC operating point (cap starts at 0V)
        dc_sol = dc!(circuit_dcop)
        V_cap_dc = voltage(dc_sol, :out)
        @test V_cap_dc ≈ 0.0  # Cap at 0V in DC mode

        # Now run transient with source at 5V, cap starting at 0V
        tran_sys = assemble!(circuit_tran)
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
        function build_with_lens(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            # params.lens(; defaults...) returns params with lens overrides merged
            p = params.lens(; Vcc=5.0, R1=1000.0, R2=1000.0)

            stamp!(VoltageSource(p.Vcc), ctx, vcc, 0)
            stamp!(Resistor(p.R1), ctx, vcc, out)
            stamp!(Resistor(p.R2), ctx, out, 0)

            return ctx
        end

        # With identity lens: all defaults
        circuit_default = MNACircuit(build_with_lens; lens=IdentityLens())
        sol_default = dc!(circuit_default)
        @test voltage(sol_default, :out) ≈ 2.5  # 5V * 1k/(1k+1k)

        # With partial override: only R1 changed
        override_lens = ParamLens((params=(R1=3000.0,),))
        circuit_override = MNACircuit(build_with_lens; lens=override_lens)
        sol_override = dc!(circuit_override)
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
        function build_temp_dependent(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
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
        circuit_27 = MNACircuit(build_temp_dependent;
                        spec=MNASpec(temp=27.0),
                        Vcc=10.0, R0=1000.0, tc=0.004, R2=1000.0)
        sol_27 = solve_dc(circuit_27)
        @test voltage(sol_27, :out) ≈ 5.0  # Equal resistors

        # At 127°C, R_temp = 1000 * (1 + 0.004*100) = 1400 Ω
        circuit_127 = with_temp(circuit_27, 127.0)
        @test circuit_127.spec.temp == 127.0
        sol_127 = solve_dc(circuit_127)
        # Voltage divider: Vout = Vcc * R2/(R_temp + R2) = 10 * 1000/2400 ≈ 4.167V
        @test voltage(sol_127, :out) ≈ 10.0 * 1000.0 / 2400.0 rtol=1e-6

        # At -73°C, R_temp = 1000 * (1 + 0.004*(-100)) = 600 Ω
        circuit_m73 = with_temp(circuit_27, -73.0)
        sol_m73 = solve_dc(circuit_m73)
        # Voltage divider: Vout = 10 * 1000/1600 = 6.25V
        @test voltage(sol_m73, :out) ≈ 10.0 * 1000.0 / 1600.0 rtol=1e-6

        # Test with_spec
        new_spec = MNASpec(temp=85.0, mode=:dcop)
        circuit_85 = with_spec(circuit_27, new_spec)
        @test circuit_85.spec.temp == 85.0
        @test circuit_85.spec.mode == :dcop

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

    # dc!/tran! are already imported at the top of this file from CedarSim
    using CedarSim.MNA: MNASolutionAccessor, scope, NodeRef, ScopedSystem

    @testset "dc! and tran! API with MNACircuit" begin
        # Define a simple RC circuit
        function build_rc_simple(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        circuit = MNACircuit(build_rc_simple; Vcc=5.0, R=1000.0, C=1e-6)
        τ = 1000.0 * 1e-6  # 1ms

        # Test dc!(circuit) - matches CedarSim API
        dc_sol = dc!(circuit)
        @test dc_sol isa DCSolution
        @test voltage(dc_sol, :out) ≈ 5.0  # DC steady state

        # Test tran!(circuit, tspan) with ODE solver (Rodas5P)
        ode_sol = tran!(circuit, (0.0, 5τ); solver=Rodas5P())
        @test ode_sol.retcode == ReturnCode.Success

        # Wrap for symbolic access
        sys = assemble!(circuit)
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

    @testset "Hierarchical scope access with MNACircuit" begin
        # Build a circuit with hierarchical-like node names
        function build_hierarchical(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            # Simulate subcircuit x1 with nodes x1_in and x1_out
            x1_in = get_node!(ctx, :x1_in)
            x1_out = get_node!(ctx, :x1_out)
            vcc = get_node!(ctx, :vcc)

            stamp!(VoltageSource(params.Vcc), ctx, vcc, 0)
            stamp!(Resistor(params.R1), ctx, vcc, x1_in)
            stamp!(Resistor(params.R2), ctx, x1_in, x1_out)
            stamp!(Capacitor(params.C), ctx, x1_out, 0)  # Add cap for transient

            return ctx
        end

        circuit = MNACircuit(build_hierarchical; Vcc=10.0, R1=1000.0, R2=1000.0, C=1e-6)
        sys = assemble!(circuit)

        # Test ScopedSystem
        s = scope(sys)
        @test s isa ScopedSystem

        # Access hierarchical node
        node_ref = s.x1.out
        @test node_ref isa NodeRef
        @test node_ref.name == :out
        @test node_ref.path == [:x1]

        # Test with tran! using ODE solver
        ode_sol = tran!(circuit, (0.0, 1e-3); solver=Rodas5P())
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
        using CedarSim.MNA: MNASpec, SinVoltageSource

        # Test PWL voltage source stamping via builder pattern
        # The builder stamps a PWL at t=0.5ms where the value is 2.5V
        function pwl_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            pwl = PWLVoltageSource([0.0, 1e-3], [0.0, 5.0]; name=:Vpwl)
            stamp!(pwl, ctx, vcc, 0, 0.5e-3, :tran)
            return ctx
        end

        circuit = MNACircuit(pwl_circuit)
        sol = dc!(circuit)
        # At t=0.5ms, PWL value = 2.5V
        @test voltage(sol, :vcc) ≈ 2.5 atol=1e-10

        # Test SIN voltage source stamping via builder pattern
        # At t=0.25ms (1/4 period of 1kHz), sin = 5*sin(90°) = 5
        function sin_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            sin_src = SinVoltageSource(0.0, 5.0, 1000.0; name=:Vsin)
            stamp!(sin_src, ctx, vcc, 0, 0.25e-3, :tran)
            return ctx
        end

        circuit2 = MNACircuit(sin_circuit)
        sol2 = dc!(circuit2)
        @test voltage(sol2, :vcc) ≈ 5.0 atol=1e-10
    end

    @testset "Time-dependent ODE with builder" begin
        using OrdinaryDiffEq

        # Build RC circuit with PWL voltage source
        function build_pwl_rc(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            # PWL: ramp from 0 to Vmax over ramp_time, then hold
            pwl = PWLVoltageSource(
                [0.0, params.ramp_time],
                [0.0, params.Vmax];
                name=:Vpwl
            )
            stamp!(pwl, ctx, vcc, 0, t, spec.mode)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        params = (Vmax=5.0, ramp_time=1e-3, R=1000.0, C=1e-6)
        τ = params.R * params.C  # 1ms

        # Create ODE problem using MNACircuit (tspan passed to Problem, not circuit)
        circuit = MNACircuit(build_pwl_rc, params, MNASpec(temp=27.0))
        prob = SciMLODEProblem(circuit, (0.0, 5e-3))

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
        function build_sin_rc(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            sin_src = SinVoltageSource(
                params.vo, params.va, params.freq;
                name=:Vsin
            )
            stamp!(sin_src, ctx, vcc, 0, t, spec.mode)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        # Low-pass RC filter with SIN source
        # R = 1k, C = 1μF, fc = 159Hz
        # Source: 1kHz (well above cutoff) - should be attenuated
        params = (vo=0.0, va=5.0, freq=1000.0, R=1000.0, C=1e-6)
        fc = 1.0 / (2π * params.R * params.C)  # ~159 Hz

        # Create ODE problem using MNACircuit (tspan passed to Problem)
        circuit = MNACircuit(build_sin_rc, params, MNASpec(temp=27.0))
        prob = SciMLODEProblem(circuit, (0.0, 5e-3))

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

    #==========================================================================#
    # Multi-Solver Transient Test
    #==========================================================================#

    @testset "Multi-solver transient comparison" begin
        # Test that all supported solvers produce consistent results
        # This ensures both ODE and DAE paths work correctly
        # Use SIN source to create a dynamic circuit with actual transient behavior

        using CedarSim.MNA: SinVoltageSource

        function build_rc_sin(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            vcc = get_node!(ctx, :vcc)
            out = get_node!(ctx, :out)

            # SIN source: DC offset + AC amplitude at given frequency
            stamp!(SinVoltageSource(params.vo, params.va, params.freq; name=:Vin),
                   ctx, vcc, 0, t, spec.mode)
            stamp!(Resistor(params.R), ctx, vcc, out)
            stamp!(Capacitor(params.C), ctx, out, 0)

            return ctx
        end

        # RC low-pass filter with 1kHz sine input
        # RC time constant = 1ms, cutoff freq = 159 Hz
        # At 1kHz input, output is attenuated
        circuit = MNACircuit(build_rc_sin;
                             vo=2.5, va=2.5, freq=1000.0,  # 0-5V sine
                             R=1000.0, C=1e-6)
        tspan = (0.0, 3e-3)  # 3 periods of 1kHz

        # Test with IDA (DAE solver - default)
        # Note: IDA needs more relaxed tolerances for circuits with fast-changing sources
        sol_ida = tran!(circuit, tspan; abstol=1e-6, reltol=1e-4)
        @test sol_ida.retcode == ReturnCode.Success

        # Test with Rodas5P (Rosenbrock ODE solver)
        sol_rodas = tran!(circuit, tspan; solver=Rodas5P(), abstol=1e-10, reltol=1e-8)
        @test sol_rodas.retcode == ReturnCode.Success

        # Test with QNDF (BDF ODE solver - supports variable mass matrices)
        sol_qndf = tran!(circuit, tspan; solver=QNDF(), abstol=1e-10, reltol=1e-8)
        @test sol_qndf.retcode == ReturnCode.Success

        # Test with FBDF (BDF ODE solver - supports variable mass matrices)
        sol_fbdf = tran!(circuit, tspan; solver=FBDF(), abstol=1e-10, reltol=1e-8)
        @test sol_fbdf.retcode == ReturnCode.Success

        # All solvers should agree at multiple time points
        # Use 1% tolerance to accommodate different solver accuracies
        test_times = [0.5e-3, 1e-3, 1.5e-3, 2e-3, 2.5e-3]
        for t in test_times
            V_ida = sol_ida(t)[2]
            V_rodas = sol_rodas(t)[2]
            V_qndf = sol_qndf(t)[2]
            V_fbdf = sol_fbdf(t)[2]

            # Solvers should match each other within 1%
            @test isapprox(V_ida, V_rodas; rtol=0.01) || (t, V_ida, V_rodas)
            @test isapprox(V_ida, V_qndf; rtol=0.01) || (t, V_ida, V_qndf)
            @test isapprox(V_ida, V_fbdf; rtol=0.01) || (t, V_ida, V_fbdf)
        end
    end

end  # @testset "MNA Core Tests"
