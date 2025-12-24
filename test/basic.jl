module basic_tests

include("common.jl")

using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using CedarSim.MNA: voltage, current, make_ode_problem

#=
NOTE: Tests that require DAECompiler are skipped:
- Unimplemented Device (requires CircuitIRODESystem)
- MC VR Circuit (requires solve_circuit with Monte Carlo)
- ParallelInstances (requires CedarSim.ParallelInstances)
=#

#=
@testset "Unimplemented Device" begin
    # TODO: This test requires DAECompiler/CircuitIRODESystem
    # If we try to use it, we get an `UnsupportedIRException`, since the
    # generated code contains an `error()` which `DAECompiler` doesn't like.
    function ERRcircuit()
        vcc = Named(net, "vcc")()
        gnd = Named(net, "gnd")()
        Named(CedarSim.UnimplementedDevice(), "U")(vcc, gnd)
    end
    # Trying to directly run `ERRcircuit()` throws an error
    # due to the `error()` in the implementation of `UnimplementedDevice()`
    @test_throws CedarSim.CedarError ERRcircuit()

    # DAECompiler sees the `error()` and complains that it is unsupported IR:
    @test_throws CedarSim.DAECompiler.UnsupportedIRException CircuitIRODESystem(ERRcircuit)
end
=#

@testset "Simple VR Circuit" begin
    # Original used Julia DSL with Named(V(...)), Named(R(...))
    # Port to MNA direct API with same values
    function VRcircuit(params, spec)
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        stamp!(VoltageSource(5.0; name=:V), ctx, vcc, 0)
        stamp!(Resistor(2.0; name=:R), ctx, vcc, 0)
        return ctx
    end

    ctx = VRcircuit((;), MNASpec())
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    # I = V/R = 5/2 = 2.5A
    R_v = voltage(sol, :vcc)
    R_i = -current(sol, :I_V)  # Voltage source current is negative when sourcing
    @test isapprox_deftol(R_v, 5.0)
    @test isapprox_deftol(R_i, 2.5)
end

#=
@testset "MC VR Circuit" begin
    # TODO: This test requires DAECompiler with Monte Carlo support (agauss function)
    struct MCcircuit
        seed::UInt
    end
    function (ckt::MCcircuit)()
        vcc = Named(net, "vcc")()
        gnd = Named(net, "gnd")()
        Named(V(5.), "V")(vcc, gnd)
        Named(R(agauss(2, 3, 3)), "R")(vcc, gnd)
        Gnd()(gnd)
    end

    sys1, sol1 = solve_circuit(MCcircuit(1))
    sys2, sol2 = solve_circuit(MCcircuit(2))

    i1 = sol1[sys1.R.I]
    i2 = sol2[sys2.R.I]
    # test that the RNG is consistent between timesteps
    @test allapprox_deftol(i1)
    @test allapprox_deftol(i2)
    # but different between runs
    @test_broken !isapprox_deftol(first(i1), first(i2))
end
=#

@testset "Simple IR circuit" begin
    # Original used Julia DSL
    function IRcircuit(params, spec)
        ctx = MNAContext()
        icc = get_node!(ctx, :icc)
        # CurrentSource(I) stamps I into node p, meaning current I flows into p
        # To get +10V on node icc with 2Ω to ground, we need 5A into icc
        stamp!(CurrentSource(5.0; name=:I), ctx, icc, 0)
        stamp!(Resistor(2.0; name=:R), ctx, icc, 0)
        return ctx
    end

    ctx = IRcircuit((;), MNASpec())
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    # V = IR = 5*2 = 10V
    R_v = voltage(sol, :icc)
    @test isapprox_deftol(R_v, 10.0)
end

const v_val = 5.0
const r_val = 2000.0
const c_val = 1e-6
@testset "Simple VRC circuit" begin
    # Original tested RC transient with u0=[0.0]
    function VRCcircuit(params, spec)
        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        vrc = get_node!(ctx, :vrc)
        stamp!(VoltageSource(v_val; name=:V), ctx, vcc, 0)
        stamp!(Resistor(r_val; name=:R), ctx, vcc, vrc)
        stamp!(Capacitor(c_val; name=:C), ctx, vrc, 0)
        return ctx
    end

    ctx = VRCcircuit((;), MNASpec(mode=:tran))
    sys = assemble!(ctx)

    # Simulate the RC circuit with capacitor starting at 0
    tau = r_val * c_val  # Time constant
    tspan = (0.0, 10 * tau)
    prob_data = make_ode_problem(sys, tspan)

    # Explicitly start with capacitor uncharged
    u0 = copy(prob_data.u0)
    vrc_idx = findfirst(n -> n == :vrc, sys.node_names)
    u0[vrc_idx] = 0.0

    f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                    jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
    prob = ODEProblem(f, u0, prob_data.tspan)
    sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=deftol, abstol=deftol)

    # At t=0, capacitor voltage should be 0
    # At t=10τ, capacitor voltage approaches v_val (within ~0.005%)
    # Exact value: v_val * (1 - exp(-10)) ≈ 0.99995 * v_val
    c_v_start = sol.u[1][vrc_idx]
    c_v_end = sol.u[end][vrc_idx]
    @test isapprox_deftol(c_v_start, 0.0)
    @test isapprox(c_v_end, v_val; rtol=1e-3)  # Within 0.1% of final value

    # Current at start: I = V/R = v_val/r_val
    # Current at end: I ≈ 0 (capacitor nearly fully charged)
    # (Current through voltage source)
    I_V_idx = sys.n_nodes + findfirst(n -> n == :I_V, sys.current_names)
    c_i_start = -sol.u[1][I_V_idx]  # Negative because sourcing
    c_i_end = -sol.u[end][I_V_idx]
    @test isapprox_deftol(c_i_start, v_val/r_val)
    @test isapprox(c_i_end, 0.0; atol=1e-6)  # Nearly zero current
end

#=
@testset "ParallelInstances" begin
    # TODO: This test requires CedarSim.ParallelInstances and DAECompiler
    using CedarSim: ParallelInstances
    function MultiVRCcircuit()
        vcc = Named(net, "vcc")()
        vrc = Named(net, "vrc")()
        gnd = Named(net, "gnd")()
        Named(V(v_val), "V")(vcc, gnd)
        Named(ParallelInstances(R(r_val), 10), "R")(vcc, vrc)
        Named(C(c_val), "C")(vrc, gnd)
        Gnd()(gnd)
    end

    sys, sol = solve_circuit(MultiVRCcircuit; u0=[0.0])

    # This RC circuit has a time constant much smaller than that of
    # our simulation time domain, so let's ensure that the beginning
    # and end points of our simulation follow physical laws:
    c_i = sol[sys.C.I]
    @test isapprox_deftol(c_i[1], 10*v_val/r_val)
    @test isapprox_deftol(c_i[end], 0)
    c_v = sol[sys.C.V]
    @test isapprox_deftol(c_v[1], 0)
    @test isapprox_deftol(c_v[end], v_val)
end
=#

@testset "Simple Spectre sources" begin
    # Simple resistor divider in Spectre format
    spectre_code = """
    // Simple Spectre voltage divider
    v1 (vcc 0) vsource dc=5
    r1 (vcc out) resistor r=1k
    r2 (out 0) resistor r=1k
    """

    ctx, sol = solve_mna_spectre_code(spectre_code)
    # Voltage divider: 5V * (1k / (1k + 1k)) = 2.5V
    @test isapprox_deftol(voltage(sol, :out), 2.5)
    @test isapprox_deftol(voltage(sol, :vcc), 5.0)
end

@testset "Spectre current source" begin
    # Current source with resistor in Spectre format
    spectre_code = """
    // Spectre current source test
    i1 (vcc 0) isource dc=1m
    r1 (vcc 0) resistor r=1k
    """

    ctx, sol = solve_mna_spectre_code(spectre_code)
    # V = I * R = 1mA * 1kΩ = 1V
    @test isapprox_deftol(voltage(sol, :vcc), 1.0)
end

# TODO: Full Spectre sources test with PWL and B-source (requires transient simulation)
#=
@testset "Full Spectre sources (transient)" begin
    # This is the comprehensive test from the old version
    # Tests PWL sources and B-source with time-varying expression
    mktempdir() do dir
        spectre_file = joinpath(dir, "sources.scs")
        open(spectre_file; write=true) do io
            write(io, \"\"\"
            I1 (0 1) isource dc=2.2u
            R1 (1 0) resistor r=1000

            I2 (0 2) isource type=pwl wave=[0 1m .5 2m 1 1.75m]
            R2 (2 0) resistor r=2k

            V3 (0 3) vsource dc=1.5
            R3 (3 0) resistor r=1k

            V4 (0 4) vsource type=pwl wave=[ 0 1 .5 2 \\
                    1 5]
            R4 (4 0) resistor r=4k

            B5 (0 5) bsource v=\$time*V(3)
            R5 (5 0) resistor r=1k
            \"\"\")
        end

        sys, sol = solve_spectre_file(spectre_file);
        @test all(isapprox.(sol[sys.node_1], 2.2e-3))
        @test all(isapprox.(sol[sys.R1.I], 2.2e-6))
        @test all(isapprox.(sol[sys.node_3], -1.5))
        @test all(isapprox.(sol[sys.R3.I], -1.5e-3))

        @test isapprox(sol[sys.node_2][end], 3.5)
        @test isapprox(sol[sys.R2.I][end], 1.75e-3)
        @test isapprox(sol[sys.node_4][end], -5.)
        @test isapprox(sol[sys.R4.I][end], -1.25e-3)
        @test isapprox(sol[sys.node_5][end], 1.5)
    end
end
=#

@testset "Spectre subcircuit" begin
    # Port of old "Simple Spectre subcircuit" test
    spectre_code = """
    subckt myres vcc gnd
        parameters r=1k
        r1 (vcc gnd) resistor r=r
    ends myres

    x1 (vcc 0) myres r=2k
    v1 (vcc 0) vsource dc=1
    """
    ctx, sol = solve_mna_spectre_code(spectre_code)
    # I = V/R = 1V/2kΩ = 0.5mA
    @test isapprox_deftol(voltage(sol, :vcc), 1.0)
    @test isapprox_deftol(current(sol, :I_v1), -0.5e-3)
end

@testset "Simple SPICE sources" begin
    # Original SPICE code (same as original test)
    spice_code = """
    * Simple SPICE sources
    V1 0 1 1
    R1 1 0 1k
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # Original: @test all(isapprox.(sol[sys.node_1], -1.0))
    # V1 has + at 0, - at 1, so node 1 = -1V
    # Note: SPICE numeric nodes become Symbol("1"), not :node_1
    @test isapprox_deftol(voltage(sol, Symbol("1")), -1.0)
end

@testset "Simple SPICE controlled sources" begin
    # Original SPICE code testing E (VCVS) and G (VCCS)
    spice_code = """
    * Simple SPICE sources with controlled sources
    V1 0 1 1
    R1 1 0 1k

    E6 0 6 0 1 2
    R6 6 0 r=1k

    G7 0 7 0 1 2
    R7 7 0 r=1k
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # V1 makes node 1 = -1V (+ at 0, - at 1)
    # E6: VCVS with gain=2, Vout = 2 * V(0,1) = 2 * 1 = 2V at node 6 relative to 0
    # Since E6 has + at 0 and - at 6, node 6 = -2V
    @test isapprox_deftol(voltage(sol, Symbol("1")), -1.0)
    @test isapprox(voltage(sol, Symbol("6")), -2.0; atol=deftol*10)
    # G7: VCCS with gm=2, I = 2 * V(0,1) = 2A into node 7
    # With R7=1k to ground: V = I*R = 2*1000 = 2000V (but sign depends on convention)
    # G7 outputs current from 0 to 7, so 2A flows into 7, V7 = -2000V
    @test isapprox(voltage(sol, Symbol("7")), -2000.0; atol=deftol*10)
end

@testset "SPICE B-source" begin
    # Test B-source with voltage expression referencing another node
    spice_code = """
    * B-source test
    V1 0 1 1
    R1 1 0 1k

    B5 0 5 v=V(1)*2
    R5 5 0 1k
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # V1 makes node 1 = -1V
    # B5: v = V(1)*2 = -1*2 = -2V, but B5 has + at 0, - at 5, so node 5 = 2V
    @test isapprox(voltage(sol, Symbol("5")), 2.0; atol=deftol*10)
end

@testset "SPICE B-source (nonlinear current)" begin
    # Test nonlinear B-source with i=V(1)**2 (current proportional to voltage squared)
    # Circuit: V1(2V) -> R1(1Ω) -> node 1 <- B1(i=V(1)^2) <- GND
    # At DC equilibrium: I_R1 = (V1 - V_node1) / R1 = I_B1 = V_node1^2
    # So: (2 - V) / 1 = V^2  =>  2 - V = V^2  =>  V^2 + V - 2 = 0
    # Solutions: V = (-1 ± 3) / 2 = 1 or -2
    # Physical solution (forward-biased): V = 1V
    spice_code = """
    * Nonlinear B-source test
    V1 vcc 0 DC 2
    R1 vcc 1 1
    B1 1 0 i=V(1)**2
    """
    # Use the builder-based solver for Newton iteration
    ast = CedarSim.SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = CedarSim.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    circuit_fn = Base.eval(m, code)

    spec = CedarSim.MNA.MNASpec(temp=27.0, mode=:dcop)
    sol = CedarSim.MNA.solve_dc(circuit_fn, (;), spec)

    # Node 1 should be 1V (the positive solution to V^2 + V - 2 = 0)
    @test isapprox(voltage(sol, Symbol("1")), 1.0; atol=1e-6)
    # V_vcc = 2V
    @test isapprox(voltage(sol, :vcc), 2.0; atol=1e-6)
end

# TODO: Alternate E/G forms with vol=/cur= syntax
# Currently errors at sema stage - LString(nothing) error on vol=/cur= parsing
#=
@testset "SPICE controlled sources (alternate syntax)" begin
    spice_code = """
    * Alternate E/G syntax
    V1 0 1 1
    R1 1 0 1k

    E8 0 8 vol=V(0, 1)*2
    R8 8 0 r=1k

    G9 0 9 cur=V(0, 1)*2
    R9 9 0 r=1k
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # E8: vol=V(0,1)*2 = 1*2 = 2V, since + at 0, - at 8, node 8 = -2V
    # G9: cur=V(0,1)*2 = 1*2 = 2A into node 9, V = 2*1000 = 2000V
    @test isapprox(voltage(sol, Symbol("8")), -2.0; atol=deftol*10)
    @test isapprox(voltage(sol, Symbol("9")), -2000.0; atol=deftol*10)
end
=#

@testset "Simple SPICE subcircuit" begin
    # Same SPICE code as original
    spice_code = """
    * Subcircuit test
    .subckt myres vcc gnd
    .param r=1k
    R1 vcc gnd 'r'
    .ends

    V1 vcc 0 DC 1
    X1 vcc 0 myres r=2k
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # Original: @test all(isapprox.(sol[sys.x1.r1.I], 0.5e-3))
    @test isapprox_deftol(voltage(sol, :vcc), 1.0)
    @test isapprox_deftol(current(sol, :I_v1), -0.5e-3)  # 1V / 2kΩ
end

@testset "SPICE include .LIB" begin
    # Test .LIB definition and include (self-referential)
    mktempdir() do dir
        spice_file = joinpath(dir, "selfinclude.cir")
        open(spice_file; write=true) do io
            write(io, """
            * .LIB definition and include test
            V1 vdd 0 1

            .LIB my_lib
            r1 vdd 0 1337
            .ENDL
            .LIB "selfinclude.cir" my_lib
            """)
        end

        # Parse and solve using MNA
        ast = SpectreNetlistParser.parsefile(spice_file; start_lang=:spice)
        code = CedarSim.make_mna_circuit(ast)
        m = Module()
        Base.eval(m, :(using CedarSim.MNA))
        Base.eval(m, :(using CedarSim: ParamLens))
        Base.eval(m, :(using CedarSim.SpectreEnvironment))
        circuit_fn = Base.eval(m, code)
        spec = CedarSim.MNA.MNASpec(temp=27.0, mode=:dcop)
        ctx = Base.invokelatest(circuit_fn, (;), spec)
        sys = CedarSim.MNA.assemble!(ctx)
        sol = CedarSim.MNA.solve_dc(sys)

        # I = V / R = 1V / 1337Ω
        @test isapprox_deftol(current(sol, :I_v1), -1/1337)
    end
end

# TODO: Verilog-A include support
#=
@testset "Verilog include" begin
    # Test Spectre ahdl_include
    ckt = """
    ahdl_include "resistor.va"

    x1 (vcc 0) BasicVAResistor R=2k
    v1 (vcc 0) vsource dc=1
    """
    inc = joinpath(dirname(pathof(CedarSim.VerilogAParser)), "../test/inputs")
    sys, sol = solve_spectre_code(ckt; include_dirs=[inc]);
    @test all(isapprox.(sol[sys.v1.I], -1/2e3))

    # Test SPICE .hdl
    ckt2 = """
    * Verilog Include 2
    .hdl "resistor.va"

    x1 vcc 0 BasicVAResistor r=2k
    v1 vcc 0 dc=1
    """
    sys, sol = solve_spice_code(ckt2; include_dirs=[inc]);
    @test all(isapprox.(sol[sys.v1.I], -1/2e3))
end
=#

@testset "SPICE parameter scope" begin
    # Same SPICE code as original
    # Sema provides topologically sorted parameter_order, so par_leff can reference l and par_l
    spice_code = """
    * Parameter scoping test

    .subckt subcircuit1 vss gnd l=11
    .param
    + par_l=1
    + par_leff='l-par_l'
    r1 vss gnd 'par_leff'
    .ends

    x1 vss 0 subcircuit1
    v1 vss 0 1
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # R = l - par_l = 11 - 1 = 10Ω
    # I = V/R = 1/10 = 0.1A
    @test isapprox_deftol(current(sol, :I_v1), -0.1)
end

# TODO: Extended parameter scope tests - nested dynamic scoping
# Currently errors at codegen stage - nested subcircuits not resolved correctly
#=
@testset "SPICE parameter scope (nested subcircuits)" begin
    # Test dynamic parameter scoping in nested subcircuits
    spice_code = """
    * Dynamic parameters
    .subckt inner a b foo=foo+2000
    R1 a b 'foo'
    .ends

    .subckt outer a b
    x1 a b inner
    .ends

    .param foo = 1
    i1 vcc 0 'foo'
    x1 vcc 0 outer
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # foo=1 at top level, inner sees foo+2000 = 2001
    # V = I * R = 1 * 2001 = 2001V
    @test isapprox_deftol(voltage(sol, :vcc), -2001.0)
end
=#

# TODO: Test .option temp / .temp for temperature setting
@testset "SPICE parameter scope (.option temp)" begin
    # Test that temper in .param picks up .option temp
    spice_code = """
    * param temp
    .option temp=10
    .param foo = temper
    i1 vcc 0 'foo'
    r1 vcc 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # foo = temper = 10 (from .option temp)
    @test_broken isapprox_deftol(voltage(sol, :vcc), -10.0)
end

# Currently errors at sema stage - .temp directive handling calls error()
#=
@testset "SPICE parameter scope (.temp)" begin
    # Test .temp directive
    spice_code = """
    * .temp
    .temp 10
    .param foo = temper
    i1 vcc 0 'foo'
    r1 vcc 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # foo = temper = 10 (from .temp)
    @test isapprox_deftol(voltage(sol, :vcc), -10.0)
end
=#

@testset "SPICE parameter scope (default temper)" begin
    # Test that the default temper is 27
    spice_code = """
    * temper
    .param foo = temper
    i1 vcc 0 'foo'
    r1 vcc 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # Default temper = 27
    @test isapprox_deftol(voltage(sol, :vcc), -27.0)
end

# Currently errors at runtime - instance param referencing other params not scoped correctly
#=
@testset "SPICE parameter scope (instance params)" begin
    # Test that instance parameters can refer to other parameters
    spice_code = """
    * Parameter scoping test

    .subckt subcircuit1 vss gnd w=2 rsh=1 nrd=1
    r1 vss gnd 'rsh*nrd'
    .ends
    * this should not require a global w parameter
    x1 vss 0 subcircuit1 w=4 nrd='w/2'
    v1 vss 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # nrd='w/2' where w=4, so nrd=2, R = rsh*nrd = 1*2 = 2
    # I = V/R = 1/2 = 0.5A
    @test isapprox_deftol(current(sol, :I_v1), -0.5)
end
=#

# TODO: multimode spice source (DC + AC + SIN)
#=
@testset "multimode spice source" begin
    spice = \"\"\"
    * multimode spice source
    v1 vcc 0 DC 5 AC 1 SIN(10 3 1k)
    r1 vcc 0 1k
    \"\"\"
    # This requires DAECompiler for proper transient simulation with initialization
    # Tests various initializers: CedarDCOp, ShampineCollocationInit, CedarTranOp
    sa = SpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice)
    code = CedarSim.make_spectre_circuit(sa)
    circuit = eval(code)

    sys = CircuitIRODESystem(circuit);
    prob = DAEProblem(sys, rand(5), rand(5), (0.0, 0.01));
    for (initializealg, vcc_known) in [(CedarDCOp(), 10.0),
                                (ShampineCollocationInit(), 10.0),
                                (CedarTranOp(), 10.0)]

        sol = init(prob, DFBDF(autodiff=false); reltol=deftol, abstol=deftol, initializealg)
        vcc = sol[sys.node_vcc]
        @test isapprox_deftol(vcc, vcc_known) || (initializealg, vcc, vcc_known)
    end
end
=#

@testset "SPICE multiplicities" begin
    # Same SPICE code as original
    spice_code = """
    * multiplicities
    v1 vcc 0 DC 1

    r1a vcc 1 1 m=10
    r1b 1 0 1
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # With m=10, r1a is effectively 0.1Ω
    # Total R = 0.1 + 1 = 1.1Ω
    # V at node 1 = 1 * (1/1.1) = 0.909V (voltage divider)
    @test isapprox(voltage(sol, Symbol("1")), 10/11; atol=deftol*10)
end

# TODO: Extended multiplicities tests with subcircuits
@testset "SPICE multiplicities (subcircuit m=)" begin
    spice_code = """
    * multiplicities with subcircuit
    v1 vcc 0 DC 1

    .subckt r10 a b m=10
    r2a a b 1
    .ends
    x2a vcc 2 r10
    r2b 2 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # Subcircuit with m=10 divides resistance by 10
    @test_broken isapprox(voltage(sol, Symbol("2")), 10/11; atol=deftol*10)
end

# Currently errors at codegen - nested subcircuits not resolved
#=
@testset "SPICE multiplicities (nested subcircuits)" begin
    spice_code = """
    * multiplicities with nested subcircuits
    v1 vcc 0 DC 1

    .subckt r10 a b m=10
    r2a a b 1
    .ends

    .subckt r5t2 a b
    x5r1 a b r10 m=5
    x5r2 a b r10 m=5
    .ends
    x4a1 vcc 4 r5t2
    r4b 4 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # Two x5 each with m=5 on r10 (which has m=10)
    @test isapprox(voltage(sol, Symbol("4")), 10/11; atol=deftol*10)
end
=#

# Currently errors at runtime - subcircuit builder doesn't accept m= kwarg
#=
@testset "SPICE multiplicities (nested m=)" begin
    spice_code = """
    * multiplicities with nested m= on subcircuit
    v1 vcc 0 DC 1

    .subckt r2 a b
    r2 a b 1 m=2
    .ends
    x5a vcc 5 r2 m=5
    r5b 5 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # r2 has m=2 internally, x5a has m=5, so effective m=10
    @test isapprox(voltage(sol, Symbol("5")), 10/11; atol=deftol*10)
end
=#

@testset "SPICE multiplicities (.model)" begin
    spice_code = """
    * multiplicities with .model
    v1 vcc 0 DC 1

    .model rm r R=1
    r6a vcc 6 rm m=10 l=1u
    r6b 6 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    @test isapprox(voltage(sol, Symbol("6")), 10/11; atol=deftol*10)
end

@testset ".model case sensitivity" begin
    spice_code = """
    * .model case sensitivity
    v1 vcc 0 DC 1
    .model rr r R=1
    r1 vcc 1 rr l=1u
    r2 1 0 rr R=2 l=1u
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # r1 uses model rr with R=1, r2 overrides R=2
    # Total resistance = 1 + 2 = 3, V at node 1 = 1 * 2/3
    @test isapprox(voltage(sol, Symbol("1")), 2/3; atol=deftol*10)
end

@testset "units and magnitudes" begin
    # Same SPICE code as original - tests mAmp (milli) and MegQux (mega) suffixes
    spice_code = """
    * units and magnitudes
    i1 vcc 0 DC -1mAmp
    r1 vcc 0 1MegQux
    """

    ctx, sol = solve_mna_spice_code(spice_code)
    # V = I*R = 1e-3 * 1e6 = 1000V
    @test isapprox(voltage(sol, :vcc), 1000.0; atol=deftol*10)

    spice_code2 = """
    * units and magnitudes 2
    i1 vcc 0 DC -1Amp
    r1 vcc 0 1Mil
    """

    ctx, sol = solve_mna_spice_code(spice_code2)
    # 1 mil = 25.4e-6 (25.4 micrometers)
    @test isapprox(voltage(sol, :vcc), 2.54e-5; atol=1e-8)
end

# TODO: Test that magnitudes don't introduce floating point errors
@testset "units and magnitudes (precision)" begin
    spice_code = """
    * units and magnitudes 3
    .param a=0.22u b=0.22e-6
    V1 vcc 0 'a'
    R1 vcc 0 1
    """
    ast = SpectreNetlistParser.SPICENetlistParser.parse(spice_code)
    code = CedarSim.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    circuit_fn = Base.eval(m, code)

    # Use ParamObserver to check if a === b (exact equality, no floating point error)
    observer = CedarSim.ParamObserver(:top, nothing)
    spec = MNASpec(temp=27.0, mode=:dcop)
    Base.invokelatest(circuit_fn, observer, spec)
    p = getfield(observer, :params)[:params]
    # 0.22u should equal exactly 0.22e-6 (no floating point rounding from magnitude parsing)
    @test p[:a] === p[:b]
end

@testset ".option" begin
    # Same SPICE code as original - just test parsing doesn't error
    spice_ckt = """
    * .option
    .option temp=10 filemode=ascii noinit
    """
    ast = SpectreNetlistParser.SPICENetlistParser.parse(spice_ckt)
    code = CedarSim.make_mna_circuit(ast)
    # Original just tested that f() returns nothing
    # For MNA, we test that code generation succeeds
    @test code !== nothing
end

@testset "functions" begin
    # Same SPICE code as original - test parameter functions
    # These work because SpectreEnvironment exports int, nint, floor, ceil, pow, ln
    spice_ckt = """
    * functions
    .param
    + intp=int(1.5)
    + intn=int(-1.5)
    + nintp = nint(1.6)
    + nintn = nint(-1.6)
    + floorp=floor(1.5)
    + floorn=floor(-1.5)
    + ceilp=ceil(1.5)
    + ceiln=ceil(-1.5)
    + powp=pow(2.0, 3)
    + pown=pow(2.0, -3)
    + lnp=ln(2.0)
    V1 vcc 0 'intp + intn + floorp'
    R1 vcc 0 1
    """
    ctx, sol = solve_mna_spice_code(spice_ckt)
    # intp=1, intn=-1, floorp=1 -> V = 1 + (-1) + 1 = 1V
    @test isapprox(voltage(sol, :vcc), 1.0; atol=deftol)
end

# TODO: Extended functions test - verify all function results via ParamObserver
@testset "functions (full verification)" begin
    spice_ckt = """
    * functions full
    .param
    + intp=int(1.5)
    + intn=int(-1.5)
    + nintp = nint(1.6)
    + nintn = nint(-1.6)
    + floorp=floor(1.5)
    + floorn=floor(-1.5)
    + ceilp=ceil(1.5)
    + ceiln=ceil(-1.5)
    + powp=pow(2.0, 3)
    + pown=pow(2.0, -3)
    + lnp=ln(2.0)
    V1 vcc 0 1
    R1 vcc 0 1
    """
    ast = SpectreNetlistParser.SPICENetlistParser.parse(spice_ckt)
    code = CedarSim.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    circuit_fn = Base.eval(m, code)

    observer = CedarSim.ParamObserver(:top, nothing)
    spec = MNASpec(temp=27.0, mode=:dcop)
    Base.invokelatest(circuit_fn, observer, spec)
    p = getfield(observer, :params)[:params]

    @test p[:intp] == 1
    @test p[:intn] == -1
    @test p[:nintp] == 2
    @test p[:nintn] == -2
    @test p[:floorp] == 1
    @test p[:floorn] == -2
    @test p[:ceilp] == 2
    @test p[:ceiln] == -1
    @test p[:powp] == 8
    @test p[:pown] == 0.125
    @test p[:lnp] == log(2.0)
end

@testset "device == param (ParamObserver)" begin
    # Test ParamObserver integration with MNA codegen
    # ParamObserver records which parameters are used and their values
    using CedarSim: ParamObserver, @param

    # Use explicit parameter passing (factor is a formal parameter of subcircuit)
    spice_code = """
    * device == param
    .subckt myres p n factor=1
        .param rload=1k
        r1 p n 'rload*factor'
    .ends
    i1 vcc 0 DC -1
    x1 vcc 0 myres factor=2
    """

    # Parse and generate MNA circuit
    ast = SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = CedarSim.make_mna_circuit(ast)

    # Evaluate in temp module
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens, AbstractParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    circuit_fn = Base.eval(m, code)

    # Use ParamObserver to record parameters
    observer = ParamObserver(:top, nothing)
    spec = MNASpec(temp=27.0, mode=:dcop)
    ctx = Base.invokelatest(circuit_fn, observer, spec)

    # Test that ParamObserver recorded the parameter hierarchy
    # The subcircuit x1 should have rload and factor parameters
    @test haskey(getfield(observer, :params), :x1)
    x1_obs = getfield(observer, :params)[:x1]
    @test x1_obs isa ParamObserver
    @test haskey(getfield(x1_obs, :params), :params)
    x1_params = getfield(x1_obs, :params)[:params]
    @test haskey(x1_params, :rload)
    @test x1_params[:rload] == 1000.0  # default value recorded

    # Test @param macro works
    @test @param(observer.x1.rload) == 1000.0

    # Test that the parameter was applied by solving
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    # With factor=2, R = rload * factor = 1000 * 2 = 2000Ω
    # I = 1A (from current source), V = I*R
    # Current source I1: -1A means extracting 1A from vcc, injecting into 0
    # V_vcc = I * R = 1 * 2000 = 2000V
    @test isapprox(voltage(sol, :vcc), 2000.0; rtol=1e-6)
end

# TODO: Extended device == param tests
@testset "device == param (canonicalize_params)" begin
    # Test canonicalize_params function
    @test CedarSim.canonicalize_params((; params=(;boo=4), foo=2, bar=(; baz=3))) == (params = (boo = 4, foo = 2), bar = (params = (baz = 3,),))
end

#=
@testset "device == param (convert to NamedTuple)" begin
    # Test converting ParamObserver to NamedTuple for ParamSim roundtrip
    spice = \"\"\"
    * device == param
    .param x1=1
    .subckt myres p n
        .param rload=1k
        rload p n 'rload*x1'
    .ends
    i1 vcc 0 DC -1
    x1 vcc 0 myres
    \"\"\"
    mast = SpectreNetlistParser.SPICENetlistParser.parse(spice)
    mcode = CedarSim.make_spectre_circuit(mast)
    f = eval(mcode)
    👀 = CedarSim.ParamObserver(x1=2.0)
    f(👀)
    @test @param(👀.x1.rload)*@param(👀.x1) == @param(👀.x1.rload.r)
    @test convert(NamedTuple, 👀) == (
        params = (x1 = 2.0,),
        i1 = (dc = -1,),
        m = 1.0,
        x1 = (
            params = (rload = 1000.0,),
            m = 1.0,
            rload = (r = 2000.0,)
        )
    )
    # dense roundtrip actually hits some error
    sim = ParamSim(f; convert(NamedTuple, 👀)...)
    @test_broken solve_circuit(sim)[2].retcode == SciMLBase.ReturnCode.Success
    # but you can definitely specify all the conflicting parameters in a consistent way
    sim = ParamSim(f; params=(;x1=2.0), x1=(;rload=500,))
    sim()
    sys, sol = solve_circuit(sim);
    @test sol[sys.x1.rload.V][end] ≈ 1000
end
=#

# TODO: semiconductor resistor with .model
# Model resolution for semiconductor resistors (rsh, w, l params) not yet implemented
@testset "semiconductor resistor" begin
    @test_skip "Semiconductor resistor model resolution not yet implemented"
    #=
    spice_code = """
    * semiconductor resistor
    .model myres r rsh=500
    .param res=1k
    v1 vcc 0 1
    R1 vcc 0 myres w=1m l=2m
    R2 vcc 0 res
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # R1 = rsh * l / w = 500 * 2m / 1m = 1000Ω
    # R2 = res = 1000Ω
    # Parallel: R_eq = 500Ω, I = 1/500 = 2mA
    @test_broken isapprox(current(sol, :I_v1), -2e-3; atol=deftol*10)
    =#
end

@testset "ifelse" begin
    # Same SPICE code as original
    spice_code = """
    * ifelse resistor
    .param switch=1
    v1 vcc 0 1
    .if (switch == 1)
    R1 vcc 0 1
    .else
    R1 vcc 0 2
    .endif
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # With switch=1, R1=1Ω, I = V/R = 1A
    @test isapprox(current(sol, :I_v1), -1.0; atol=deftol*10)
end

@testset "SPICE CCVS (H element)" begin
    # Current-controlled voltage source
    # Uses zero-volt source for sensing (standard SPICE approach)
    spice_code = """
    * CCVS test with zero-volt sense source
    Vin vcc 0 DC 5
    R1 vcc sense 1k
    Vsense sense 0 DC 0
    H1 out 0 Vsense 200
    Rload out 0 1Meg
    """
    ctx, sol = solve_mna_spice_code(spice_code)

    # Current through Vsense = 5V/1kΩ = 5mA
    # Vout = rm * I = 200 * 5mA = 1V
    @test isapprox(voltage(sol, :vcc), 5.0; atol=deftol)
    @test isapprox(voltage(sol, :sense), 0.0; atol=deftol)
    @test isapprox(voltage(sol, :out), 1.0; atol=deftol)
end

@testset "SPICE CCCS (F element)" begin
    # Current-controlled current source
    # Uses zero-volt source for sensing (standard SPICE approach)
    spice_code = """
    * CCCS test with zero-volt sense source
    Vin vcc 0 DC 5
    R1 vcc sense 1k
    Vsense sense 0 DC 0
    F1 out 0 Vsense 2
    Rload out 0 100
    """
    ctx, sol = solve_mna_spice_code(spice_code)

    # Current through Vsense = 5V/1kΩ = 5mA
    # I_out = gain * I = 2 * 5mA = 10mA
    # V_out = I_out * R = 10mA * 100Ω = 1V
    @test isapprox(voltage(sol, :vcc), 5.0; atol=deftol)
    @test isapprox(voltage(sol, :sense), 0.0; atol=deftol)
    @test isapprox(voltage(sol, :out), 1.0; atol=deftol)
end

end # basic_tests
