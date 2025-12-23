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
- device == param (requires ParamSim and ParamObserver)
- semiconductor resistor (requires .model support)
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
@testset "Simple Spectre sources" begin
    # TODO: Spectre parsing needs MNA codegen support
    # Original test used solve_spectre_file
end
=#

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

#=
# TODO: .LIB include handling needs to be implemented for MNA codegen
@testset "SPICE include .LIB" begin
    # Test .LIB definition and include
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
        ast = SpectreNetlistParser.parse(spice_file; start_lang=:spice)
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

        # Original: @test isapprox_deftol(sol[sys.r1.I][end], 1/1337)
        @test isapprox_deftol(current(sol, :I_v1), -1/1337)
    end
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

#=
# TODO: .if/.else/.endif conditional handling needs work for MNA codegen
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
=#

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
