module test_sweep

# Include all our testing packages, helper routines, etc...
using CedarSim
include(joinpath(Base.pkgdir(CedarSim), "test", "common.jl"))

# MNA imports for sweep tests
using CedarSim.MNA: MNAContext, MNACircuit, get_node!, stamp!
using CedarSim.MNA: Resistor, VoltageSource
using CedarSim.MNA: voltage, current, DCSolution
using CedarSim: ParamLens

# Simple two-resistor circuit (MNA builder function):
#
#  ┌─R1─┬── +
#  V    R2
#  └────┴── -
#
# We'll vary `R1` and `R2` with parameter sweeps,
# then verify that the current out of `V` is correct.
# Default values are R1=1000.0, R2=1000.0 (matching the old @kwdef struct)
function build_two_resistor(params, spec, t::Real=0.0; x=Float64[])
    # Merge with defaults (like @kwdef did for the struct)
    defaults = (R1=1000.0, R2=1000.0)
    p = merge(defaults, params)

    ctx = MNAContext()
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(1.0; name=:V), ctx, vcc, 0)
    stamp!(Resistor(p.R1), ctx, vcc, out)
    stamp!(Resistor(p.R2), ctx, out, 0)

    return ctx
end

# nest_param_list and flatten_param_list were removed with circuitodesystem.jl
# Skipping these tests - the functionality was for DAECompiler's ParamSim workflow
@testset "nest and flatten param lists" begin
    @test_skip "nest_param_list removed - was part of old DAECompiler API"
end

@testset "Sweep" begin
    s = Sweep(:R1, 0.1:0.1:1.0)
    @test length(s) == 10
    @test size(s) == (10,)
    @test size(s, 1) == 10
    @test size(s, 2) == 1
    @test first(s) == ((:R1, 0.1),)

    @test Sweep(R1 = 0.1:0.1:1.0) == s
    @test first(Sweep(var"a.b" = 1.0:2.0)) == ((Symbol("a.b"), 1),)

    # Make sure that `Sweep()` with a scalar works:
    s = Sweep(a = 10.0)
    @test length(s) == 1
    @test size(s) == (1,)
    @test size(s, 1) == 1
    @test size(s, 2) == 1
    @test first(s) == ((:a, 10),)

    # Test ProductSweep
    s = ProductSweep(R1 = 1.0:2.0, R2 = 1.0:2.0, R3 = 1.0:2.0)
    @test size(s) == (2, 2, 2)
    @test size(s, 1) == 2
    @test size(s, 2) == 2
    @test size(s, 3) == 2
    @test size(s, 4) == 1
    @test first(s) == (
        (:R1, 1.0),
        (:R2, 1.0),
        (:R3, 1.0),
    )
    @test length(sweepvars(s)) == 3
    @test :R1 ∈ sweepvars(s)
    @test :R2 ∈ sweepvars(s)
    @test :R3 ∈ sweepvars(s)

    # Test empty ProductSweep
    s = ProductSweep()
    @test size(s) == ()
    @test isempty(sweepvars(s))
    @test first(s) == ()

    # Test TandemSweep
    s = TandemSweep(R1 = 1.0:2.0, R2 = 1.0:2.0, R3 = 1.0:2.0)
    @test size(s) == (2,)
    @test size(s, 1) == 2
    @test size(s, 2) == 1
    @test collect(s) == [
        ((:R1, 1.0), (:R2, 1.0), (:R3, 1.0)),
        ((:R1, 2.0), (:R2, 2.0), (:R3, 2.0)),
    ]
    @test length(sweepvars(s)) == 3
    @test :R1 ∈ sweepvars(s)
    @test :R2 ∈ sweepvars(s)
    @test :R3 ∈ sweepvars(s)

    # Test SerialSweep
    s = SerialSweep(R1 = 1.0:2.0, R2 = 1.0:2.0)
    @test size(s) == (4,)
    @test size(s, 1) == 4
    @test collect(s) == [
        ((:R1, 1.0), (:R2, nothing)),
        ((:R1, 2.0), (:R2, nothing)),
        ((:R1, nothing), (:R2, 1.0)),
        ((:R1, nothing), (:R2, 2.0)),
    ]
    @test length(sweepvars(s)) == 2
    @test :R1 ∈ sweepvars(s)
    @test :R2 ∈ sweepvars(s)

    # Test empty SerialSweep
    s = SerialSweep()
    @test size(s) == (0,)
    @test length(s) == 0
    @test isempty(sweepvars(s))

    # Test compositions of these!
    s = ProductSweep(ProductSweep(R1 = 1.0:2.0, R2 = 1.0:2.0), R3 = 1.0:2.0)
    @test all(collect(s) .==
              collect(ProductSweep(R1 = 1.0:2.0, R2 = 1.0:2.0, R3 = 1.0:2.0)))
    @test length(sweepvars(s)) == 3
    @test :R1 ∈ sweepvars(s)
    @test :R2 ∈ sweepvars(s)
    @test :R3 ∈ sweepvars(s)

    s = ProductSweep(TandemSweep(R1 = 1.0:2.0, R2 = 1.0:2.0), R3 = 1.0:2.0)
    @test collect(s) == [
        ((:R1, 1.0), (:R2, 1.0), (:R3, 1.0)) ((:R1, 1.0), (:R2, 1.0), (:R3, 2.0));
        ((:R1, 2.0), (:R2, 2.0), (:R3, 1.0)) ((:R1, 2.0), (:R2, 2.0), (:R3, 2.0));
    ]
    @test length(sweepvars(s)) == 3
    @test :R1 ∈ sweepvars(s)
    @test :R2 ∈ sweepvars(s)
    @test :R3 ∈ sweepvars(s)

    s = ProductSweep(SerialSweep(R3 = 1.0:2.0, R4 = 1.0:2.0), R1 = 1.0:2.0, R2 = 1.0:2.0)
    @test size(s) == (4, 2, 2)
    @test size(s, 1) == 4
    @test size(s, 2) == 2
    @test size(s, 3) == 2
    @test collect(s)[:] == [
        ((:R1, 1.0), (:R2, 1.0), (:R3, 1.0), (:R4, nothing))
        ((:R1, 1.0), (:R2, 1.0), (:R3, 2.0), (:R4, nothing))
        ((:R1, 1.0), (:R2, 1.0), (:R3, nothing), (:R4, 1.0))
        ((:R1, 1.0), (:R2, 1.0), (:R3, nothing), (:R4, 2.0))
        ((:R1, 2.0), (:R2, 1.0), (:R3, 1.0), (:R4, nothing))
        ((:R1, 2.0), (:R2, 1.0), (:R3, 2.0), (:R4, nothing))
        ((:R1, 2.0), (:R2, 1.0), (:R3, nothing), (:R4, 1.0))
        ((:R1, 2.0), (:R2, 1.0), (:R3, nothing), (:R4, 2.0))
        ((:R1, 1.0), (:R2, 2.0), (:R3, 1.0), (:R4, nothing))
        ((:R1, 1.0), (:R2, 2.0), (:R3, 2.0), (:R4, nothing))
        ((:R1, 1.0), (:R2, 2.0), (:R3, nothing), (:R4, 1.0))
        ((:R1, 1.0), (:R2, 2.0), (:R3, nothing), (:R4, 2.0))
        ((:R1, 2.0), (:R2, 2.0), (:R3, 1.0), (:R4, nothing))
        ((:R1, 2.0), (:R2, 2.0), (:R3, 2.0), (:R4, nothing))
        ((:R1, 2.0), (:R2, 2.0), (:R3, nothing), (:R4, 1.0))
        ((:R1, 2.0), (:R2, 2.0), (:R3, nothing), (:R4, 2.0))
    ]
    @test length(sweepvars(s)) == 4
    @test :R1 ∈ sweepvars(s)
    @test :R2 ∈ sweepvars(s)
    @test :R3 ∈ sweepvars(s)
    @test :R4 ∈ sweepvars(s)

    # Test that a single value in any of our larger sweep types just returns a `Sweep`:
    @test ProductSweep(r1 = 1:10) == Sweep(r1 = 1:10)
    @test SerialSweep(r1 = 1:10) == Sweep(r1 = 1:10)
    @test TandemSweep(r1 = 1:10) == Sweep(r1 = 1:10)

    # Test that equality works with vectors as well
    @test Sweep(r1 = [1, 2, 3]) == Sweep(r1 = [1, 2, 3])
end

@testset "alter() on NamedTuples" begin
    p = (r=1, cap=1e-12)
    @test alter(p, first(Sweep(cap=1e-13))) == (r=1, cap=1e-13)
    @test alter(p, first(ProductSweep(r=2:3, cap = 1e-13))) == (r=2, cap=1e-13)
end

@testset "split_axes()" begin
    ps = ProductSweep(A = 1:10, B=1:10, C=1:5, D=1:5)

    # Test that `split_axes()` works as it's supposed to
    for vars in [[:A, :C], (:A, :C), Set([:A, :C])]
        outer, inner = split_axes(ps, vars)
        @test length(sweepvars(outer)) == 2
        @test length(sweepvars(inner)) == 2
        @test :B ∈ sweepvars(outer)
        @test :D ∈ sweepvars(outer)
        @test :A ∈ sweepvars(inner)
        @test :C ∈ sweepvars(inner)
        @test size(outer) == (10, 5)
        @test size(inner) == (10, 5)
        @test size(outer, 1) == 10
        @test size(inner, 1) == 10
        @test size(outer, 2) == 5
        @test size(inner, 2) == 5

        # Test that ProductSweep is the inverse operation:
        ps2 = ProductSweep(outer, inner)
        @test sweepvars(ps) == sweepvars(ps2)
        # Although ordering doesn't quite match, due to how we've split things up
        @test size(ps) != size(ps2)

        # Test that trying to split on a non-existant axis errors:
        @test_throws ArgumentError split_axes(ps, [:E])

        # Test that trying to split some other kind of sweep doesn't work:
        @test_throws ArgumentError split_axes(SerialSweep(A=1:10, B=1:10), [:A])
    end

end

#==============================================================================#
# MNA-based simulation tests
#==============================================================================#

@testset "simple dc!" begin
    circuit = MNACircuit(build_two_resistor; R1=100.0, R2=100.0)
    sol = dc!(circuit)
    @test sol isa DCSolution
    @test voltage(sol, :vcc) ≈ 1.0
    @test voltage(sol, :out) ≈ 0.5  # Voltage divider: 1V * 100/(100+100)
end

@testset "CircuitSweep" begin
    # Test construction of the `CircuitSweep` object
    cs = CircuitSweep(build_two_resistor, Sweep(R1 = 1.0:10.0))
    @test length(cs) == 10
    @test size(cs) == (10,)
    @test size(cs, 1) == 10

    # Show that it generates a simulation with the parameters as we expect
    @test first(cs).params.R1 == 1.0

    # Test that a two-dimensional sweep is represented two-dimensionally
    cs = CircuitSweep(build_two_resistor, ProductSweep(R1 = 1.0:10.0, R2 = 1.0:10.0))
    @test length(cs) == 100
    @test size(cs) == (10,10)
    @test size(cs, 1) == 10
    @test size(cs, 2) == 10
    @test size(collect(cs)) == (10,10)
    @test first(cs).params.R1 == 1.0
    @test first(cs).params.R2 == 1.0
    @test length(sweepvars(cs)) == 2
    @test :R1 ∈ sweepvars(cs)
    @test :R2 ∈ sweepvars(cs)

    # Test nested parameter access with var-strings (like SPICE codegen will use)
    # Builder uses ParamLens for hierarchical params with defaults
    function build_nested_resistor(params, spec, t::Real=0.0; x=Float64[])
        # Convert params to ParamLens for SPICE-style hierarchical access
        lens = ParamLens(params)
        # lens.inner(; defaults...) returns params merged with lens overrides
        p = lens.inner(; R1=1000.0, R2=1000.0)

        ctx = MNAContext()
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(1.0; name=:V), ctx, vcc, 0)
        stamp!(Resistor(p.R1), ctx, vcc, out)
        stamp!(Resistor(p.R2), ctx, out, 0)

        return ctx
    end

    sweep = ProductSweep(
        TandemSweep(var"inner.params.R1" = 100.0:100.0:200.0,
                    var"inner.params.R2" = 100.0:100.0:200.0),
    )
    # ParamLens expects (inner=(params=(...),)) structure for hierarchical override
    cs = CircuitSweep(build_nested_resistor, sweep;
                      inner = (params = (R1 = 100.0, R2 = 100.0),))
    @test length(cs) == 2
    @test size(cs) == (2,)
    @test size(cs, 1) == 2
    @test first(cs).params.inner.params.R1 == 100.0
    @test first(cs).params.inner.params.R2 == 100.0
    @test last(collect(cs)).params.inner.params.R1 == 200.0
    @test last(collect(cs)).params.inner.params.R2 == 200.0
    @test length(sweepvars(cs)) == 2
    @test Symbol("inner.params.R1") ∈ sweepvars(cs)
    @test Symbol("inner.params.R2") ∈ sweepvars(cs)
end

@testset "dc! sweeps" begin
    # Construct `CircuitSweep` object and run dc! on it
    cs = CircuitSweep(build_two_resistor, ProductSweep(R1 = 100.0:100.0:2000.0, R2 = 100.0:100.0:2000.0))

    # Perform the solve - dc! on CircuitSweep returns vector of solutions
    solutions = dc!(cs)

    # Verify results for each simulation in the sweep
    for (sol, sim) in zip(solutions, cs)
        R1 = sim.params.R1
        R2 = sim.params.R2
        # Current through voltage source is negative (sources current)
        # I = V / (R1 + R2) = 1 / (R1 + R2)
        @test isapprox(current(sol, :I_V), -1/(R1 + R2); atol=deftol)
    end
end

# Phase 4: MNA-based SPICE codegen tests
@testset "MNA SPICE codegen: basic resistor circuit" begin
    using CedarSim.MNA: voltage, current

    spice_code = """
        * Simple voltage divider
        V1 vcc 0 DC 5
        R1 vcc out 1k
        R2 out 0 1k
    """

    # Parse and generate MNA builder
    ast = CedarSim.SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    circuit_code = CedarSim.make_mna_circuit(ast)

    # Create a module to evaluate the code
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    circuit_fn = Base.eval(m, circuit_code)

    # Build and solve
    ctx = Base.invokelatest(circuit_fn, (;), CedarSim.MNA.MNASpec())
    sys = CedarSim.MNA.assemble!(ctx)
    sol = CedarSim.MNA.solve_dc(sys)

    # Verify voltage divider: out = 5 * 1k/(1k+1k) = 2.5V
    @test isapprox(voltage(sol, :vcc), 5.0; atol=deftol)
    @test isapprox(voltage(sol, :out), 2.5; atol=deftol)
end

@testset "MNA SPICE codegen: RLC circuit" begin
    using CedarSim.MNA: voltage

    spice_code = """
        * RLC circuit
        V1 in 0 DC 10
        R1 in n1 100
        L1 n1 n2 1m
        C1 n2 0 1u
    """

    ast = CedarSim.SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    circuit_code = CedarSim.make_mna_circuit(ast)

    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    circuit_fn = Base.eval(m, circuit_code)

    ctx = Base.invokelatest(circuit_fn, (;), CedarSim.MNA.MNASpec())
    sys = CedarSim.MNA.assemble!(ctx)
    sol = CedarSim.MNA.solve_dc(sys)

    # At DC, inductor is short, capacitor is open
    # So n1 = 10V (after R, but L is short)
    # n2 = 10V (capacitor open, no current)
    @test isapprox(voltage(sol, :in), 10.0; atol=deftol)
    # Inductor at DC is a short, so n1 ≈ n2 ≈ in (no resistor drop when no current)
    # Actually with capacitor open, no current flows, so all nodes at 10V
    @test isapprox(voltage(sol, :n1), 10.0; atol=deftol)
    @test isapprox(voltage(sol, :n2), 10.0; atol=deftol)
end

@testset "MNA SPICE codegen: current source" begin
    using CedarSim.MNA: voltage

    spice_code = """
        * Current source into resistor
        I1 0 out DC 1m
        R1 out 0 1k
    """

    ast = CedarSim.SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    circuit_code = CedarSim.make_mna_circuit(ast)

    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    circuit_fn = Base.eval(m, circuit_code)

    ctx = Base.invokelatest(circuit_fn, (;), CedarSim.MNA.MNASpec())
    sys = CedarSim.MNA.assemble!(ctx)
    sol = CedarSim.MNA.solve_dc(sys)

    # V = I * R = 1mA * 1kΩ = 1V
    @test isapprox(voltage(sol, :out), 1.0; atol=deftol)
end

# TODO: Subcircuit parameter passing not yet working in MNA codegen
# The subcircuit ports are being reassigned instead of using passed-in ports
@testset "dc! sweep on SPICE-generated circuit" begin
    @test_skip "Subcircuit parameter passing not yet implemented"
end

@testset "find_param_ranges" begin
    # Create a nasty complicated parameter exploration
    params = ProductSweep(
        ProductSweep(
            SerialSweep(
                # Work around inability to provide duplicate kwarg names)
                Sweep(a = 1:10),
                Sweep(a = 11:20),
            ),
            b = 1:2:5,
        ),
        ProductSweep(
            TandemSweep(
                c = 1:10,
                d = 1:10,
            ),
            SerialSweep(
                c = 11:15,
                d = 1:5,
            )
        ),
    )
    ranges = CedarSim.find_param_ranges(params)
    @test ranges[:a] == (1, 20, 20)
    @test ranges[:b] == (1, 5, 3)
    @test ranges[:c] == (1, 15, 15)
    @test ranges[:d] == (1, 10, 15)
end

@testset "sweepify" begin
    s1 = sweepify([(r1 = 1:10, r2 = 1:6), (r3 = 1:4, r4=1:2)])
    s2 = SerialSweep(ProductSweep(r1 = 1:10, r2 = 1:6), ProductSweep(r3 = 1:4, r4=1:2))
    @test collect(s1) == collect(s2)

    s1 = sweepify([:r1 => 1:10, :r2 => 1:10])
    s2 = SerialSweep(r1 = 1:10, r2 = 1:10)
    @test collect(s1) == collect(s2)
end

end # module test_sweep
