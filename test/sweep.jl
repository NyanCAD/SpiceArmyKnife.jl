module test_sweep

# Include all our testing packages, helper routines, etc...
using CedarSim
include(joinpath(Base.pkgdir(CedarSim), "test", "common.jl"))

# MNA imports for sweep tests
using CedarSim.MNA: MNAContext, MNASim, get_node!, stamp!
using CedarSim.MNA: Resistor, VoltageSource
using CedarSim.MNA: voltage, current, DCSolution

# Simple two-resistor circuit (MNA builder function):
#
#  ┌─R1─┬── +
#  V    R2
#  └────┴── -
#
# We'll vary `R1` and `R2` with parameter sweeps,
# then verify that the current out of `V` is correct.
function build_two_resistor(params, spec)
    ctx = MNAContext()
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(1.0; name=:V), ctx, vcc, 0)
    stamp!(Resistor(params.R1), ctx, vcc, out)
    stamp!(Resistor(params.R2), ctx, out, 0)

    return ctx
end

using CedarSim: nest_param_list, flatten_param_list
@testset "nest and flatten param lists" begin
    # Test that it works with a Dict
    param_list_dict = Dict(
        :R1 => 1,
        Symbol("x1.R3") => 2,
        Symbol("x1.x2.R1") => 3,
        Symbol("x1.x2.R2") => 4,
    )
    param_list_tuple = (
        (:R1, 1.0),
        (Symbol("x1.R3"), 2),
        (Symbol("x1.x2.R1"), 3),
        (Symbol("x1.x2.R2"), 4),
    )
    # Named tuples seem to have their own sorting method
    param_nested_namedtuple = (
        R1 = 1,
        x1 = (
            x2 = (R2 = 4, R1 = 3),
            R3 = 2,
        ),
    )
    @test nest_param_list(param_list_dict) == param_nested_namedtuple
    @test nest_param_list(param_list_tuple) == param_nested_namedtuple
    @test flatten_param_list(param_nested_namedtuple) == param_list_tuple
    @test flatten_param_list(nest_param_list(param_list_tuple)) == param_list_tuple

    # Trying to assign a prefix of another value is an error
    @test_throws CedarSim.WrappedCedarException{<:ArgumentError} CedarSim.nest_param_list((
        (Symbol("x1"), 1),
        (Symbol("x1.R1"), 2),
    ))

    # Double-assigning a value is an error:
    @test_throws CedarSim.WrappedCedarException{<:ArgumentError} CedarSim.nest_param_list((
        (Symbol("x1"), 1),
        (Symbol("x1"), 2),
    ))
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
    sim = MNASim(build_two_resistor; R1=100.0, R2=100.0)
    sol = dc!(sim)
    @test sol isa DCSolution
    @test voltage(sol, :vcc) ≈ 1.0
    @test voltage(sol, :out) ≈ 0.5  # Voltage divider: 1V * 100/(100+100)
end

@testset "CircuitSweep" begin
    # Test construction of the `CircuitSweep` object
    cs = CircuitSweep(build_two_resistor, Sweep(R1 = 1.0:10.0); R2=1000.0)
    @test length(cs) == 10
    @test size(cs) == (10,)
    @test size(cs, 1) == 10

    # Show that it generates a simulation with the parameters as we expect
    @test first(cs).params.R1 == 1.0
    @test first(cs).params.R2 == 1000.0

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

# Test nested parameter access with var-strings
# Builder that uses nested params: params.inner.R1, params.inner.R2
function build_nested_resistor(params, spec)
    ctx = MNAContext()
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(1.0; name=:V), ctx, vcc, 0)
    stamp!(Resistor(params.inner.R1), ctx, vcc, out)
    stamp!(Resistor(params.inner.R2), ctx, out, 0)

    return ctx
end

@testset "nested params in CircuitSweep" begin
    # Test that var-strings reach into nested parameter structures
    sweep = ProductSweep(
        TandemSweep(var"inner.R1" = 100.0:100.0:200.0,
                    var"inner.R2" = 100.0:100.0:200.0),
    )

    # Default params provide the nested structure
    cs = CircuitSweep(build_nested_resistor, sweep;
                      inner = (R1 = 100.0, R2 = 100.0))

    @test length(cs) == 2
    @test first(cs).params.inner.R1 == 100.0
    @test first(cs).params.inner.R2 == 100.0
    @test last(collect(cs)).params.inner.R1 == 200.0
    @test last(collect(cs)).params.inner.R2 == 200.0

    @test length(sweepvars(cs)) == 2
    @test Symbol("inner.R1") ∈ sweepvars(cs)
    @test Symbol("inner.R2") ∈ sweepvars(cs)

    # Verify the circuit works
    solutions = dc!(cs)
    for (sol, sim) in zip(solutions, cs)
        R1 = sim.params.inner.R1
        R2 = sim.params.inner.R2
        @test isapprox(current(sol, :I_V), -1/(R1 + R2); atol=deftol)
    end
end

# TODO: Re-enable when SPICE codegen is ported to MNA backend
# @testset "dc! sweep on spice code" begin
#     spice_code =
#     """
#         * Parameter scoping test
#
#         .subckt subcircuit1 vss gnd
#         .param r_load=1
#         r1 vss gnd 'r_load'
#         .ends
#
#         .param v_in=1
#         x1 vss 0 subcircuit1
#         v1 vss 0 'v_in'
#     """
#     circuit_code = CedarSim.make_spectre_circuit(
#         CedarSim.SpectreNetlistParser.SPICENetlistParser.SPICENetlistCSTParser.parse(spice_code),
#     );
#     circuit = eval(circuit_code);
#     cs = CircuitSweep(circuit, ProductSweep(v_in = 1.0:10.0, var"x1.r_load" = 1.0:10.0))
#     solutions = dc!(cs; abstol=deftol, reltol=deftol)
#
#     for sol in solutions
#         params = sol.prob.p.params
#         v_in = params.params.v_in
#         r_load = params.x1.params.r_load
#         @test isapprox_deftol(v_in/r_load, sol[cs.sys.x1.r1.I][end])
#     end
# end

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
