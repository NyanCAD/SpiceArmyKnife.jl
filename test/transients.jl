module transient_tests

include("common.jl")

using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using CedarSim.MNA: PWLVoltageSource, PWLCurrentSource, SinVoltageSource
using CedarSim.MNA: voltage, current, MNACircuit
using SciMLBase: ODEProblem

# We'll create a piecewise linear current source that goes through a resistor
#
# The circuit diagram is:
#
#  ┌──┬── +
#  I  R
#  └──┴── -

const i_max = 2
const r_val_pwl = 2
@testset "PWL" begin
    # Helper function that creates the piecewise linear ramp
    # from 0 -> 1 over the course of 1ms -> 9ms
    function pwl_val(t)
        if t < 1e-3
            0.0
        elseif t > 9e-3
            1.0
        else
            (t-1e-3)/8e-3
        end
    end

    # The analytic solution of this circuit is easily calculated in terms of `pwl_val(t)`
    vout_analytic_sol(t) = pwl_val(t) * i_max * r_val_pwl

    # Test using SPICE PWL source
    # SPICE convention: I n+ n- injects current into n-, extracts from n+
    # So i1 0 vout injects current into vout (n-)
    spice_code = """
    * PWL test
    i1 0 vout PWL(1m 0 9m $(i_max))
    R1 vout 0 r=$(r_val_pwl)
    """

    # Parse and generate MNA builder
    ast = SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = CedarSim.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    builder = Base.eval(m, code)

    # Solve for 10ms using MNACircuit API
    tspan = (0.0, 10e-3)
    spec = MNASpec(temp=27.0, mode=:tran, time=0.0)
    circuit = Base.invokelatest(MNACircuit, builder, (;), spec)
    prob = Base.invokelatest(ODEProblem, circuit, tspan)
    sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-6)

    # Get node index for vout
    dc_spec = MNASpec(temp=27.0, mode=:dcop, time=0.0)
    ctx = Base.invokelatest(builder, (;), dc_spec)
    sys = CedarSim.MNA.assemble!(ctx)
    vout_idx = findfirst(n -> n == :vout, sys.node_names)

    # Check that solution matches analytic
    for (i, t) in enumerate(sol.t)
        expected = vout_analytic_sol(t)
        actual = sol.u[i][vout_idx]
        @test isapprox(actual, expected; atol=0.1)
    end

    # Also test using direct MNA API with PWLCurrentSource
    function PWLIRcircuit(params, spec, t::Real=0.0; x=Float64[])
        ctx = MNAContext()
        vout = get_node!(ctx, :vout)
        # PWL: 0->0 at 1ms, 0->i_max at 9ms
        times = [1e-3, 9e-3]
        values = [0.0, Float64(i_max)]
        stamp!(PWLCurrentSource(times, values; name=:I), ctx, vout, 0, t, spec.mode)
        stamp!(Resistor(Float64(r_val_pwl); name=:R), ctx, vout, 0)
        return ctx
    end

    circuit2 = MNACircuit(PWLIRcircuit, (;), MNASpec(temp=27.0))
    prob2 = ODEProblem(circuit2, tspan)
    sol2 = OrdinaryDiffEq.solve(prob2, Rodas5P(); reltol=1e-6, abstol=1e-6, maxiters=100000)

    # Check direct API matches
    for (i, t) in enumerate(sol2.t)
        expected = vout_analytic_sol(t)
        actual = sol2.u[i][1]  # vout is node 1
        @test isapprox(actual, expected; atol=0.1)
    end
end

#=
@testset "PWL derivative" begin
    # This test requires Diffractor which may not be available
    # Skip for now - the PWL interpolation is tested above
end
=#

# Create a third-order Butterworth filter, according to https://en.wikipedia.org/wiki/Butterworth_filter#Example
# The circuit diagram is:
#
#  ┌─L1─┬─L3─┬── +
#  V    C2   R4
#  └────┴────┴── -
#
# We take the simple example, with values:
#  L1 = 3/2 H
#  C2 = 4/3 F
#  L3 = 1/2 H
#  R4 = 1 Ω
#
# This yields a transfer function of:
#   H(s) = 1/(1 + 2s + 2s^2 + s^3)
# The magnitude of the steady-state response is:
#   G(ω) = 1/sqrt(1 + ω^6)
# so at ω=1 we should get 1/2 gain (note, ω is supplied in radians, so the actual value
# will be divided by 2π!)
#
# If we drive this system with a sinusoidal input with frequency 1, we get the following transfer function:
#   H(s) = 1/(s^2 + 1) * 1/(1 + 2s + 2s^2 + s^3)
# This corresponds to a time-domain solution via the inverse laplace transform of:
#   vout(t) = (e^(-t) - sin(t) - cos(t))/2 + (2 * sin((sqrt(3) * t)/2))/(sqrt(3) * sqrt(e^t))
const L1_val = 3/2
const C2_val = 4/3
const L3_val = 1/2
const R4_val = 1
const ω_val = 1

@testset "Butterworth Filter" begin
    # Helper function to calculate RMS of a signal
    rms(sig) = sqrt(sum(sig.^2)/length(sig))

    vout_analytic_sol(t) = (exp(-t) - sin(t) - cos(t))/2 + (2 * sin((sqrt(3) * t)/2))/(sqrt(3) * sqrt(exp(t)))

    # Test using SPICE SIN source
    spice_code = """
    *Third order low pass filter, butterworth, with ω_c = 1

    V1 vin 0 SIN(0, 1, $(ω_val/2π))
    L1 vin n1 $(L1_val)
    C2 n1 0 $(C2_val)
    L3 n1 vout $(L3_val)
    R4 vout 0 $(R4_val)
    """

    # Parse and generate MNA builder
    ast = SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = CedarSim.make_mna_circuit(ast)
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    builder = Base.eval(m, code)

    # This is a very low-frequency circuit; simulate for a long enough time
    # that we can get a nice steady-state response in the end
    tspan = (0.0, 100.0)

    # Get initial conditions (all zeros for inductors/capacitors)
    spec = CedarSim.MNA.MNASpec(temp=27.0, mode=:dcop, time=0.0)
    ctx = Base.invokelatest(builder, (;), spec)
    sys = CedarSim.MNA.assemble!(ctx)

    # Use MNACircuit API with zero initial conditions (capacitor/inductor start uncharged)
    circuit = Base.invokelatest(MNACircuit, builder, (;), MNASpec(temp=27.0))
    n = CedarSim.MNA.system_size(circuit)
    u0 = zeros(n)
    prob = Base.invokelatest(ODEProblem, circuit, tspan; u0=u0)
    sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-6, maxiters=100000)

    # Get node index for vout
    vout_idx = findfirst(n -> n == :vout, sys.node_names)

    # Check that solution matches analytic at sample points
    @test isapprox(sol.u[1][vout_idx], vout_analytic_sol(sol.t[1]); atol=0.1)
    @test isapprox(sol.u[end][vout_idx], vout_analytic_sol(sol.t[end]); atol=0.1)

    # Also assert that the RMS of the steady-state portion is approximately correct
    # At ω=1, gain should be ~0.5
    steady_state_vout = [sol.u[i][vout_idx] for i in (length(sol.u)÷2):length(sol.u)]
    @test isapprox(rms(steady_state_vout), 0.5; atol=0.15, rtol=0.15)

    # Test using direct MNA API with SinVoltageSource
    function butterworth_circuit(params, spec, t::Real=0.0; x=Float64[])
        ctx = MNAContext()
        vin = get_node!(ctx, :vin)
        n1 = get_node!(ctx, :n1)
        vout = get_node!(ctx, :vout)

        # SIN source: V(t) = sin(ω*t)
        stamp!(SinVoltageSource(0.0, 1.0, ω_val/2π; name=:V), ctx, vin, 0, t, spec.mode)
        stamp!(Inductor(L1_val; name=:L1), ctx, vin, n1)
        stamp!(Capacitor(C2_val; name=:C2), ctx, n1, 0)
        stamp!(Inductor(L3_val; name=:L3), ctx, n1, vout)
        stamp!(Resistor(R4_val; name=:R4), ctx, vout, 0)
        return ctx
    end

    circuit2 = MNACircuit(butterworth_circuit, (;), MNASpec(temp=27.0))
    n2 = CedarSim.MNA.system_size(circuit2)
    u0_2 = zeros(n2)
    prob2 = ODEProblem(circuit2, tspan; u0=u0_2)
    sol2 = OrdinaryDiffEq.solve(prob2, Rodas5P(); reltol=1e-6, abstol=1e-6, maxiters=100000)

    # Get vout index from direct API circuit
    ctx2 = butterworth_circuit((;), CedarSim.MNA.MNASpec(temp=27.0, mode=:dcop, time=0.0), 0.0)
    sys2 = CedarSim.MNA.assemble!(ctx2)
    vout_idx2 = findfirst(n -> n == :vout, sys2.node_names)

    # Check direct API also matches
    @test isapprox(sol2.u[1][vout_idx2], vout_analytic_sol(sol2.t[1]); atol=0.1)
    @test isapprox(sol2.u[end][vout_idx2], vout_analytic_sol(sol2.t[end]); atol=0.1)

    steady_state_vout2 = [sol2.u[i][vout_idx2] for i in (length(sol2.u)÷2):length(sol2.u)]
    @test isapprox(rms(steady_state_vout2), 0.5; atol=0.15, rtol=0.15)
end

end # module transient_tests
