#==============================================================================#
# Standalone Oscillator Test Script
#
# Tests BJT astable multivibrator oscillator.
#
# STATUS: Work in Progress
# The oscillator simulation currently fails because:
# 1. Starting from all-zeros causes numerical issues with BJT exp() equations
# 2. CedarUICOp warmup struggles to find a consistent initial state
# 3. The circuit has no stable DC operating point by design
#
# Possible future solutions:
# - Add startup circuit (PWL source) to bias one side initially
# - Use MOSFET ring oscillator instead (better behaved in subthreshold)
# - Implement ".ic" initial conditions from SPICE
# - Improve CedarUICOp warmup convergence for oscillators
#
# Run with: julia --project=test test/mna/oscillator_test.jl
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!
using CedarSim.MNA: voltage, current
using CedarSim.MNA: VoltageSource, Resistor, Capacitor
using CedarSim.MNA: MNACircuit, MNASolutionAccessor
using CedarSim.MNA: reset_for_restamping!
using VerilogAParser
using SciMLBase
using CedarSim: tran!, dc!, parse_spice_to_mna
using CedarSim.MNA: CedarUICOp
using OrdinaryDiffEq: Rodas5P

#==============================================================================#
# Simple NPN BJT model (Ebers-Moll, same as audio_integration.jl)
#==============================================================================#

va"""
module npnbjt(b, e, c);
    inout b, e, c;
    electrical b, e, c;
    analog I(b,e) <+ 1.0e-14*(exp(V(b,e)/25.0e-3) - 1) - 1.0/(1 + 100.0)*1.0e-14*(exp(V(b,c)/25.0e-3) - 1);
    analog I(c,b) <+ 100.0/(1 + 100.0)*1.0e-14*(exp(V(b,e)/25.0e-3) - 1) - 1.0e-14*(exp(V(b,c)/25.0e-3) - 1);
endmodule
"""

#==============================================================================#
# Circuit Definition
#==============================================================================#

# Astable multivibrator circuit built programmatically with the simple BJT model
function build_astable_multivibrator(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        reset_for_restamping!(ctx)
    end

    # Power supply
    vcc = get_node!(ctx, :vcc)
    stamp!(VoltageSource(params.Vcc; name=:Vcc), ctx, vcc, 0)

    # Q1 nodes
    q1_coll = get_node!(ctx, :q1_coll)
    q1_base = get_node!(ctx, :q1_base)

    # Q2 nodes
    q2_coll = get_node!(ctx, :q2_coll)
    q2_base = get_node!(ctx, :q2_base)

    # Collector resistors (Vcc to collectors)
    stamp!(Resistor(params.Rc; name=:Rc1), ctx, vcc, q1_coll)
    stamp!(Resistor(params.Rc; name=:Rc2), ctx, vcc, q2_coll)

    # Base bias resistors (Vcc to bases)
    stamp!(Resistor(params.Rb; name=:Rb1), ctx, vcc, q1_base)
    stamp!(Resistor(params.Rb; name=:Rb2), ctx, vcc, q2_base)

    # Cross-coupling capacitors (slight asymmetry for startup)
    # C1: Q1 collector to Q2 base
    stamp!(Capacitor(params.C1; name=:C1), ctx, q1_coll, q2_base)
    # C2: Q2 collector to Q1 base
    stamp!(Capacitor(params.C2; name=:C2), ctx, q2_coll, q1_base)

    # BJT Q1: (base, emitter, collector)
    stamp!(npnbjt(), ctx, q1_base, 0, q1_coll;
           _mna_spec_=spec, _mna_x_=x)

    # BJT Q2: (base, emitter, collector)
    stamp!(npnbjt(), ctx, q2_base, 0, q2_coll;
           _mna_spec_=spec, _mna_x_=x)

    return ctx
end

@testset "Oscillator Tests" begin

    @testset "BJT astable multivibrator" begin
        # This test is marked as broken until initialization issues are resolved
        @test_broken false  # Placeholder - oscillator simulation not yet working

        # The circuit definition and component values are preserved for future work:
        # Rc = 470 Ohm (collector load)
        # Rb = 20k Ohm (base bias)
        # C ≈ 0.1uF (coupling capacitor, with slight asymmetry for startup)
        # Vcc = 5V
        # Expected frequency ≈ 1/(1.4 * 20e3 * 0.1e-6) ≈ 357 Hz
        # Expected period ≈ 2.8ms

        # Uncomment below to attempt simulation (currently diverges):
        #=
        circuit = MNACircuit(build_astable_multivibrator;
                             Vcc=5.0, Rc=470.0, Rb=20e3, C1=0.099e-6, C2=0.101e-6)

        expected_period = 1.4 * 20e3 * 0.1e-6
        tspan = (0.0, 15 * expected_period)

        sol = tran!(circuit, tspan; solver=Rodas5P(), abstol=1e-6, reltol=1e-4,
                    initializealg=CedarUICOp(warmup_steps=100, dt=1e-9))
        @test sol.retcode == SciMLBase.ReturnCode.Success
        =#
    end

end
