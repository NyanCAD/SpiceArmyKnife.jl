#==============================================================================#
# Astable Multivibrator Test with sp_bjt
#
# TODO: This test is a work in progress. The astable multivibrator requires
# special initialization handling because:
# 1. It has no stable DC operating point (free-running oscillator)
# 2. CedarUICOp warmup struggles with sp_bjt internal nodes
# 3. The symmetric DC point causes both BJTs to saturate
#
# The CMOS ring oscillator (using sp_mos1) works with CedarUICOp because:
# - Much faster timescales (femtoseconds vs milliseconds)
# - MOSFET behavior is more linear near threshold
#
# Approaches to try:
# 1. Improve CedarUICOp warmup for BJT circuits
# 2. Use initial conditions that break symmetry
# 3. Add a startup perturbation (PWL or SIN source)
# 4. Use source stepping from 0 to Vcc
#
# Run with: julia --project=test test/mna/astable_bjt_test.jl
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNACircuit, MNASolutionAccessor, voltage, assemble!
using CedarSim.MNA: CedarDCOp, CedarUICOp
using CedarSim: tran!, parse_spice_to_mna
using VerilogAParser
using SciMLBase
using Sundials: IDA

# Load sp_bjt model directly from VA file
println("Loading sp_bjt model...")
const bjt_path = joinpath(@__DIR__, "..", "..", "models", "VADistillerModels.jl", "va", "bjt.va")
const bjt_va = VerilogAParser.parsefile(bjt_path)
if bjt_va.ps.errored
    error("Failed to parse bjt.va")
end
Core.eval(@__MODULE__, CedarSim.make_mna_module(bjt_va))

# Astable multivibrator circuit (free-running oscillator)
# Frequency ~ 1/(1.4*R*C) = 1/(1.4*10k*1u) ≈ 71 Hz
const astable_code = parse_spice_to_mna("""
* Astable Multivibrator (Free-Running Oscillator) using sp_bjt
Vcc vcc 0 DC 5.0
R1 vcc q1_base 10k
R2 vcc q2_base 10k
Rc1 vcc q1_coll 1k
Rc2 vcc q2_coll 1k
C1 q1_coll q2_base 1u
C2 q2_coll q1_base 1u
XQ1 q1_coll q1_base 0 0 sp_bjt bf=100 is=1e-15
XQ2 q2_coll q2_base 0 0 sp_bjt bf=100 is=1e-15
.END
"""; circuit_name=:astable_multivibrator, imported_hdl_modules=[sp_bjt_module])
eval(astable_code)

@testset "sp_bjt Astable Multivibrator (Oscillator)" skip=true begin
    # This test is skipped until CedarUICOp improvements are made
    circuit = MNACircuit(astable_multivibrator)
    tspan = (0.0, 50e-3)  # 50ms to capture multiple oscillation cycles

    @info "Running sp_bjt astable transient simulation" tspan

    # TODO: Need to figure out proper initialization for BJT astable
    # CedarUICOp fails with sp_bjt internal nodes
    # CedarDCOp finds metastable point where both BJTs are saturated
    sol = tran!(circuit, tspan;
                solver=IDA(),
                initializealg=CedarDCOp(),
                abstol=1e-6, reltol=1e-4)

    @info "Transient result" sol.retcode

    @test sol.retcode == SciMLBase.ReturnCode.Success

    sys = assemble!(circuit)
    acc = MNASolutionAccessor(sol, sys)

    # Sample Q1 collector voltage in last half (after startup transient)
    times = range(25e-3, 50e-3; length=500)
    V_q1 = [voltage(acc, :q1_coll, t) for t in times]

    q1_min, q1_max = extrema(V_q1)

    @info "Q1 collector voltage range" q1_min q1_max swing=(q1_max - q1_min)

    # Verify significant voltage swing (oscillation occurring)
    @test (q1_max - q1_min) > 2.0  # At least 2V swing
    @test q1_max > 3.5  # Near Vcc when Q1 OFF
    @test q1_min < 1.5  # Near ground when Q1 ON (saturated)

    # Count zero crossings to verify oscillation
    midpoint = (q1_max + q1_min) / 2
    crossings = sum(
        (V_q1[i-1] < midpoint && V_q1[i] >= midpoint) ||
        (V_q1[i-1] > midpoint && V_q1[i] <= midpoint)
        for i in 2:length(V_q1)
    )

    @info "Oscillation verification" midpoint crossings

    # Expected frequency ~71 Hz, so ~1.8 cycles in 25ms = ~3-4 crossings
    @test crossings >= 2  # At least one full oscillation cycle
end
