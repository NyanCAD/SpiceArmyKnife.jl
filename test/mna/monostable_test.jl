#==============================================================================#
# Monostable Multivibrator Test Script (BJT)
#
# Tests monostable (one-shot) multivibrator using vadistiller sp_bjt model.
# This circuit has numerical stability challenges due to BJT exponentials.
#
# Status: For future testing - BJT circuits need solver improvements
#
# Run with: julia --project=test test/mna/monostable_test.jl
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNACircuit, MNASolutionAccessor
using CedarSim.MNA: voltage, assemble!, CedarUICOp
using VerilogAParser
using SciMLBase
using CedarSim: tran!, parse_spice_to_mna
using OrdinaryDiffEq: Rodas5P

#==============================================================================#
# Load sp_bjt model from vadistiller
#==============================================================================#

const bjt_path = joinpath(@__DIR__, "..", "vadistiller", "models", "bjt.va")
const bjt_va = VerilogAParser.parsefile(bjt_path)
if bjt_va.ps.errored
    error("Failed to parse bjt.va")
end
Core.eval(@__MODULE__, CedarSim.make_mna_module(bjt_va))

#==============================================================================#
# Load SPICE netlist
#==============================================================================#

const spice_path = joinpath(@__DIR__, "..", "vadistiller", "circuits", "monostable_multivibrator.spice")
const spice_code = read(spice_path, String)

const monostable_code = parse_spice_to_mna(spice_code;
    circuit_name=:monostable_multivibrator,
    imported_hdl_modules=[sp_bjt_module])
eval(monostable_code)

#==============================================================================#
# Tests
#==============================================================================#

@testset "Monostable Multivibrator Tests" begin

    @testset "sp_bjt monostable one-shot" begin
        circuit = MNACircuit(monostable_multivibrator)

        # Expected pulse width: T ≈ 0.7 * R1 * C1 = 0.7 * 10k * 10u = 70ms
        # Trigger at t=1ms, so pulse ends around t=71ms
        # Simulate for 100ms to capture full pulse
        tspan = (0.0, 100e-3)

        @info "Running monostable transient simulation" tspan

        # Use CedarUICOp for initialization - BJT circuits often lack stable DC point
        sol = tran!(circuit, tspan;
                    solver=Rodas5P(),
                    initializealg=CedarUICOp(warmup_steps=20, dt=1e-12),
                    abstol=1e-8, reltol=1e-6,
                    dtmax=1e-4)

        @test sol.retcode == SciMLBase.ReturnCode.Success

        sys = assemble!(circuit)
        acc = MNASolutionAccessor(sol, sys)

        # Check stable state before trigger (t < 1ms)
        # Q1 ON (collector low), Q2 OFF (collector high ≈ Vcc)
        v_q1_coll_before = voltage(acc, :q1_coll, 0.5e-3)
        v_q2_coll_before = voltage(acc, :q2_coll, 0.5e-3)

        @info "Before trigger" v_q1_coll_before v_q2_coll_before

        # Q1 saturated: collector near 0V
        @test v_q1_coll_before < 1.0
        # Q2 off: collector near Vcc (5V)
        @test v_q2_coll_before > 4.0

        # Check during pulse (t ≈ 10ms, well within 70ms pulse)
        v_q1_coll_during = voltage(acc, :q1_coll, 10e-3)
        v_q2_coll_during = voltage(acc, :q2_coll, 10e-3)

        @info "During pulse" v_q1_coll_during v_q2_coll_during

        # During pulse: Q1 OFF (collector high), Q2 ON (collector low)
        @test v_q1_coll_during > 4.0
        @test v_q2_coll_during < 1.0

        # Check after pulse recovery (t ≈ 90ms, after 70ms pulse ends)
        v_q1_coll_after = voltage(acc, :q1_coll, 90e-3)
        v_q2_coll_after = voltage(acc, :q2_coll, 90e-3)

        @info "After pulse" v_q1_coll_after v_q2_coll_after

        # Back to stable state: Q1 ON, Q2 OFF
        @test v_q1_coll_after < 1.0
        @test v_q2_coll_after > 4.0
    end

end
