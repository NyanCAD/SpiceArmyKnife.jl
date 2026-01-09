#==============================================================================#
# Standalone Oscillator Test Script
#
# Tests CMOS ring oscillator using vadistiller sp_mos1 model.
# Run with: julia --project=test test/mna/oscillator_test.jl
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!
using CedarSim.MNA: voltage, current
using CedarSim.MNA: VoltageSource, Resistor, Capacitor
using CedarSim.MNA: MNACircuit, MNASolutionAccessor
using CedarSim.MNA: reset_for_restamping!, CedarUICOp
using VerilogAParser
using SciMLBase
using CedarSim: tran!, parse_spice_to_mna
using OrdinaryDiffEq: Rodas5P

#==============================================================================#
# Load sp_mos1 model from vadistiller
#==============================================================================#

const mos1_path = joinpath(@__DIR__, "..", "vadistiller", "models", "mos1.va")
const mos1_va = VerilogAParser.parsefile(mos1_path)
if mos1_va.ps.errored
    error("Failed to parse mos1.va")
end
Core.eval(@__MODULE__, CedarSim.make_mna_module(mos1_va))

#==============================================================================#
# Ring Oscillator SPICE Netlist
#
# 3-stage CMOS ring oscillator:
# - Each stage is an inverter (NMOS + PMOS)
# - Output of each stage drives the next
# - Output of last stage feeds back to first stage
#
# Oscillation frequency ≈ 1 / (2 * n * t_delay)
# where n = number of stages, t_delay = inverter delay
#==============================================================================#

const ring_osc_code = parse_spice_to_mna("""
* 3-stage CMOS Ring Oscillator
* Uses sp_mos1 with minimal parameters

* Power supply
Vdd vdd 0 DC 3.3

* Stage 1: Inverter (in1 -> out1)
* PMOS: type=-1, NMOS: type=1
XMP1 out1 in1 vdd vdd sp_mos1 type=-1 vto=-0.7 kp=50e-6 w=2e-6 l=1e-6
XMN1 out1 in1 0 0 sp_mos1 type=1 vto=0.7 kp=100e-6 w=1e-6 l=1e-6

* Stage 2: Inverter (out1 -> out2)
XMP2 out2 out1 vdd vdd sp_mos1 type=-1 vto=-0.7 kp=50e-6 w=2e-6 l=1e-6
XMN2 out2 out1 0 0 sp_mos1 type=1 vto=0.7 kp=100e-6 w=1e-6 l=1e-6

* Stage 3: Inverter (out2 -> in1) - feedback
XMP3 in1 out2 vdd vdd sp_mos1 type=-1 vto=-0.7 kp=50e-6 w=2e-6 l=1e-6
XMN3 in1 out2 0 0 sp_mos1 type=1 vto=0.7 kp=100e-6 w=1e-6 l=1e-6

* Load capacitors (represent gate capacitance and wiring)
C1 out1 0 10f
C2 out2 0 10f
C3 in1 0 10f

.END
"""; circuit_name=:ring_oscillator, imported_hdl_modules=[sp_mos1_module])
eval(ring_osc_code)

#==============================================================================#
# Tests
#==============================================================================#

@testset "Oscillator Tests" begin

    @testset "3-stage CMOS ring oscillator" begin
        # Create circuit
        circuit = MNACircuit(ring_oscillator)

        # Expected frequency estimation:
        # For CMOS inverter, delay ~ C * Vdd / (Kp * (Vgs - Vt)^2)
        # With C=10fF, roughly delay ~ 0.1-1ns per stage
        # 3 stages oscillating at half period = 3 * delay, full period = 6 * delay
        # Expected period ~ 1-10ns, frequency ~ 100MHz - 1GHz
        expected_period_min = 0.5e-9  # 500ps
        expected_period_max = 50e-9   # 50ns

        # Simulate for 200ns to observe oscillation
        tspan = (0.0, 200e-9)

        # Use Rodas5P solver with CedarUICOp initialization
        # CedarUICOp uses pseudo-transient relaxation for oscillators without stable DC equilibrium
        @info "Running ring oscillator transient simulation" tspan
        sol = tran!(circuit, tspan;
                    solver=Rodas5P(),
                    initializealg=CedarUICOp(warmup_steps=20, dt=1e-15),
                    abstol=1e-9, reltol=1e-6,
                    dtmax=1e-9)

        @test sol.retcode == SciMLBase.ReturnCode.Success

        sys = assemble!(circuit)
        acc = MNASolutionAccessor(sol, sys)

        # Sample output voltages in the last half of simulation (after startup transient)
        t_start = 100e-9
        t_end = 200e-9
        n_samples = 1000
        times = range(t_start, t_end; length=n_samples)

        V_out1 = [voltage(acc, :out1, t) for t in times]
        V_out2 = [voltage(acc, :out2, t) for t in times]
        V_in1 = [voltage(acc, :in1, t) for t in times]

        # Verify oscillation occurs - outputs should swing significantly
        out1_min, out1_max = extrema(V_out1)
        out2_min, out2_max = extrema(V_out2)
        in1_min, in1_max = extrema(V_in1)

        @info "Output voltage ranges" out1_min out1_max out2_min out2_max in1_min in1_max

        # Check voltage swings are significant (at least 2V swing for 3.3V supply)
        @test (out1_max - out1_min) > 2.0
        @test (out2_max - out2_min) > 2.0
        @test (in1_max - in1_min) > 2.0

        # Check outputs reach near rail voltages
        @test out1_max > 2.5  # Near Vdd
        @test out1_min < 0.8  # Near ground
        @test out2_max > 2.5
        @test out2_min < 0.8

        # Estimate frequency by counting zero crossings on out1
        midpoint = (out1_max + out1_min) / 2
        crossings = 0
        for i in 2:n_samples
            if (V_out1[i-1] < midpoint && V_out1[i] >= midpoint) ||
               (V_out1[i-1] > midpoint && V_out1[i] <= midpoint)
                crossings += 1
            end
        end

        # Each period has 2 crossings, so frequency = crossings / (2 * duration)
        duration = t_end - t_start
        measured_freq = crossings / (2 * duration)
        measured_period = 1 / measured_freq

        @info "Frequency measurement" crossings measured_freq measured_period

        # Check oscillation frequency is in reasonable range
        @test measured_period > expected_period_min
        @test measured_period < expected_period_max

        # Verify phase relationship: out1, out2, in1 should be ~120° apart
        # (each inverter adds 180° but with 3 stages total = 540° = 180° per stage)
        # Actually for ring oscillator, each node is 360°/3 = 120° apart
        # This is harder to test precisely, so just verify they're not in phase
        correlation_12 = sum(V_out1 .* V_out2) / n_samples
        correlation_23 = sum(V_out2 .* V_in1) / n_samples

        # Normalized to check anti-correlation tendency
        # For 120° phase shift, correlation should be negative (closer to -0.5)
        mean_v = (out1_max + out1_min) / 2
        V_out1_centered = V_out1 .- mean_v
        V_out2_centered = V_out2 .- mean_v
        normalized_corr = sum(V_out1_centered .* V_out2_centered) /
                          sqrt(sum(V_out1_centered.^2) * sum(V_out2_centered.^2))

        @info "Phase correlation" normalized_corr
        # 120° phase shift gives correlation of cos(120°) = -0.5
        @test normalized_corr < 0.5  # Not in phase
    end

end
