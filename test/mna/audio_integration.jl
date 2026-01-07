#==============================================================================#
# Audio Integration Tests: SPICE Circuits with Verilog-A BJT Models
#
# This file tests SPICE netlists with Verilog-A device models using:
# - va"""...""" macro for creating VA device models (BJT Ebers-Moll)
# - SPICE netlists with X device syntax for VA model instantiation
# - Top-level eval pattern for production-style SPICE loading
# - SPICE .param with CircuitSweep for parameter sweeps
# - Transient simulation with sinusoidal sources (SIN syntax)
# - Signal processing analysis (gain measurement)
#
# Key patterns demonstrated:
# 1. VA device models defined separately, imported via imported_hdl_modules
# 2. SPICE X device syntax to instantiate VA models in netlists
# 3. Top-level parse_spice_to_mna + eval (no invokelatest overhead)
# 4. CircuitSweep + Sweep for parameter sweeps with tran!()
# 5. MNASolutionAccessor for accessing transient results
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNACircuit, MNASolutionAccessor
using CedarSim.MNA: voltage, current, assemble!
using CedarSim: tran!, parse_spice_to_mna, CircuitSweep, Sweep
using OrdinaryDiffEq
using SciMLBase

include(joinpath(@__DIR__, "..", "common.jl"))

#==============================================================================#
# BJT Device Model: Simplified Ebers-Moll
#
# The Ebers-Moll model describes BJT behavior using two coupled diodes:
# - Base-emitter junction: forward biased in active mode
# - Base-collector junction: reverse biased in active mode
#
# Parameters (hardcoded for simplicity):
# - Is = 1e-14 A (saturation current)
# - Vt = 25 mV (thermal voltage at room temp)
# - BF = 100 (forward current gain)
#
# Terminal currents:
# Ibe = Is*(exp(Vbe/Vt) - 1) - Is/(1+BF)*(exp(Vbc/Vt) - 1)
# Icb = BF/(1+BF)*Is*(exp(Vbe/Vt) - 1) - Is*(exp(Vbc/Vt) - 1)
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
# Circuit Definitions (top-level eval - no invokelatest needed)
#
# These circuits are parsed and eval'd at file load time, so subsequent
# code can use them without world age issues.
#==============================================================================#

# Circuit 1: BJT with fixed voltage sources (for DC testing)
const bjt_fixed_voltages_code = parse_spice_to_mna("""
* BJT with fixed voltage sources
Vbe base 0 DC 0.65
Vce coll 0 DC 5.0
X1 base 0 coll npnbjt
"""; circuit_name=:bjt_fixed_voltages, imported_hdl_modules=[npnbjt_module])
eval(bjt_fixed_voltages_code)

# Circuit 2: BJT with collector resistor
const bjt_with_rc_code = parse_spice_to_mna("""
* BJT with collector resistor
Vcc vcc 0 DC 12.0
Vb base 0 DC 0.65
Rc vcc coll 4.7k
X1 base 0 coll npnbjt
"""; circuit_name=:bjt_with_rc, imported_hdl_modules=[npnbjt_module])
eval(bjt_with_rc_code)

# Circuit 3: Common emitter amplifier with parameterized sine input
# Uses .param for values that can be overridden via MNACircuit kwargs
const ce_amplifier_code = parse_spice_to_mna("""
* Common emitter amplifier with parameterized input
.param vac=0.001
.param freq=1000.0
Vcc vcc 0 DC 12.0
Vin base 0 DC 0.65 SIN 0.65 vac freq
Rc vcc coll 4.7k
Cload coll 0 100p
X1 base 0 coll npnbjt
"""; circuit_name=:ce_amplifier, imported_hdl_modules=[npnbjt_module])
eval(ce_amplifier_code)

#==============================================================================#
# Tests
#==============================================================================#

@testset "Audio Integration Tests" begin

    #==========================================================================#
    # Test 1: BJT with Fixed Voltage Sources
    #
    # Test the BJT model at a known operating point using voltage sources.
    # This verifies the BJT model produces correct terminal currents.
    #==========================================================================#
    @testset "BJT with fixed voltages - active region" begin
        spec = MNA.MNASpec(mode=:dcop)
        sol = MNA.solve_dc(bjt_fixed_voltages, (;), spec)

        # Check voltages are as expected
        @test isapprox(voltage(sol, :base), 0.65; atol=1e-6)
        @test isapprox(voltage(sol, :coll), 5.0; atol=1e-6)

        # The voltage source currents tell us the terminal currents
        I_Vbe = current(sol, :I_vbe)  # Current into base
        I_Vce = current(sol, :I_vce)  # Current into collector

        # Base current should be small (μA range), collector current larger (mA range)
        @test abs(I_Vbe) > 1e-6   # Base current exists
        @test abs(I_Vce) > abs(I_Vbe)  # Collector current > base current

        # Check that Ic/Ib ≈ BF (current gain)
        BF = 100.0
        @test abs(I_Vce / I_Vbe) > 0.8 * BF
    end

    #==========================================================================#
    # Test 2: BJT with Collector Resistor
    #
    # Use voltage sources for base bias (stable convergence) and
    # a resistor load at the collector. This forms the simplest amplifier.
    #==========================================================================#
    @testset "BJT with collector resistor" begin
        spec = MNA.MNASpec(mode=:dcop)
        sol = MNA.solve_dc(bjt_with_rc, (;), spec)

        V_coll = voltage(sol, :coll)
        V_base = voltage(sol, :base)

        # Base should be at Vb
        @test isapprox(V_base, 0.65; atol=1e-6)

        # Collector voltage = Vcc - Ic * Rc
        # For Vbe=0.65V, Ic ≈ 1.9mA, so Vc ≈ 12 - 1.9e-3 * 4.7e3 ≈ 3.1V
        @test V_coll > 1.0 && V_coll < 10.0  # In reasonable range
        @test V_coll < 12.0  # Below supply
    end

    #==========================================================================#
    # Test 3: Common Emitter Amplifier Transient with Sine Input
    #
    # This is the main test: a common emitter amplifier with:
    # - Fixed DC bias using voltage source (for stable convergence)
    # - Small-signal sinusoidal input superimposed
    # - Load capacitor for transient stability
    # - Measurement of voltage gain
    #==========================================================================#
    @testset "Common emitter amplifier transient" begin
        freq = 1000.0
        Vac = 0.001
        Vbias = 0.65

        # First verify DC operating point
        spec = MNA.MNASpec(mode=:dcop)
        dc_sol = MNA.solve_dc(ce_amplifier, (;), spec)
        V_coll_dc = voltage(dc_sol, :coll)
        V_base_dc = voltage(dc_sol, :base)

        # Verify BJT is biased properly
        @test isapprox(V_base_dc, Vbias; atol=0.01)
        @test V_coll_dc > 1.0 && V_coll_dc < 11.0

        # Create circuit for transient (uses top-level defined builder)
        circuit = MNACircuit(ce_amplifier)

        # Run transient simulation
        period = 1.0 / freq
        tspan = (0.0, 5 * period)

        sol = tran!(circuit, tspan; solver=Rodas5P(), abstol=1e-9, reltol=1e-7)
        @test sol.retcode == ReturnCode.Success

        # Access results
        sys = assemble!(circuit)
        acc = MNASolutionAccessor(sol, sys)

        # Measure output after initial transient (last 2 periods)
        t_start = 3 * period
        times = range(t_start, 5*period; length=100)

        V_coll = [voltage(acc, :coll, t) for t in times]
        V_base = [voltage(acc, :base, t) for t in times]

        # Calculate peak-to-peak voltages
        Vout_pp = maximum(V_coll) - minimum(V_coll)
        Vin_pp = maximum(V_base) - minimum(V_base)

        # Input should be 2mV peak-to-peak (1mV amplitude)
        @test isapprox(Vin_pp, 2 * Vac; rtol=0.15)

        # Calculate voltage gain magnitude
        gain = Vout_pp / Vin_pp

        # Common emitter has inverting gain
        @test gain > 10.0    # Should have significant amplification
        @test gain < 1000.0  # But not unreasonably large

        # Verify phase inversion (input peak → output trough)
        t_max_inp = times[argmax(V_base)]
        V_out_at_inp_max = voltage(acc, :coll, t_max_inp)

        # At input maximum, output should be below DC level (inverted)
        @test V_out_at_inp_max < V_coll_dc
    end

    #==========================================================================#
    # Test 4: Gain Measurement with Different Signal Levels
    #
    # Test linearity by measuring gain with different input amplitudes.
    # Uses CircuitSweep API with SPICE .param parameter sweeps.
    #==========================================================================#
    @testset "Gain linearity with signal level" begin
        freq = 1000.0
        period = 1.0 / freq
        tspan = (0.0, 5 * period)

        # Create sweep over vac values
        sweep = Sweep(vac = [0.0005, 0.001, 0.002])
        cs = CircuitSweep(ce_amplifier, sweep; vac=0.001, freq=freq)

        # Run transient sweep - returns vector of solutions
        solutions = tran!(cs, tspan; solver=Rodas5P(), abstol=1e-9, reltol=1e-7)

        gains = Float64[]
        for (sol, circuit) in zip(solutions, cs)
            if sol.retcode == ReturnCode.Success
                sys = assemble!(circuit)
                acc = MNASolutionAccessor(sol, sys)

                t_start = 3 * period
                times = range(t_start, 5*period; length=100)

                V_coll = [voltage(acc, :coll, t) for t in times]
                V_base = [voltage(acc, :base, t) for t in times]

                Vout_pp = maximum(V_coll) - minimum(V_coll)
                Vin_pp = maximum(V_base) - minimum(V_base)

                push!(gains, Vout_pp / Vin_pp)
            else
                push!(gains, NaN)
            end
        end

        # All simulations should succeed
        @test all(!isnan, gains)

        # For small signals, gain should be relatively constant (within 30%)
        if all(!isnan, gains) && length(gains) >= 2
            avg_gain = sum(gains) / length(gains)
            @test all(abs.(gains .- avg_gain) ./ avg_gain .< 0.3)
        end
    end

    #==========================================================================#
    # Test 5: Frequency Response
    #
    # Test gain at different frequencies to verify proper operation
    # across audio band. Uses CircuitSweep API with frequency-dependent tspans.
    #==========================================================================#
    @testset "Frequency response" begin
        vac = 0.001
        freqs = [100.0, 1000.0, 10000.0]

        # Create sweep over frequency values
        sweep = Sweep(freq = freqs)
        cs = CircuitSweep(ce_amplifier, sweep; vac=vac, freq=1000.0)

        gains = Float64[]
        # Iterate over sweep - each frequency needs different tspan
        for circuit in cs
            freq = circuit.params.freq
            period = 1.0 / freq
            tspan = (0.0, 10 * period)

            sol = tran!(circuit, tspan; solver=Rodas5P(), abstol=1e-9, reltol=1e-7)

            if sol.retcode == ReturnCode.Success
                sys = assemble!(circuit)
                acc = MNASolutionAccessor(sol, sys)

                # Measure last 3 periods
                t_start = 7 * period
                t_end = 10 * period
                times = range(t_start, t_end; length=100)

                V_coll = [voltage(acc, :coll, t) for t in times]
                V_base = [voltage(acc, :base, t) for t in times]

                Vout_pp = maximum(V_coll) - minimum(V_coll)
                Vin_pp = maximum(V_base) - minimum(V_base)

                push!(gains, Vout_pp / Vin_pp)
            else
                push!(gains, NaN)
            end
        end

        # At least one should succeed
        @test any(!isnan, gains)

        # For a resistor-only load, gain should be constant with frequency
        valid_gains = filter(!isnan, gains)
        if length(valid_gains) >= 2
            @test maximum(valid_gains) / minimum(valid_gains) < 2.0
        end
    end

end  # testset "Audio Integration Tests"
