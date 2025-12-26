#==============================================================================#
# MNA Phase 6: VADistiller Model Integration Tests
#
# Tests for compatibility with models from https://codeberg.org/arpadbuermen/VADistiller
# These are SPICE3 models converted to Verilog-A.
#
# Approach:
# 1. Start with simplified versions that strip unsupported constructs
# 2. Gradually add support for missing VA features
# 3. Progress from simple (resistor) to complex (MOSFET) models
#==============================================================================#

using Test
using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!, solve_dc
using CedarSim.MNA: voltage, current, make_ode_problem
using CedarSim.MNA: va_ddt, stamp_current_contribution!, evaluate_contribution
using CedarSim.MNA: VoltageSource, Resistor, Capacitor, CurrentSource
using ForwardDiff: Dual, value, partials
using OrdinaryDiffEq
using VerilogAParser

const deftol = 1e-6
isapprox_deftol(a, b) = isapprox(a, b; atol=deftol, rtol=deftol)

@testset "VADistiller Models" begin

    #==========================================================================#
    # Tier 1: Simplified Linear Passives (2-terminal, no special constructs)
    #==========================================================================#

    @testset "Tier 1: Simplified Passives" begin

        @testset "Simple resistor" begin
            # Minimal resistor: just resistance parameter and I = V/R
            # Use unique module name to avoid conflicts
            va"""
            module VADResistor(pos, neg);
                parameter real resistance = 1000.0;
                inout pos, neg;
                electrical pos, neg;
                analog I(pos,neg) <+ V(pos,neg)/resistance;
            endmodule
            """

            # Test in voltage divider
            function resistor_divider(params, spec)
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                mid = get_node!(ctx, :mid)

                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VADResistor(resistance=1000.0), ctx, vcc, mid)
                stamp!(VADResistor(resistance=1000.0), ctx, mid, 0)

                return ctx
            end

            ctx = resistor_divider((;), MNASpec())
            sys = assemble!(ctx)
            sol = solve_dc(sys)

            @test isapprox_deftol(voltage(sol, :vcc), 5.0)
            @test isapprox_deftol(voltage(sol, :mid), 2.5)  # Voltage divider
            @test isapprox(current(sol, :I_V1), -0.0025; atol=1e-5)  # 5V / 2kΩ
        end

        @testset "Simple capacitor" begin
            # Minimal capacitor: just capacitance parameter and I = C*dV/dt
            va"""
            module VADCapacitor(pos, neg);
                parameter real capacitance = 1e-6;
                inout pos, neg;
                electrical pos, neg;
                analog I(pos,neg) <+ capacitance*ddt(V(pos,neg));
            endmodule
            """

            # Test in RC circuit transient
            R_val = 1000.0
            C_val = 1e-6
            V_val = 5.0

            function rc_circuit(params, spec)
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                cap = get_node!(ctx, :cap)

                stamp!(VoltageSource(V_val; name=:V1), ctx, vcc, 0)
                stamp!(Resistor(R_val; name=:R1), ctx, vcc, cap)
                stamp!(VADCapacitor(capacitance=C_val), ctx, cap, 0)

                return ctx
            end

            ctx = rc_circuit((;), MNASpec(mode=:tran))
            sys = assemble!(ctx)

            # Set up transient
            tau = R_val * C_val
            tspan = (0.0, 5 * tau)
            prob_data = make_ode_problem(sys, tspan)

            u0 = copy(prob_data.u0)
            cap_idx = findfirst(n -> n == :cap, sys.node_names)
            u0[cap_idx] = 0.0

            f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                            jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
            prob = ODEProblem(f, u0, prob_data.tspan)
            sol = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-6, abstol=1e-8)

            # Check RC charging: V_cap(t) = V * (1 - e^(-t/τ))
            @test isapprox(sol.u[1][cap_idx], 0.0; atol=1e-6)
            expected_final = V_val * (1 - exp(-5))
            @test isapprox(sol.u[end][cap_idx], expected_final; rtol=0.01)
        end

    end

    #==========================================================================#
    # Tier 2: Diode (nonlinear, 2-terminal)
    #==========================================================================#

    @testset "Tier 2: Simple Diode" begin

        @testset "Ideal diode (exponential)" begin
            # Minimal Shockley diode: I = Is*(exp(V/Vt) - 1)
            va"""
            module VADDiode(a, c);
                parameter real Is = 1e-14;
                parameter real N = 1.0;
                inout a, c;
                electrical a, c;
                analog begin
                    I(a,c) <+ Is*(exp(V(a,c)/(N*0.02585)) - 1.0);
                end
            endmodule
            """

            # Test forward bias: V=0.6V should give I ≈ 1e-4 A
            # Use Newton iteration DC solver for nonlinear devices
            function diode_circuit(params, spec; x=Float64[])
                ctx = MNAContext()
                anode = get_node!(ctx, :anode)

                stamp!(VoltageSource(0.6; name=:V1), ctx, anode, 0)
                stamp!(VADDiode(Is=1e-14, N=1.0), ctx, anode, 0; x=x)

                return ctx
            end

            # Use builder-based solve_dc which does Newton iteration
            sol = solve_dc(diode_circuit, (;), MNASpec())

            # Expected: I = 1e-14 * (exp(0.6/0.02585) - 1) ≈ 1.05e-4 A
            Vt = 0.02585
            expected_I = 1e-14 * (exp(0.6/Vt) - 1)
            actual_I = -current(sol, :I_V1)  # Negative because V1 sources current
            @test isapprox(actual_I, expected_I; rtol=0.01)
        end

        @testset "Diode with series resistance" begin
            # Diode with Rs for more realistic behavior
            # This tests internal nodes which require phase 6 features
            # NOTE: VA definition commented out - internal nodes not yet supported
            # Once internal node support is added, uncomment this:
            #=
            va"""
            module VADDiodeRs(a, c);
                parameter real Is = 1e-14;
                parameter real N = 1.0;
                parameter real Rs = 10.0;
                inout a, c;
                electrical a, c, a_int;
                analog begin
                    I(a, a_int) <+ V(a, a_int) / Rs;
                    I(a_int, c) <+ Is*(exp(V(a_int,c)/(N*0.02585)) - 1.0);
                end
            endmodule
            """
            =#

            # Forward bias test with internal node
            # This will fail until we add internal node support
            @test_broken false  # Placeholder: internal nodes not yet supported
        end

    end

    #==========================================================================#
    # Tier 3: 3-terminal devices (BJT, JFET)
    #==========================================================================#

    @testset "Tier 3: Simple 3-Terminal Devices" begin

        @testset "Simple VCCS (JFET approximation)" begin
            # Linear VCCS: Ids = gm * Vgs
            va"""
            module VADVCCS(d, g, s);
                parameter real gm = 1e-3;
                inout d, g, s;
                electrical d, g, s;
                analog begin
                    I(d,s) <+ gm * V(g,s);
                end
            endmodule
            """

            # Test: Vgs controls Ids
            function vccs_circuit(params, spec)
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                # Power supply
                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                # Gate bias
                stamp!(VoltageSource(1.0; name=:Vg), ctx, gate, 0)
                # Load resistor
                stamp!(Resistor(1000.0; name=:Rd), ctx, vdd, drain)
                # VCCS: Ids = gm * Vgs
                stamp!(VADVCCS(gm=1e-3), ctx, drain, gate, 0)

                return ctx
            end

            ctx = vccs_circuit((;), MNASpec())
            sys = assemble!(ctx)
            sol = solve_dc(sys)

            # Ids = gm * Vgs = 1e-3 * 1.0 = 1mA
            # Vdrain = Vdd - Ids * Rd = 5 - 1e-3 * 1000 = 4V
            @test isapprox(voltage(sol, :drain), 4.0; atol=0.01)
        end

        @testset "Simple square-law MOSFET" begin
            # Level 1 MOSFET in saturation: Ids = (K/2)(Vgs - Vth)²
            # Simplified: always-on for testing (no threshold cutoff)
            va"""
            module VADMOS(d, g, s);
                parameter real Kp = 1e-4;
                parameter real Vth = 0.5;
                inout d, g, s;
                electrical d, g, s;
                analog I(d,s) <+ (Kp/2.0) * (V(g,s) - Vth) * (V(g,s) - Vth);
            endmodule
            """

            # Common source amplifier
            # Use Newton iteration DC solver for nonlinear devices
            function mos_circuit(params, spec; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(1.5; name=:Vg), ctx, gate, 0)  # Vgs = 1.5V
                stamp!(Resistor(10000.0; name=:Rd), ctx, vdd, drain)  # 10kΩ load
                stamp!(VADMOS(Kp=1e-4, Vth=0.5), ctx, drain, gate, 0; x=x)

                return ctx
            end

            # Use builder-based solve_dc which does Newton iteration
            sol = solve_dc(mos_circuit, (;), MNASpec())

            # Ids = (1e-4/2) * (1.5 - 0.5)² = 0.5e-4 * 1 = 50µA
            # Vdrain = 5 - 50e-6 * 10000 = 5 - 0.5 = 4.5V
            @test isapprox(voltage(sol, :drain), 4.5; atol=0.1)
        end

    end

    #==========================================================================#
    # Tier 4: N-type MOSFET with bulk terminal
    #==========================================================================#

    @testset "Tier 4: 4-Terminal Devices" begin

        @testset "Simple NMOS with bulk" begin
            # 4-terminal NMOS: d, g, s, b
            # Simplified square-law in saturation (no body effect for now)
            va"""
            module VADNMOS4(d, g, s, b);
                parameter real Kp = 1e-4;
                parameter real Vth = 0.5;
                inout d, g, s, b;
                electrical d, g, s, b;
                analog I(d,s) <+ (Kp/2.0) * (V(g,s) - Vth) * (V(g,s) - Vth);
            endmodule
            """

            # Common source amplifier with bulk tied to source
            function nmos4_circuit(params, spec; x=Float64[])
                ctx = MNAContext()
                vdd = get_node!(ctx, :vdd)
                gate = get_node!(ctx, :gate)
                drain = get_node!(ctx, :drain)

                stamp!(VoltageSource(5.0; name=:Vdd), ctx, vdd, 0)
                stamp!(VoltageSource(1.5; name=:Vg), ctx, gate, 0)
                stamp!(Resistor(10000.0; name=:Rd), ctx, vdd, drain)
                # 4-terminal stamp: d=drain, g=gate, s=0, b=0 (bulk tied to source)
                stamp!(VADNMOS4(Kp=1e-4, Vth=0.5), ctx, drain, gate, 0, 0; x=x)

                return ctx
            end

            # Use builder-based solve_dc
            sol = solve_dc(nmos4_circuit, (;), MNASpec())

            # Same result as 3-terminal MOSFET
            @test isapprox(voltage(sol, :drain), 4.5; atol=0.1)
        end

    end

    #==========================================================================#
    # Feature Tests: VA System Functions
    #==========================================================================#

    @testset "Feature: VA System Functions" begin

        @testset "\$temperature access" begin
            # Test temperature-dependent resistor using $temperature
            va_code = raw"""
            module VATempResistor(p, n);
                parameter real R0 = 1000.0;
                parameter real TC = 0.004;
                inout p, n;
                electrical p, n;
                analog begin
                    I(p,n) <+ V(p,n) / (R0 * (1.0 + TC * ($temperature() - 300.0)));
                end
            endmodule
            """
            va = VerilogAParser.parse(IOBuffer(va_code))
            Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

            # Test at default temp (27C = 300.15K)
            function temp_resistor_circuit(params, spec; x=Float64[])
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)

                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VATempResistor(R0=1000.0, TC=0.004), ctx, vcc, 0; x=x, spec=spec)

                return ctx
            end

            # At 27°C (300.15K), resistance should be ~R0
            spec = MNASpec(temp=27.0)
            sol = solve_dc(temp_resistor_circuit, (;), spec)
            # R = 1000 * (1 + 0.004 * (300.15 - 300)) = 1000 * 1.0006 ≈ 1000.6
            # I = 5V / 1000.6 ≈ 0.005
            @test isapprox(current(sol, :I_V1), -0.005; atol=1e-4)

            # At 100°C (373.15K), resistance should be higher
            spec_hot = MNASpec(temp=100.0)
            sol_hot = solve_dc(temp_resistor_circuit, (;), spec_hot)
            # R = 1000 * (1 + 0.004 * (373.15 - 300)) = 1000 * 1.293 ≈ 1292.6
            # I = 5V / 1292.6 ≈ 0.00387
            @test isapprox(current(sol_hot, :I_V1), -0.00387; atol=1e-4)
        end

        @testset "\$simparam access" begin
            # Test $simparam with default value
            va_code = raw"""
            module VAGminResistor(p, n);
                parameter real R = 1000.0;
                inout p, n;
                electrical p, n;
                analog begin
                    I(p,n) <+ V(p,n) / R + V(p,n) * $simparam("gmin", 1e-12);
                end
            endmodule
            """
            va = VerilogAParser.parse(IOBuffer(va_code))
            Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

            function gmin_circuit(params, spec; x=Float64[])
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)

                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VAGminResistor(R=1000.0), ctx, vcc, 0; x=x, spec=spec)

                return ctx
            end

            # With default gmin=1e-12
            spec = MNASpec()
            sol = solve_dc(gmin_circuit, (;), spec)
            # I ≈ 5/1000 + 5*1e-12 ≈ 0.005
            @test isapprox(current(sol, :I_V1), -0.005; atol=1e-6)

            # With custom gmin=1e-3 (very high for testing)
            spec_gmin = MNASpec(gmin=1e-3)
            sol_gmin = solve_dc(gmin_circuit, (;), spec_gmin)
            # I = 5/1000 + 5*1e-3 = 0.005 + 0.005 = 0.01
            @test isapprox(current(sol_gmin, :I_V1), -0.01; atol=1e-6)
        end

        @testset "\$param_given check" begin
            # Test $param_given to check if parameter was explicitly set
            # Uses ternary expression since if/else with contributions needs more work
            va_code = raw"""
            module VAOptionalParam(p, n);
                parameter real R = 1000.0;
                parameter real Ralt = 500.0;
                inout p, n;
                electrical p, n;
                analog begin
                    I(p,n) <+ V(p,n) / ($param_given(R) ? R : Ralt);
                end
            endmodule
            """
            va = VerilogAParser.parse(IOBuffer(va_code))
            Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

            # With explicit R=2000 (R is "given")
            sol_explicit = solve_dc((p,s; x=Float64[]) -> begin
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VAOptionalParam(R=2000.0, Ralt=500.0), ctx, vcc, 0; x=x, spec=s)
                return ctx
            end, (;), MNASpec())
            # $param_given(R) is true, so uses R=2000
            # I = 5V / 2000Ω = 0.0025A
            @test isapprox(current(sol_explicit, :I_V1), -0.0025; atol=1e-6)

            # With only Ralt given (R is NOT "given")
            sol_default = solve_dc((p,s; x=Float64[]) -> begin
                ctx = MNAContext()
                vcc = get_node!(ctx, :vcc)
                stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
                stamp!(VAOptionalParam(Ralt=500.0), ctx, vcc, 0; x=x, spec=s)
                return ctx
            end, (;), MNASpec())
            # $param_given(R) is false, so uses Ralt=500
            # I = 5V / 500Ω = 0.01A
            @test isapprox(current(sol_default, :I_V1), -0.01; atol=1e-6)
        end

    end

    #==========================================================================#
    # Feature Tests: Parser Extensions Still Needed
    #==========================================================================#

    @testset "Feature: aliasparam" begin
        # Test aliasparam declaration - allows parameter aliases
        va_code = raw"""
        module VAAliasTest(p, n);
            parameter real tnom = 27.0;
            aliasparam tref = tnom;
            inout p, n;
            electrical p, n;
            analog begin
                I(p,n) <+ V(p,n) / (1000.0 * (1.0 + 0.001 * (tnom - 27.0)));
            end
        endmodule
        """
        va = VerilogAParser.parse(IOBuffer(va_code))
        Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

        # Test 1: Using the real parameter name (tnom)
        sol_tnom = solve_dc((p,s; x=Float64[]) -> begin
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
            stamp!(VAAliasTest(tnom=100.0), ctx, vcc, 0; x=x, spec=s)
            return ctx
        end, (;), MNASpec())
        # R = 1000 * (1 + 0.001 * (100 - 27)) = 1000 * 1.073 = 1073
        # I = 5V / 1073Ω ≈ 0.00466A
        @test isapprox(current(sol_tnom, :I_V1), -0.00466; atol=1e-4)

        # Test 2: Using the alias (tref) - should have same effect as tnom
        sol_tref = solve_dc((p,s; x=Float64[]) -> begin
            ctx = MNAContext()
            vcc = get_node!(ctx, :vcc)
            stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
            stamp!(VAAliasTest(tref=100.0), ctx, vcc, 0; x=x, spec=s)
            return ctx
        end, (;), MNASpec())
        # Same result - alias forwards to tnom
        @test isapprox(current(sol_tref, :I_V1), -0.00466; atol=1e-4)

        # Test 3: Verify property access - dev.tref should return dev.tnom
        dev = VAAliasTest(tnom=50.0)
        @test dev.tnom == dev.tref  # Alias returns target value
    end

    @testset "Feature: Parser Extensions Needed" begin

        @testset "Real variable with initialization" begin
            # VADistiller uses `real G = 0.0;` at module scope
            # Currently broken: parser error at inline initialization
            # Note: This requires VerilogAParser changes
            @test_broken false
        end

        @testset "Internal nodes" begin
            # VADistiller models like BJT have internal nodes (b_int, c_int, etc.)
            # Current stamp! doesn't allocate internal nodes automatically
            @test_broken false
        end

    end

end

#==============================================================================#
# Summary of VADistiller Model Compatibility
#==============================================================================#
#
# WORKING (simplified versions without advanced features):
# ✅ Resistor - I(p,n) <+ V(p,n)/R
# ✅ Capacitor - I(p,n) <+ C*ddt(V(p,n))
# ✅ Diode - I(a,c) <+ Is*(exp(V/Vt) - 1) with Newton iteration
# ✅ 3-terminal VCCS - I(d,s) <+ gm*V(g,s)
# ✅ 3-terminal MOSFET - I(d,s) <+ Kp/2*(Vgs-Vth)^2 with Newton iteration
# ✅ 4-terminal NMOS - same as above with bulk terminal
#
# WORKING VA SYSTEM FUNCTIONS:
# ✅ $temperature() - simulator temperature access (returns Kelvin)
# ✅ $simparam("name", default) - simulator parameter queries (gmin, tnom, etc.)
# ✅ $param_given(param) - check if parameter was specified
# ✅ $vt() - thermal voltage (kT/q)
# ✅ aliasparam - parameter aliases (aliasparam tref = tnom;)
#
# MISSING PARSER FEATURES (needed for full VADistiller models):
# ❌ $mfactor - device multiplicity
# ❌ $limit() - voltage limiting for Newton convergence
# ❌ Real variable initialization at module scope (real x = 0.0;)
# ❌ @(initial_step) - initialization event handling
# ❌ analog function definitions (used for reusable computations)
#
# MISSING MNA FEATURES (needed for complex models):
# ❌ Internal node allocation in stamp!
# ❌ Branch-based stamping (branch declaration)
# ❌ Noise functions (not needed for DC/transient)
#
# TESTED VADistiller MODELS:
# - resistor.va: Parser fails at `real REStemp = 0;`
# - capacitor.va: Parser fails at `real CAPtemp = 0;`
# - inductor.va: Parser fails at inline real init
# - diode.va: Parser fails at inline real init + uses internal nodes
# - bjt.va: Complex - internal nodes, aliasparam, $simparam, $temperature
# - mos1.va: Complex - same issues as BJT
# - bsim4v8.va: Very complex - needs all features above
#==============================================================================#
