module varegress

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, get_node!, stamp!, assemble!
using CedarSim.MNA: voltage, current, make_ode_problem
using CedarSim.MNA: VoltageSource, Capacitor, MNACircuit
using CedarSim: dc!
using OrdinaryDiffEq
using Test

# Simple Verilog-A Resistor for smoke testing
# Note: disciplines are implicit - do NOT use `include "disciplines.vams"`
va"""
module VAR(p,n);
    parameter real R = 1000.0 from (0.0:inf] ;
    electrical p, n;
    analog I(p,n) <+ V(p,n)/R;
endmodule
"""

@testset "VA Resistor (normal port order)" begin
    # Circuit: V(1V) -- VAR(1kΩ) -- C(1nF) -- GND
    # This is an RC charging circuit
    function circuit1(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            CedarSim.MNA.reset_for_restamping!(ctx)
        end
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(1.0; name=:V), ctx, vcc, 0)
        stamp!(VAR(R=1000.0), ctx, vcc, out)
        stamp!(Capacitor(1e-9; name=:C), ctx, out, 0)

        return ctx
    end

    # DC analysis - at steady state, no current through capacitor
    # V_out should equal V_vcc = 1V (no DC current through capacitor)
    circuit = MNACircuit(circuit1)
    sol = dc!(circuit)
    @test isapprox(voltage(sol, :vcc), 1.0; atol=1e-10)
    @test isapprox(voltage(sol, :out), 1.0; atol=1e-10)

    # Transient analysis - RC charging
    ctx = circuit1((;), MNASpec(mode=:tran))
    sys = assemble!(ctx)

    tau = 1000.0 * 1e-9  # R*C = 1μs
    tspan = (0.0, 5*tau)  # 5 time constants

    prob_data = make_ode_problem(sys, tspan)

    # Start with capacitor uncharged
    u0 = copy(prob_data.u0)
    out_idx = findfirst(n -> n == :out, sys.node_names)
    u0[out_idx] = 0.0

    f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                    jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
    prob = ODEProblem(f, u0, prob_data.tspan)
    sol_tran = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-8, abstol=1e-10)

    # At t=0, V_out = 0
    @test isapprox(sol_tran.u[1][out_idx], 0.0; atol=1e-6)

    # At t=5τ, V_out ≈ V_vcc * (1 - exp(-5)) ≈ 0.993 * V_vcc
    expected_final = 1.0 * (1 - exp(-5))
    @test isapprox(sol_tran.u[end][out_idx], expected_final; rtol=0.01)

    # Current through resistor should always be positive (flowing from vcc to out)
    # During charging: I = (V_vcc - V_out) / R > 0
    vcc_idx = findfirst(n -> n == :vcc, sys.node_names)
    for u in sol_tran.u
        V_vcc = u[vcc_idx]
        V_out = u[out_idx]
        I_resistor = (V_vcc - V_out) / 1000.0
        @test I_resistor >= -1e-10  # Should be non-negative (allow small numerical error)
    end
end

# Simple Verilog-A Resistor with reversed port order
va"""
module VAR_rev(p,n);
    parameter real R = 1000.0 from (0.0:inf] ;
    electrical p, n;
    analog I(n,p) <+ V(n,p)/R;
endmodule
"""

@testset "VA Resistor (reversed port order)" begin
    # Same circuit but with VAR_rev which uses I(n,p) <+ V(n,p)/R
    # This should behave identically - current still flows from vcc to out
    function circuit2(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            CedarSim.MNA.reset_for_restamping!(ctx)
        end
        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(1.0; name=:V), ctx, vcc, 0)
        stamp!(VAR_rev(R=1000.0), ctx, vcc, out)
        stamp!(Capacitor(1e-9; name=:C), ctx, out, 0)

        return ctx
    end

    # DC analysis
    circuit = MNACircuit(circuit2)
    sol = dc!(circuit)
    @test isapprox(voltage(sol, :vcc), 1.0; atol=1e-10)
    @test isapprox(voltage(sol, :out), 1.0; atol=1e-10)

    # Transient analysis
    ctx = circuit2((;), MNASpec(mode=:tran))
    sys = assemble!(ctx)

    tau = 1000.0 * 1e-9
    tspan = (0.0, 5*tau)

    prob_data = make_ode_problem(sys, tspan)
    u0 = copy(prob_data.u0)
    out_idx = findfirst(n -> n == :out, sys.node_names)
    u0[out_idx] = 0.0

    f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                    jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
    prob = ODEProblem(f, u0, prob_data.tspan)
    sol_tran = OrdinaryDiffEq.solve(prob, Rodas5P(); reltol=1e-8, abstol=1e-10)

    # Current through resistor should still be non-negative
    # (The reversed I(n,p) <+ V(n,p)/R is mathematically equivalent)
    vcc_idx = findfirst(n -> n == :vcc, sys.node_names)
    for u in sol_tran.u
        V_vcc = u[vcc_idx]
        V_out = u[out_idx]
        I_resistor = (V_vcc - V_out) / 1000.0
        @test I_resistor >= -1e-10
    end
end

end # module varegress
