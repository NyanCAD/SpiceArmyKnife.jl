using CedarSim
using CedarSim.SpectreEnvironment
using SpectreNetlistParser
using Test
using Random

# Simulation packages
using OrdinaryDiffEq
using SciMLBase
using Sundials

const deftol = 1e-8

# Our default tolerances are one order of magnitude above our default solve tolerances
isapprox_deftol(x, y) = isapprox(x, y; atol=deftol*10, rtol=deftol*10)
isapprox_deftol(x) = y->isapprox(x, y; atol=deftol*10, rtol=deftol*10)

allapprox_deftol(itr) = isempty(itr) ? true : all(isapprox_deftol(first(itr)), itr)

#==============================================================================#
# MNA-based solve functions (Phase 4+, no DAECompiler required)
#==============================================================================#

using CedarSim.MNA: MNAContext, MNASpec, assemble!, solve_dc, solve_ac
using CedarSim.MNA: voltage, current, get_node!, stamp!
using CedarSim.MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using CedarSim.MNA: make_ode_problem

"""
    solve_mna_spice_code(spice_code::String; temp=27.0) -> (ctx, sol)

Parse SPICE code and solve DC operating point using MNA backend.
Returns (MNAContext, DCSolution).

# Example
```julia
code = \"\"\"
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
\"\"\"
ctx, sol = solve_mna_spice_code(code)
voltage(sol, :out)  # Returns 2.5
```
"""
function solve_mna_spice_code(spice_code::String; temp::Real=27.0, imported_hdl_modules::Vector{Module}=Module[])
    ast = CedarSim.SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = CedarSim.make_mna_circuit(ast; imported_hdl_modules)

    # Evaluate in temporary module
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    # Import VA device types
    for hdl_mod in imported_hdl_modules
        for name in names(hdl_mod; all=true, imported=false)
            if !startswith(String(name), "#") && isdefined(hdl_mod, name)
                val = getfield(hdl_mod, name)
                isa(val, Type) && Base.eval(m, :(const $name = $val))
            end
        end
    end
    circuit_fn = Base.eval(m, code)

    spec = MNASpec(temp=Float64(temp), mode=:dcop)
    ctx = Base.invokelatest(circuit_fn, (;), spec)
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    return ctx, sol
end

"""
    solve_mna_spectre_code(spectre_code::String; temp=27.0) -> (ctx, sol)

Parse Spectre code and solve DC operating point using MNA backend.
Returns (MNAContext, DCSolution).

# Example
```julia
code = \"\"\"
v1 (vcc 0) vsource dc=5
r1 (vcc out) resistor r=1k
r2 (out 0) resistor r=1k
\"\"\"
ctx, sol = solve_mna_spectre_code(code)
voltage(sol, :out)  # Returns 2.5
```
"""
function solve_mna_spectre_code(spectre_code::String; temp::Real=27.0, imported_hdl_modules::Vector{Module}=Module[])
    ast = CedarSim.SpectreNetlistParser.parse(IOBuffer(spectre_code); start_lang=:spectre)
    code = CedarSim.make_mna_circuit(ast; imported_hdl_modules)

    # Evaluate in temporary module
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
    # Import VA device types
    for hdl_mod in imported_hdl_modules
        for name in names(hdl_mod; all=true, imported=false)
            if !startswith(String(name), "#") && isdefined(hdl_mod, name)
                val = getfield(hdl_mod, name)
                isa(val, Type) && Base.eval(m, :(const $name = $val))
            end
        end
    end
    circuit_fn = Base.eval(m, code)

    spec = MNASpec(temp=Float64(temp), mode=:dcop)
    ctx = Base.invokelatest(circuit_fn, (;), spec)
    sys = assemble!(ctx)
    sol = solve_dc(sys)

    return ctx, sol
end

"""
    solve_mna_circuit(builder; params=(;), temp=27.0) -> (ctx, sol)

Build and solve a circuit using MNA backend.
`builder` should be a function (params, spec) -> MNAContext.

# Example
```julia
function my_circuit(params, spec, t::Real=0.0)
    ctx = MNAContext()
    vcc = get_node!(ctx, :vcc)
    stamp!(VoltageSource(5.0), ctx, vcc, 0)
    stamp!(Resistor(params.R), ctx, vcc, 0)
    return ctx
end
ctx, sol = solve_mna_circuit(my_circuit; params=(R=1000.0,))
```
"""
function solve_mna_circuit(builder; params=(;), temp::Real=27.0)
    spec = MNASpec(temp=Float64(temp), mode=:dcop)
    ctx = builder(params, spec)
    sys = assemble!(ctx)
    sol = solve_dc(sys)
    return ctx, sol
end

"""
    tran_mna_circuit(builder, tspan; params=(;), temp=27.0, solver=Rodas5P()) -> (ctx, sol)

Build and solve a transient simulation using MNA backend.

# Example
```julia
function rc_circuit(params, spec, t::Real=0.0)
    ctx = MNAContext()
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)
    stamp!(VoltageSource(params.V), ctx, vcc, 0)
    stamp!(Resistor(params.R), ctx, vcc, out)
    stamp!(Capacitor(params.C), ctx, out, 0)
    return ctx
end
ctx, sol = tran_mna_circuit(rc_circuit, (0.0, 1e-3); params=(V=5.0, R=1000.0, C=1e-6))
```
"""
function tran_mna_circuit(builder, tspan::Tuple; params=(;), temp::Real=27.0, solver=Rodas5P())
    spec = MNASpec(temp=Float64(temp), mode=:tran)
    ctx = builder(params, spec)
    sys = assemble!(ctx)

    # Create and solve ODE problem
    prob_data = make_ode_problem(sys, tspan)
    f = ODEFunction(prob_data.f; mass_matrix=prob_data.mass_matrix,
                    jac=prob_data.jac, jac_prototype=prob_data.jac_prototype)
    prob = ODEProblem(f, prob_data.u0, prob_data.tspan)
    sol = solve(prob, solver; reltol=deftol, abstol=deftol)

    return ctx, sol
end
