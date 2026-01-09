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
using CedarSim.MNA: make_ode_problem, ZERO_VECTOR

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

    # Use solve_dc(builder, params, spec) which properly handles Newton iteration
    # for nonlinear devices (VA BJTs, diodes, etc.)
    spec = MNASpec(temp=Float64(temp), mode=:dcop)
    sol = Base.invokelatest(solve_dc, circuit_fn, (;), spec)

    # Build context for returning (at the solved operating point)
    ctx = Base.invokelatest(circuit_fn, (;), spec, 0.0; x=sol.x)

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

    # Use solve_dc(builder, params, spec) which properly handles Newton iteration
    # for nonlinear devices (VA BJTs, diodes, etc.)
    spec = MNASpec(temp=Float64(temp), mode=:dcop)
    sol = Base.invokelatest(solve_dc, circuit_fn, (;), spec)

    # Build context for returning (at the solved operating point)
    ctx = Base.invokelatest(circuit_fn, (;), spec, 0.0; x=sol.x)

    return ctx, sol
end

"""
    solve_mna_circuit(builder; params=(;), temp=27.0) -> (ctx, sol)

Build and solve a circuit using MNA backend.
`builder` should be a function (params, spec, t; x=...) -> MNAContext.

# Example
```julia
function my_circuit(params, spec, t::Real=0.0; x=Float64[])
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
    # Use solve_dc(builder, params, spec) which properly handles Newton iteration
    # for nonlinear devices (VA BJTs, diodes, etc.)
    spec = MNASpec(temp=Float64(temp), mode=:dcop)
    sol = solve_dc(builder, params, spec)
    ctx = builder(params, spec, 0.0; x=sol.x)
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

using CedarSim.MNA: MNACircuit

"""
    make_mna_spice_circuit(spice_code::String; temp=27.0, imported_hdl_modules=Module[]) -> MNACircuit

Parse SPICE code and return an MNACircuit ready for simulation.

This is a convenience function for test code. It uses `invokelatest` internally
to handle world age issues with runtime-parsed code.

For production code loading SPICE from files, use the top-level eval pattern:

```julia
# At module/file load time (top level)
using CedarSim: parse_spice_to_mna
const circuit_code = parse_spice_to_mna(read("circuit.sp", String);
                                        imported_hdl_modules=[MyVA_module])
eval(circuit_code)  # Defines `circuit` function, advances world

# Now can use normally without invokelatest
circuit = MNACircuit(circuit)
sol = tran!(circuit, (0.0, 1e-3))
```

# Example (test code)
```julia
spice = \"\"\"
* BJT amplifier
V1 vcc 0 DC 12
Vb base 0 DC 0.65
Rc vcc coll 4.7k
X1 base 0 coll npnbjt
\"\"\"

circuit = make_mna_spice_circuit(spice; imported_hdl_modules=[npnbjt_module])
sol = tran!(circuit, (0.0, 1e-3))
```
"""
function make_mna_spice_circuit(spice_code::String; temp::Real=27.0, imported_hdl_modules::Vector{Module}=Module[])
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

    # Wrap in invokelatest to handle world age issues.
    # This is needed because circuit_fn was defined via eval and is called from
    # ODE solver callbacks which are in an older world.
    # For production performance, use the sp"..." macro instead.
    wrapped_fn = (args...; kwargs...) -> Base.invokelatest(circuit_fn, args...; kwargs...)

    spec = MNASpec(temp=Float64(temp), mode=:tran)
    return MNACircuit(wrapped_fn, (;), spec)
end
