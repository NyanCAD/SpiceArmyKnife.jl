using CedarSim
using CedarSim.SpectreEnvironment
using SpectreNetlistParser
using Test
using Random

# Phase 0: Conditional imports - simulation packages only if available
using OrdinaryDiffEq
using SciMLBase
using Sundials

# Phase 0: DAECompiler may not be available
const HAS_DAECOMPILER = CedarSim.USE_DAECOMPILER

if HAS_DAECOMPILER
    using DAECompiler
end

# These must be top-level `const` values, otherwise `DAECompiler` doesn't know
# that we're not about to redefine it halfway through the model, and some of our
# static assumptions break.  We have a test asserting this in `test/compilation.jl`.
const R = CedarSim.SimpleResistor
const C = CedarSim.SimpleCapacitor
const L = CedarSim.SimpleInductor
const V(v) = CedarSim.VoltageSource(dc=v)
const I(i) = CedarSim.CurrentSource(dc=i)

# Phase 0: sim_time comes from stubs when DAECompiler not available
const sim_time = if HAS_DAECOMPILER
    CedarSim.DAECompiler.sim_time
else
    CedarSim.DAECompilerStubs.sim_time
end

const deftol = 1e-8

# Our default tolerances are one order of magnitude above our default solve tolerances
isapprox_deftol(x, y) = isapprox(x, y; atol=deftol*10, rtol=deftol*10)
isapprox_deftol(x) = y->isapprox(x, y; atol=deftol*10, rtol=deftol*10)

allapprox_deftol(itr) = isempty(itr) ? true : all(isapprox_deftol(first(itr)), itr)

# Phase 0: Simulation functions require DAECompiler
if HAS_DAECOMPILER

"""
    solve_circuit(circuit::Function; time_bounds = (0.0, 1.0), reltol, abstol, u0)

Solves a circuit provided in the form of a function. Uses default values for various
pieces of the solver machinery, override them with the appropriate kwargs.
Returns `sys`, `sol`.
"""
function solve_circuit(circuit::CedarSim.AbstractSim; time_bounds::Tuple = (0.0, 1.0),
                       reltol=deftol, abstol=deftol, u0=nothing, debug_config = (;))
    sys = CircuitIRODESystem(circuit; debug_config)
    prob = ODEProblem(sys, u0, time_bounds, circuit)
    sol = solve(prob, FBDF(autodiff=false); reltol, abstol, initializealg=CedarTranOp(;abstol=1e-14))
    return sys, sol
end
solve_circuit(circuit; kwargs...) = solve_circuit(CedarSim.DefaultSim(circuit); kwargs...)

"""
    solve_spice_file(spice_file; kwargs...)

Read in the given SPICE file, generate a CircuitIRODESystem, and pass it off to `solve_circuit()`
"""
function solve_spice_file(spice_file::String; include_dirs::Vector{String} = String[dirname(spice_file)], kwargs...)
    circuit_code = CedarSim.make_spectre_circuit(
        CedarSim.SpectreNetlistParser.SPICENetlistParser.SPICENetlistCSTParser.parsefile(spice_file),
        include_dirs,
    );
    circuit = eval(circuit_code)
    invokelatest(circuit)
    return solve_circuit(circuit; kwargs...)
end

function solve_spice_code(spice_code::String; include_dirs::Vector{String} = String[], kwargs...)
    circuit_code = CedarSim.make_spectre_circuit(
        CedarSim.SpectreNetlistParser.SPICENetlistParser.SPICENetlistCSTParser.parse(spice_code),
        include_dirs,
    );
    fn = eval(circuit_code)
    invokelatest(fn)
    return solve_circuit(fn; kwargs...)
end


"""
    solve_spectre_file(spice_file; kwargs...)

Read in the given Spectre file, generate a CircuitIRODESystem, and pass it off to `solve_circuit()`
"""
function solve_spectre_file(spectre_file::String; include_dirs::Vector{String} = String[dirname(spectre_file)], kwargs...)
    circuit_code = CedarSim.make_spectre_circuit(
        CedarSim.SpectreNetlistParser.parsefile(spectre_file),
        include_dirs,
    );
    circuit = eval(circuit_code)
    invokelatest(circuit)
    return solve_circuit(circuit; kwargs...)
end

function solve_spectre_code(spectre_code::String; include_dirs::Vector{String} = String[], kwargs...)
    circuit_code = CedarSim.make_spectre_circuit(
        CedarSim.SpectreNetlistParser.parse(spectre_code),
        include_dirs,
    );
    fn = eval(circuit_code);
    invokelatest(fn)
    return solve_circuit(fn; kwargs...)
end

else
    # Phase 0: Stub implementations that error if called
    function solve_circuit(args...; kwargs...)
        error("solve_circuit requires DAECompiler (Phase 1+)")
    end
    function solve_spice_file(args...; kwargs...)
        error("solve_spice_file requires DAECompiler (Phase 1+)")
    end
    function solve_spice_code(args...; kwargs...)
        error("solve_spice_code requires DAECompiler (Phase 1+)")
    end
    function solve_spectre_file(args...; kwargs...)
        error("solve_spectre_file requires DAECompiler (Phase 1+)")
    end
    function solve_spectre_code(args...; kwargs...)
        error("solve_spectre_code requires DAECompiler (Phase 1+)")
    end
end  # if HAS_DAECOMPILER

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
function solve_mna_spice_code(spice_code::String; temp::Real=27.0)
    ast = CedarSim.SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    code = CedarSim.make_mna_circuit(ast)

    # Evaluate in temporary module
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
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
function solve_mna_spectre_code(spectre_code::String; temp::Real=27.0)
    ast = CedarSim.SpectreNetlistParser.parse(IOBuffer(spectre_code); start_lang=:spectre)
    code = CedarSim.make_mna_circuit(ast)

    # Evaluate in temporary module
    m = Module()
    Base.eval(m, :(using CedarSim.MNA))
    Base.eval(m, :(using CedarSim: ParamLens))
    Base.eval(m, :(using CedarSim.SpectreEnvironment))
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
function my_circuit(params, spec)
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
function rc_circuit(params, spec)
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

#=
# This is a useful define for ensuring that our SPICE simulations are giving reasonable answers
using NgSpice
function ngspice_simulate(spice_file::String, net::String; time_bounds=(0.0, 1.0))
    NgSpice.source(spice_file)
    NgSpice.cmd("tran $((time_bounds[2] - time_bounds[1])/1000) $(time_bounds[2])")
    return NgSpice.getvec(net)[3]
end
=#
