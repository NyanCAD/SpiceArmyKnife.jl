module CedarSim

using LinearAlgebra
using SparseArrays
using DiffEqBase
using OrdinaryDiffEq
using Printf
using ScopedValues

#=
Utility types and functions
=#

abstract type CedarException <: Exception end
Base.showerror(io::IO, ex::CedarException, bt; backtrace=true) = showerror(io, ex)

struct CedarError <: CedarException
    msg::String
end
Base.showerror(io::IO, err::CedarError) = println(io, err.msg)
cedarerror(msg) = throw(CedarError(msg))

# DefaultOr type for handling default vs user-specified parameters
struct DefaultOr{T}
    val::T
    is_default::Bool
    DefaultOr{T}(val::T, is_default::Bool) where {T} = new{T}(val, is_default)
    global mkdefault
    @eval DefaultOr(val::T, is_default::Bool) where {T} = $(Expr(:new, :(DefaultOr{T}), :val, :is_default))
    @eval mkdefault(t::T) where {T} = $(Expr(:new, :(DefaultOr{T}), :t, true))
end

Base.convert(::Type{DefaultOr{T}}, x::DefaultOr{T}) where {T} = x
Base.convert(::Type{DefaultOr{S}}, x::DefaultOr{T}) where {S, T} = DefaultOr{S}(convert(S, x.val), x.is_default)
Base.convert(::Type{DefaultOr{T}}, x::T) where {T} = DefaultOr{T}(x, false)
Base.convert(::Type{DefaultOr{T}}, x) where {T} = DefaultOr{T}(convert(T, x), false)
Base.convert(::Type{T}, x::DefaultOr) where {T} = Base.convert(T, x.val)
DefaultOr(d::DefaultOr, is_default::Bool) = DefaultOr(d.val, is_default)
DefaultOr(d::DefaultOr) = DefaultOr(d.val, true)
mknondefault(t::T) where {T} = DefaultOr{T}(t, false)
isdefault(x::DefaultOr) = x.is_default
undefault(x::DefaultOr) = x.val
undefault(x) = x

export DefaultOr, mkdefault, mknondefault, isdefault, undefault

#=
MNA Core - Modified Nodal Analysis
=#

include("mna.jl")
include("mna_devices.jl")

# Re-exports for MNA core
export MNACircuit, MNANet, MNAODESystem, DiodeModel
export resistor!, capacitor!, inductor!, vsource!, isource!, ground!, diode!
export vcvs!, vccs!
export solve_dc!, transient_problem, compile_circuit, compile_ode_system
export get_net!, get_branch!, node_index, branch_index
export stamp_conductance!, stamp_capacitance!, stamp_voltage_source!, stamp_current_source!
export stamp_inductor!, stamp_vcvs!, stamp_vccs!, stamp_diode!
export add_nonlinear_element!, add_time_source!

#=
MNA-Spectre Integration Layer
=#

include("mna_spectre.jl")

#=
SPICE/Spectre Netlist Parser
=#

include("mna_parser.jl")

# Re-exports for parser
export parse_spice, parse_spice_file, MNANetlist, build_circuit, simulate_netlist, @spice_str

#=
MNA Verilog-A Integration
=#

include("mna_veriloga.jl")

# Re-exports for Verilog-A integration
export VADevice, MNAVAResistor, varesistor!, MNAMosfet, nmos!, pmos!

# Re-exports for MNA-Spectre integration
export mna_circuit, MNANetRef, mna_net, mna_ground
export MNASimpleResistor, MNASimpleCapacitor, MNASimpleInductor
export SpcVoltageSource, SpcCurrentSource, MNASimpleDiode
export SpcVCVS, SpcVCCS, MNAGnd
export MNAParallelInstances, MNANamed
export build_mna_circuit, simulate_dc, simulate_tran
export mna_sim_mode, mna_spec, MNASimSpec

#=
Simulation Interface
=#

"""
    DCResult

Result of DC operating point analysis.
"""
struct DCResult
    circuit::MNACircuit
    solution::Vector{Float64}
    node_names::Vector{Symbol}
    branch_names::Vector{Symbol}
end

function Base.show(io::IO, result::DCResult)
    println(io, "DC Operating Point:")
    println(io, "  Node Voltages:")
    for (name, net) in result.circuit.nets
        if net.index > 0
            @printf(io, "    V(%s) = %.6g V\n", name, result.solution[net.index])
        end
    end
    if !isempty(result.circuit.branches)
        println(io, "  Branch Currents:")
        for (name, branch) in result.circuit.branches
            idx = result.circuit.num_nodes + branch.index
            @printf(io, "    I(%s) = %.6g A\n", name, result.solution[idx])
        end
    end
end

"""
    dc!(circuit::MNACircuit; kwargs...) -> DCResult

Perform DC operating point analysis.
"""
function dc!(circuit::MNACircuit; kwargs...)
    solution = solve_dc!(circuit; kwargs...)
    node_names = [name for (name, net) in circuit.nets if net.index > 0]
    branch_names = [name for (name, branch) in circuit.branches]
    return DCResult(circuit, solution, node_names, branch_names)
end

export dc!, DCResult

"""
    TransientResult

Result of transient simulation.
"""
struct TransientResult
    circuit::MNACircuit
    solution::Any  # ODE solution
    sys::MNAODESystem
end

function Base.show(io::IO, result::TransientResult)
    println(io, "Transient Analysis:")
    println(io, "  Time span: ", result.solution.t[1], " to ", result.solution.t[end])
    println(io, "  Time points: ", length(result.solution.t))
end

"""
    get_voltage(result::TransientResult, node::Symbol)

Get voltage waveform for a node.
"""
function get_voltage(result::TransientResult, node::Symbol)
    net = result.circuit.nets[node]
    if net.index == 0
        return zeros(length(result.solution.t))
    end
    return [u[net.index] for u in result.solution.u]
end

"""
    get_time(result::TransientResult)

Get time vector from transient result.
"""
get_time(result::TransientResult) = result.solution.t

export TransientResult, get_voltage, get_time

"""
    tran!(circuit::MNACircuit, tspan; u0=nothing, solver=nothing, kwargs...)

Perform transient simulation.
"""
function tran!(circuit::MNACircuit, tspan; u0=nothing, solver=nothing, kwargs...)
    f!, u0_calc, tspan_calc, mass_matrix, sys = transient_problem(circuit, tspan; u0=u0)

    # Choose solver
    if solver === nothing
        if mass_matrix !== nothing && !isdiag(mass_matrix)
            # DAE with mass matrix - use Rodas5 or similar
            solver = Rodas5()
        else
            # ODE - use TRBDF2
            solver = TRBDF2()
        end
    end

    # Create ODE problem
    if mass_matrix !== nothing
        prob = ODEProblem(
            ODEFunction(f!; mass_matrix=mass_matrix),
            u0_calc, tspan_calc;
            kwargs...
        )
    else
        prob = ODEProblem(f!, u0_calc, tspan_calc; kwargs...)
    end

    # Solve
    sol = solve(prob, solver)

    return TransientResult(circuit, sol, sys)
end

export tran!

#=
MNA SpectreEnvironment - MNA-compatible device aliases
This provides the same API as the original SpectreEnvironment
but uses MNA devices instead of DAECompiler devices.
=#

baremodule MNASpectreEnvironment

import ..Base
import ..CedarSim
import ..CedarSim: SpcVCVS, SpcVCCS

# Math functions
import Base:
    +, *, -, ==, !=, /, ^, >, <,  <=, >=,
    max, min, abs,
    log, exp, sqrt,
    sinh, cosh, tanh,
    sin, cos, tan,
    asinh, acosh, atanh,
    zero, atan,
    floor, ceil, trunc

const arctan = atan
const ln = log
const pow = ^
int(x) = trunc(Base.Int, x)
nint(x) = Base.round(Base.Int, x)

export !, +, *, -, ==, !=, /, ^, >, <,  <=, >=,
    max, min, abs,
    ln, log, exp, sqrt,
    sinh, cosh, tanh,
    sin, cos, tan, atan, arctan,
    asinh, acosh, atanh,
    int, nint, floor, ceil, pow

# MNA-compatible devices
const resistor = CedarSim.MNASimpleResistor
const capacitor = CedarSim.MNASimpleCapacitor
const inductor = CedarSim.MNASimpleInductor
const vsource = CedarSim.SpcVoltageSource
const isource = CedarSim.SpcCurrentSource
const diode = CedarSim.MNASimpleDiode
const vcvs = CedarSim.SpcVCVS
const vccs = CedarSim.SpcVCCS
const Gnd = CedarSim.MNAGnd

# Placeholder for unimplemented devices
struct UnimplementedDevice
    params
end
UnimplementedDevice(;kwargs...) = UnimplementedDevice(kwargs)
function (::UnimplementedDevice)(args...; dscope=Base.nothing)
    CedarSim.cedarerror("Unimplemented device")
end

const M_1_PI = 1/Base.pi

# Time-dependent sources (simplified for now)
function var"$time"()
    spec = CedarSim.mna_spec[]
    return spec.time
end

temper() = CedarSim.mna_spec[].temp

const dc = :dc
const ac = :ac
const tran = :tran

export resistor, capacitor, inductor, vsource, isource, diode, vcvs, vccs,
    UnimplementedDevice, M_1_PI, dc, ac, tran, var"$time", Gnd, temper

end # baremodule MNASpectreEnvironment

export MNASpectreEnvironment

#=
ParamLens infrastructure for hierarchical parameterization
(Simplified version for MNA integration)
=#

abstract type AbstractParamLens end

struct IdentityLens <: AbstractParamLens end
Base.getproperty(lens::IdentityLens, ::Symbol; type=:unknown) = lens
(::IdentityLens)(;kwargs...) = values(kwargs)
(::IdentityLens)(val) = val

struct ValLens{T} <: AbstractParamLens
    val::T
end
Base.getproperty(lens::ValLens, ::Symbol; type=:unknown) = cedarerror("Reached terminal lens")
(lens::ValLens)(val) = getfield(lens, :val)

"""
    ParamLens(::NamedTuple)

Takes a nested named tuple to override arguments.
"""
struct ParamLens{NT<:NamedTuple} <: AbstractParamLens
    nt::NT
    function ParamLens(nt::NT=(;)) where {NT<:NamedTuple}
        new{typeof(nt)}(nt)
    end
end

function Base.getproperty(🔍::ParamLens{T}, sym::Symbol; type=:unknown) where T
    nt = getfield(🔍, :nt)
    nnt = get(nt, sym, (;))
    if !isa(nnt, NamedTuple)
        return ValLens(nnt)
    end
    isempty(nnt) && return IdentityLens()
    return ParamLens(nnt)
end

function (🔍::ParamLens)(;kwargs...)
    nt = getfield(🔍, :nt)
    hasfield(typeof(nt), :params) || return values(kwargs)
    merge(values(kwargs), nt.params)
end

(🔍::ParamLens{typeof((;))})(val) = val

export AbstractParamLens, IdentityLens, ValLens, ParamLens

#=
Named device wrapper (for compatibility with spectre.jl codegen)
=#

struct Named{T}
    element::T
    name::Symbol
    Named(element::T, name::Union{String, Symbol}) where {T} = new{Core.Typeof(element)}(element, Symbol(name))
end

function (n::Named)(args...; kwargs...)
    return n.element(args...; dscope=n.name, kwargs...)
end

export Named

#=
ParallelInstances wrapper
=#

struct ParallelInstances
    device
    multiplier::Float64

    function ParallelInstances(device, multiplier::Float64)
        if multiplier < 0.0
            cedarerror("Cannot construct ParallelInstances with non-positive multiplier '$multiplier'")
        end
        return new(device, multiplier)
    end
end

ParallelInstances(device, multiplier::Number) = ParallelInstances(device, Float64(multiplier))
ParallelInstances(device, multiplier::DefaultOr) = ParallelInstances(device, undefault(multiplier))

function (pi::ParallelInstances)(nets...; kwargs...)
    nets_scaled = map(nets) do net
        if net isa MNANetRef
            MNANetRef(net, pi.multiplier)
        else
            net
        end
    end
    return pi.device(nets_scaled...; m=pi.multiplier, kwargs...)
end

export ParallelInstances

#=
spicecall - instantiate SPICE devices with parameter handling
=#

function spicecall(model; m=1.0, kwargs...)
    ParallelInstances(model(;kwargs...), m)
end

function spicecall(model::Type; m=1.0, kwargs...)
    ParallelInstances(model(;kwargs...), m)
end

export spicecall

#=
Convenience circuit building functions
=#

"""
    @circuit(block)

Macro for building circuits with a DSL.
"""
macro circuit(block)
    quote
        circuit = MNACircuit()
        $(esc(block))
        circuit
    end
end

export @circuit

#=
net() function for MNA circuits
=#

"""
    net(name::Union{Symbol, String, Nothing}=nothing)

Create a net in the current MNA circuit context.
"""
function net(name::Union{Symbol, String, Nothing}=nothing)
    ckt = mna_circuit[]
    if ckt !== nothing
        # MNA mode
        if name === nothing
            return mna_net()
        else
            return mna_net(name)
        end
    else
        error("No circuit context available. Use @with mna_circuit => circuit begin ... end")
    end
end

export net

#=
Precompilation
=#

using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # Simple RC circuit
        circuit = MNACircuit()
        vsource!(circuit, :vcc, :gnd; dc=5.0, name=:V1)
        resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
        capacitor!(circuit, :out, :gnd, 1e-6; name=:C1)
        ground!(circuit, :gnd)

        # DC solve
        result = dc!(circuit)
    end
end

end # module CedarSim
