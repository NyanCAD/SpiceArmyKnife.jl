using ChainRulesCore, StaticArrays

# Phase 0: Use stubs instead of DAECompiler
@static if CedarSim.USE_DAECOMPILER
    using DAECompiler: time_periodic_singularity!
else
    using ..DAECompilerStubs: time_periodic_singularity!
end

# This gets imported by all generated Spectre code. The function names
# exported here should correspond to what is made available by Spectre

struct PWLConstructError <: CedarException
    ts
    ys
end

function Base.showerror(io::IO, err::PWLConstructError)
    println(io, "PWL must have an equal number of x and y values")
end

function find_t_in_ts(ts, t)
    idx = Base.searchsortedfirst(ts, t)
    if idx <= length(ts) && ts[idx] == t
        return idx + 1
    end
    return idx
end

# Disable autodiff through `t` search
function ChainRulesCore.frule((_, _), ::typeof(find_t_in_ts), ts, t)
    return find_t_in_ts(ts, t), ZeroTangent()
end


rem_right_semi(t, r) = t % r
function ChainRulesCore.frule((_, δt, δr), ::typeof(rem_right_semi), t, r)
    return (rem_right_semi(t, r), δt)
end

# Split our `wave` into `ts` and `ys`, hinting to the compiler what the length
# of these views are.
function wave_split(wave::SVector)
    idxs = SVector{div(length(wave),2)}(1:2:length(wave))
    ts = @view(wave[idxs])
    ys = @view(wave[idxs.+1])
    return (ts, ys)
end

function pwl_at_time(ts, ys, t)
    if length(ts) != length(ys)
        throw(CedarSim.PWLConstructError(ts, ys))
    end
    i = find_t_in_ts(ts, t)
    type_stable_time = 0. * t
    if i <= 1
        # Signal is before the first timepoint, hold the first value.
        return ys[1] + type_stable_time
    end
    if i > length(ts)
        # Signal is beyond the final timepoint, hold the final value.
        return ys[end] + type_stable_time
    end
    if ys[i-1] == ys[i]
        # signal is constant/flat (singularity in y)
        return ys[i] + type_stable_time
    end
    if ts[i] == ts[i-1]
        # signal is infinitely steep (singularity in t)
        # This can occur when loading a serialized signal that had its time digits truncated.
        return (ys[i-1] + ys[i])/2 + type_stable_time
    end
    # The general case, where we must perform linear interpolation
    slope = (ys[i] - ys[i-1])/(ts[i] - ts[i-1])
    return ys[i-1] + (t - ts[i-1])*slope
end

@generated function time_periodic_singularities!(ts::StaticArrays.SVector, period = ts[end], count = 1)
    body = Expr(:block)
    for i in 1:length(ts) # length of the type!
        # Phase 0: Use imported time_periodic_singularity! instead of DAECompiler qualified name
        push!(body.args, :(time_periodic_singularity!(ts[$i], period, count)))
    end
    return body
end

#==============================================================================#
# Parameter Lens System
#
# Runtime utilities for hierarchical parameter access and override.
# Used by MNA codegen for parameter sweeps and subcircuit instantiation.
#==============================================================================#

setproperties(obj, nt::@NamedTuple{}) = obj
@generated function setproperties(obj, nt::NamedTuple)
    T = obj
    values = Expr[]
    for fieldname in fieldnames(T)
        if fieldname in fieldnames(nt)
            push!(values, :(nt.$fieldname))
        else
            push!(values, :(obj.$fieldname))
        end
    end
    return :($(T.name.wrapper)($(values...)))
end
function setproperties(obj; kw...)
    setproperties(obj, (;kw...))
end

@generated function canonicalize_params(nt::NamedTuple)
    par = []
    ch = []
    for p in fieldnames(nt)
        if p == :params
            append!(par, [:($pp=nt.params.$pp) for pp in fieldnames(fieldtype(nt, :params))])
        elseif fieldtype(nt, p) <: Number
            push!(par, :($p=nt.$p))
        elseif fieldtype(nt, p) <: NamedTuple
            push!(ch, :($p=canonicalize_params(nt.$p)))
        end
    end
    return quote
        (; params=(;$(par...)), $(ch...))
    end
end

function canonicalize_params(p::Dict)
    res = empty(p)
    for (k, v) in p
        if v isa Dict
            res[k] = canonicalize_params(v)
        elseif v isa NamedTuple
            res[k] = canonicalize_params(Dict(pairs(v)))
        elseif v isa Number
            get!(res, :params, Dict{Symbol, Number}())[k] = v
        end
    end
    res
end

@generated function compact_params(nt::NamedTuple)
    par = []
    ch = []
    for p in fieldnames(nt)
        if p == :params
            for pp in fieldnames(fieldtype(nt, :params))
                if pp in fieldnames(nt)
                    push!(par, :($pp=nt.params.$pp))
                else
                    push!(ch, :($pp=nt.params.$pp))
                end
            end
        elseif fieldtype(nt, p) <: Number
            push!(ch, :($p=nt.$p))
        elseif fieldtype(nt, p) <: NamedTuple
            push!(ch, :($p=compact_params(nt.$p)))
        end
    end
    if isempty(par)
        return quote
            (; $(ch...))
        end
    else
        return quote
            (; params=(;$(par...)), $(ch...))
        end
    end
end

abstract type AbstractParamLens end

struct IdentityLens <: AbstractParamLens; end
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
For example `sweep.foo(bar=1)` by default returns `(bar=1,)`
unless `ParamLens((foo=(bar=2,),))` is used, in which case it'll return `(bar=2,)`
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
@generated function (🔍::AbstractParamLens)(val)
    isprimitivetype(val) && error("Should have reached trivial lens before this point")
    Expr(:new, :(typeof(val)), (:(getproperty(🔍, $(QuoteNode(name)))(getfield(val, $(QuoteNode(name))))) for name in fieldnames(val))...)
end

struct ApplyLens{T}
    circuit::T
    ApplyLens(circuit) = new{Core.Typeof(circuit)}(circuit)
end
(apply::ApplyLens)(🔍::AbstractParamLens) = (🔍(apply.circuit()))()

"""
    ParamObserver()

An "observer" lens that, when passed to a circuit, collects the hierarchy
of requested parameters and their default values.
"""
struct ParamObserver <: AbstractParamLens
    name::Symbol
    type::Any
    params::Dict{Symbol, Any}
end
ParamObserver(name=:top, type=nothing; kwargs...) = ParamObserver(name, type, canonicalize_params(Dict{Symbol, Any}(kwargs...)))

function Base.propertynames(👀::ParamObserver)
    return [fieldnames(ParamObserver)..., keys(getfield(👀, :params))...]
end

function Base.getproperty(👀::ParamObserver, sym::Symbol; type=nothing)
    # unlike the lens, this allows access to properties as well
    if sym == :params
        return NamedTuple(getfield(👀, :params)[:params])
    elseif hasfield(ParamObserver, sym)
        return getfield(👀, sym)
    end
    # By default, look up a subcircuit
    dict = getfield(👀, :params)
    get!(dict, sym, ParamObserver(sym, type))
end

function (👀::ParamObserver)(;kwargs...)
    # Look up a set of local parameters
    dict = get!(getfield(👀, :params), :params, Dict{Symbol, Number}())
    for (param, value) in kwargs
        get!(dict, param, value)
    end
    return (; (k=>dict[k] for k in keys(kwargs))...)
end

function Base.show(io::IO, ::MIME"text/plain", 👀::ParamObserver; indent=0)
    print(io, "(ParamObserver) $(getfield(👀, :name))::$(getfield(👀, :type)) ")
    pretty_print(io, getfield(👀, :params), 0)
    print(io, "\n")
end

make_nt(x) = x
make_nt(x::ParamObserver) = make_nt(getfield(x, :params))
make_nt(dict::Dict) = (; (k => make_nt(v) for (k, v) in dict)...)
function Base.convert(to::Type{NamedTuple}, 👀::ParamObserver)
    return compact_params(make_nt(👀))
end

function pretty_print(io::IO, d::Dict, indent = 0; Δindent = 4)
    outerpadding = " " ^ (indent)
    padding = " " ^ (indent + Δindent)
    println(io, "(;")
    for (k,v) in sort(d; by=x->x==:params, rev=true)
        print(io,  padding * string(k) * " = ")
        if v isa ParamObserver
            print(io, "(ParamObserver) $(getfield(v, :name))::$(nameof(getfield(v, :type))) ")
            pretty_print(io, getfield(v, :params), indent + Δindent)
        elseif v isa Dict
            pretty_print(io, v, indent + Δindent)
        else
            print(io, v)
        end
        println(io, ",")
    end
    print(io, outerpadding * ")")
    return nothing
end

macro param(path)
    esc(Expr(:., Expr(:., path.args[1], QuoteNode(:params)), path.args[2]))
end

export @param

function fieldvalues(x::T) where {T}
     !isstructtype(T) && throw(ArgumentError("$(T) is not a struct type"))

     return ((CedarSim.undefault(getfield(x, name)) for name in fieldnames(T))...,)
end

function ntfromstruct(x::T) where {T}
     !isstructtype(T) && throw(ArgumentError("$(T) is not a struct type"))
     names = fieldnames(T)
     values = fieldvalues(x)
     return NamedTuple{names}(values)
end
ntfromstruct(x::CedarSim.ParallelInstances) = (; m=x.multiplier, ntfromstruct(x.device)...)

function modelparams(m)
    args = m.params
    t = m.type
    i = spicecall(t; NamedTuple(args)...)
    ntfromstruct(i)
end

# source: https://rosettacode.org/wiki/Topological_sort#Julia
function toposort(data::Dict{T,Set{T}}) where T
    selfdeps = Set{T}()
    for (k, v) in data
        if k ∈ v
            push!(selfdeps, k)
            delete!(v, k)
        end
    end
    extdeps = setdiff(reduce(∪, values(data); init=Set{T}()), keys(data))
    for item in extdeps
        data[item] = Set{T}()
    end
    rst = Vector{T}()
    while true
        ordered = Set(item for (item, dep) in data if isempty(dep))
        if isempty(ordered) break end
        append!(rst, ordered)
        data = Dict{T,Set{T}}(item => setdiff(dep, ordered) for (item, dep) in data if item ∉ ordered)
    end
    isempty(data) || cedarerror("a cyclic dependency exists amongst $(data)")
    # items that depend on external items or on themselves
    # (meaning, on the same item in the parent scope)
    extraitems = union(selfdeps, extdeps)
    return extraitems, rst
end

#==============================================================================#
# Model Types
#
# ParsedModel, BinnedModel, and spicecall for device instantiation.
#==============================================================================#

struct BinnedModel{B<:Tuple}
    scale::Float64
    bins::B
    BinnedModel(scale, bins::B) where B = new{B}(float(scale), bins)
end

const ParsedNT = NamedTuple{names, types} where {names, types<:Tuple{Vararg{Union{DefaultOr{Int}, DefaultOr{Float64}, DefaultOr{Bool}}}}}
struct ParsedModel{T}
    model::T
end
function ParsedModel(model, kwargs)
    ParsedModel{model}(model(;kwargs...))
end

Base.show(io::IO, m::ParsedModel) = print(io, "ParsedModel($(m.model), ...)")
Base.nameof(m::ParsedModel{T}) where T = nameof(T)
Base.nameof(m::BinnedModel) = nameof(first(m.bins))

modelfields(m) = ()
modelfields(m::DataType) = fieldnames(m)
modelfields(::Type{ParsedModel{T}}) where T = modelfields(T)
modelfields(::Type{BinnedModel{T}}) where T = modelfields(eltype(T))

Base.@assume_effects :foldable function case_adjust_kwargs_fallback(model::Type{T}, kwargs::NamedTuple{Names}) where {Names, T}
    case_insensitive = Dict(Symbol(lowercase(String(kw))) => kw for kw in fieldnames(T))
    pairs = Pair[]
    for kw in (Names::Tuple{Vararg{Symbol}})
        push!(pairs, get(case_insensitive, Symbol(lowercase(String(kw))), kw)=>getfield(kwargs, kw))
    end
    (; pairs...)
end

function _case_adjust_kwargs(model::Type{T}, kwargs::NamedTuple{Names}) where {Names, T}
    if @generated
        case_insensitive = Dict(Symbol(lowercase(String(kw))) => kw for kw in fieldnames(T))
        return :((;$(map(Names) do kw
            Expr(:kw,
                get(case_insensitive, Symbol(lowercase(String(kw))), kw),
                Expr(:call, :getfield, :kwargs, quot(Symbol(kw))))
        end...)))
    else
        return case_adjust_kwargs_fallback(model, kwargs)
    end
end

"""
    case_adjust_kwargs(model, kwargs)

Adjust the case of `kwargs` (which are assumed to be all lowercase) to match the
case of the fieldnames of `model`.
"""
Base.@assume_effects :total function case_adjust_kwargs(model::Type, kwargs::ParsedNT)
    _case_adjust_kwargs(model, kwargs)::ParsedNT
end

Base.@assume_effects :total function case_adjust_kwargs(model::Type, kwargs::NamedTuple)
    _case_adjust_kwargs(model, kwargs)::NamedTuple
end

function (pm::ParsedModel)(;kwargs...)
    setproperties(pm.model, values(kwargs))
end

struct NoBinExpection <: CedarException
    bm::BinnedModel
    l::Float64
    w::Float64
end
Base.showerror(io::IO, bin::NoBinExpection) = print(io, "NoBinExpection: no bin for BinnedModel $(typeof(bin.bm)) of size (l=$(bin.l), w=$(bin.w)).")

Base.@assume_effects :consistent :effect_free :terminates_globally @noinline function find_bin(bm::BinnedModel, l, w)
    l = bm.scale*l
    w = bm.scale*w
    for bin in bm.bins
        (; LMIN, LMAX, WMIN, WMAX) = bin.model
        if undefault(LMIN::DefaultOr{Float64}) <= l < undefault(LMAX::DefaultOr{Float64}) && undefault(WMIN::DefaultOr{Float64}) <= w < undefault(WMAX::DefaultOr{Float64})
            return bin
        end
    end
    throw(NoBinExpection(bm, l, w))
end

function (bm::BinnedModel)(; l, w, kwargs...)
    find_bin(bm, l, w)(; l, w, kwargs...)
end

"Instantiate a model using SPICE case insensitive semantics"
function spicecall(model; m=1.0, kwargs...)
    ParallelInstances(model(;kwargs...), m)
end

@Base.assume_effects :foldable function mknondefault_nt(nt::NamedTuple)
    if @generated
        names = Base._nt_names(nt)
        types = Any[]
        args = Any[]
        for i = 1:length(names)
            T = fieldtype(nt, i)
            arg = :(getfield(nt, $i))
            if T <: DefaultOr
                push!(args, arg)
            else
                push!(args, Expr(:new, DefaultOr{T}, arg, false))
                T = DefaultOr{T}
            end
            push!(types, T)
        end
        nttypes = Tuple{types...}
        Expr(:new, :(NamedTuple{$names, $nttypes}), args...)
    else
        map(mknondefault, nt)
    end
end

function spicecall(pm::ParsedModel{T}; m=1, kwargs...) where T
    instkwargs = case_adjust_kwargs(T, mknondefault_nt(values(kwargs)))::ParsedNT
    inst = setproperties(pm.model, instkwargs)
    ParallelInstances(inst, m)
end

function spicecall(bm::BinnedModel; l, w, kwargs...)
    spicecall(find_bin(bm, l, w); l, w, kwargs...)
end

spicecall(::Type{ParsedModel}, model, kwargs) = ParsedModel(model, case_adjust_kwargs(model, kwargs))

export AbstractParamLens, ParamLens, IdentityLens, ValLens, ParamObserver, ApplyLens
export ParsedModel, BinnedModel, spicecall, setproperties

#==============================================================================#
# SpectreEnvironment Baremodule
#==============================================================================#

baremodule SpectreEnvironment

import ..Base
import ..CedarSim
import ..CedarSim: vcvs, vccs, Switch
import ForwardDiff
import Compat
import Distributions
import StaticArrays

import Base:
    +, *, -, ==, !=, /, ^, >, <,  <=, >=,
    max, min, abs,
    log, exp, sqrt,
    sinh, cosh, tanh,
    sin, cos, tan,
    asinh, acosh, atanh,
    zero, atan,
    floor, ceil, trunc
import Base.Experimental: @overlay
import ..rem_right_semi, ..time_periodic_singularities!, ..pwl_at_time, ..wave_split

const arctan = atan
const ln = log
const pow = ^
int(x) = trunc(Int, x)
nint(x) = Base.round(Int, x)

export !, +, *, -, ==, !=, /, ^, >, <,  <=, >=,
    max, min, abs,
    ln, log, exp, sqrt,
    sinh, cosh, tanh,
    sin, cos, tan, atan, arctan,
    asinh, acosh, atanh,
    int, nint, floor, ceil, pow


const resistor = CedarSim.SimpleResistor
const capacitor = CedarSim.SimpleCapacitor
const inductor = CedarSim.SimpleInductor
const vsource = CedarSim.VoltageSource
const isource = CedarSim.CurrentSource
const diode = CedarSim.SimpleDiode
const UnimplementedDevice = CedarSim.UnimplementedDevice
const Gnd = CedarSim.Gnd

# bsource is weird. It can basically be any circuit element.
# This maps to the appropriate element, based on the keyword arguments
function bsource(;kwargs...)
    keys = Base.keys(kwargs)
    if Base.in(:v, keys)
        return vsource(tran=kwargs[:v])
    elseif Base.in(:i, keys)
        return isource(tran=kwargs[:i])
    elseif Base.in(:r, keys)
        return resistor(r=kwargs[:r])
    elseif Base.in(:c, keys)
        return capacitor(c=kwargs[:c])
    else
        cedarerror("BSOURCE with args $kwargs not supported.")
    end
end

const M_1_PI = 1/Base.pi

function pwl(wave)
    ts, ys = wave_split(wave)
    # Notify singularities at each of our timepoints
    time_periodic_singularities!(ts)

    # Actually calculate the value to return
    return pwl_at_time(ts, ys, var"$time"())
end

function pulse(v1, v2, td, tr, tf, pw=Base.Inf, period=Base.Inf, count=-1)
    ts = StaticArrays.@SVector[
        td, td+tr, td+tr+pw, td+tr+pw+tf,
    ]
    ys = StaticArrays.@SVector[
        v1, v2, v2, v1,
    ]
    # Notify singularities at each of our timepoints, repeat forever
    time_periodic_singularities!(ts, period, count)

    # Calculate value modulo our period
    t = rem_right_semi(CedarSim.spec[].time, period)
    return pwl_at_time(ts, ys, t)
end

# don't pirate Base.sin
function spsin(vo, va, freq, td=0, theta=0, phase=0, ncyles=Base.Inf)
    # see https://ltwiki.org/LTspiceHelp/LTspiceHelp/V_Voltage_Source.htm
    if td < var"$time"() < ncyles/freq
        vo+va*Base.exp(-(var"$time"()-td)*theta)*Base.sind(360*freq*(var"$time"()-td)+phase)
    else
        vo + va*Base.sind(phase)
    end
end

function agauss(nom, avar, sigma)
    rng = CedarSim.spec[].rng
    if rng === nothing
        nom
    else
        d = Distributions.Normal(0.0, avar)
        rn = Base.@noinline Base.rand(rng, d)
        nom + rn/sigma
    end
end

# Gets replaced by simulator time in the compiler override
function var"$time"()
    if CedarSim.sim_mode[] === :dcop || CedarSim.sim_mode[] === :tranop
        return 0.0
    else
        return CedarSim.spec[].time
    end
end

temper() = CedarSim.undefault(CedarSim.spec[].temp) # Celsius

const dc = :dc
const ac = :ac
const tran = :tran
export resistor, capacitor, inductor, vsource, isource, bsource, vcvs, vccs, UnimplementedDevice,
    M_1_PI, dc, ac, tran, pwl, pulse, spsin, var"$time", Gnd, agauss, temper

end # baremodule SpectreEnvironment

export SpectreEnvironment
