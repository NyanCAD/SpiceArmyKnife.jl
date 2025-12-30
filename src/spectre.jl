using SpectreNetlistParser
using SpectreNetlistParser: SpectreNetlistCSTParser, SPICENetlistParser
using .SPICENetlistParser: SPICENetlistCSTParser
using .SpectreNetlistCSTParser:
    SpectreNetlistSource
using .SPICENetlistCSTParser:
    SPICENetlistSource
using Base.Meta
using StaticArrays
using DecFP

const SNode = SpectreNetlistCSTParser.Node

const SC = SpectreNetlistCSTParser
const SP = SPICENetlistCSTParser

LString(s::SNode{<:SP.Terminal}) = lowercase(String(s))
LString(s::SNode{<:SP.AbstractASTNode}) = lowercase(String(s))
LString(s::SNode{<:SC.Terminal}) = String(s)
LString(s::SNode{<:SC.AbstractASTNode}) = String(s)
LString(s::AbstractString) = lowercase(s)
LString(s::Symbol) = lowercase(String(s))
LSymbol(s) = Symbol(LString(s))

# Phase 0: LineNumberNode is already defined in SpectreNetlistParser/src/parse/errors.jl
# Removed duplicate definition to avoid method overwriting error


abstract type AbstractParamLens end

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
        #nnt = canonicalize_params(nt)
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

function ParamSim(circuit::Type, mode, spec, params)
    return ParamSim(ApplyLens(circuit), mode, spec, params)
end

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

noiseparams(circ::AbstractSim) = noiseparams(circ.circuit)
function noiseparams(circ)
    observer = ParamObserver()
    circ(observer)
    noiseparams(observer)
end
function noiseparams(👀::ParamObserver)
    t = something(getfield(👀, :type), Nothing)
    noisefields = filter((fn -> startswith(String(fn), "ϵ"), modelfields(typeof(t)))...)
    args = getfield(👀, :params)
    childfields = []
    for (k, v) in args
        if typeof(v) <: ParamObserver
            fields = noiseparams(v)
            if !isempty(fields)
                push!(childfields, (k => fields))
            end
        end
    end
    (; (f => 0.0 for f in noisefields)...,
        childfields...)
end

function spice_select_device(devkind, level, version, stmt; dialect=:ngspice)
    if devkind == :d
        return :(SpectreEnvironment.diode)
    elseif devkind == :r
        return :(SpectreEnvironment.resistor)
    elseif devkind == :c
        return :(SpectreEnvironment.capacitor)
    end
    if dialect == :ngspice
        if devkind in (:pmos, :nmos)
            if level == 5
                #error("bsim2 not supported")
                #return :bsim2
            elseif level == 8 || level == 49
                #error("bsim3 not supported")
                #return :bsim3
            elseif level == 14 || level == 54
                return :bsim4
            elseif level == 17 || level == 72
                if version == 107 || version === nothing
                    return :bsimcmg107
                else
                    file = stmt.ps.srcfile.path
                    line = SpectreNetlistParser.LineNumbers.compute_line(stmt.ps.srcfile.lineinfo, stmt.startof)
                    @warn "Version $version of mosfet $devkind at level $level not implemented" _file=file _line=line
                    return :UnimplementedDevice
                end
            else
                file = stmt.ps.srcfile.path
                line = SpectreNetlistParser.LineNumbers.compute_line(stmt.ps.srcfile.lineinfo, stmt.startof)
                @warn "Mosfet $devkind at level $level not implemented" _file=file _line=line
                return :UnimplementedDevice
            end
        elseif devkind == :sw
            return :(SpectreEnvironment.Switch)
        end
    end
    file = stmt.ps.srcfile.path
    line = SpectreNetlistParser.LineNumbers.compute_line(stmt.ps.srcfile.lineinfo, stmt.startof)
    @warn "Device $devkind at level $level not implemented" _file=file _line=line
    return :UnimplementedDevice
end

function devtype_param(model_kind, mosfet_kind)
    if model_kind == :bsim4
        return :TYPE => (mosfet_kind == :pmos ? -1 : 1)
    elseif startswith(String(model_kind), "bsimcmg")
        return :DEVTYPE => (mosfet_kind == :pmos ? 0 : 1)
    elseif model_kind == :UnimplementedDevice
        # skip
        return nothing
    else
        error("Needs to be filled in per model")
    end
end

function hasparam(params, name)
    for p in params
        if LString(p.name) == name
            return true
        end
    end
    return false
end

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
    #_uppercase_kwargs(model, kwargs)::NamedTuple{<:Any, types}
    _case_adjust_kwargs(model, kwargs)::ParsedNT
end

Base.@assume_effects :total function case_adjust_kwargs(model::Type, kwargs::NamedTuple)
    #_uppercase_kwargs(model, kwargs)::NamedTuple{<:Any, types}
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

# Used by aliasextract.jl for setting up debug scopes for net names
function net_alias(net, name)
    observed!(net.V, DScope(debug_scope[], name))
end

function modify_spice(io::IO, node::SNode, nt::NamedTuple, startof)
    params = get(nt, :params, NamedTuple())
    for childnode in AbstractTrees.children(node)
        childnode === nothing && continue
        if (childnode isa SNode{<:SP.Parameter}
         && hasproperty(params, LSymbol(childnode.name))
         && getproperty(params, LSymbol(childnode.name)) isa Number
         && childnode.val !== nothing)
            val = getproperty(params, LSymbol(childnode.name))
            e = childnode.val
            endoff = e.startof+e.expr.off-1
            SpectreNetlistParser.RedTree.print_contents(io, node.ps, startof, endoff)
            print(io, val)
            startof = e.startof+e.expr.off+e.expr.width
        elseif isa(childnode, SNode{<:SP.Subckt}) ||
               isa(childnode, SNode{SP.Model}) ||
               isa(childnode, SNode{SP.SubcktCall}) ||
               isa(childnode, SNode{SP.MOSFET}) ||
               isa(childnode, SNode{SP.Capacitor}) ||
               isa(childnode, SNode{SP.Diode}) ||
               isa(childnode, SNode{SP.BipolarTransistor}) ||
               isa(childnode, SNode{SP.Voltage}) ||
               isa(childnode, SNode{SP.Current}) ||
               isa(childnode, SNode{SP.Resistor}) ||
               isa(childnode, SNode{SP.Inductor}) ||
               isa(childnode, SNode{SC.Model}) ||
               isa(childnode, SNode{SC.Instance})
            chnt = get(nt, LSymbol(childnode.name), NamedTuple())
            startof = modify_spice(io, childnode, chnt, startof)
        else
            startof = modify_spice(io, childnode, nt, startof)
        end
    end
    startof
end

function alter(io::IO, node::SNode, nt::NamedTuple)
    startof=node.startof+node.expr.off
    startof = modify_spice(io, node, canonicalize_params(nt), startof)
    endoff = node.startof+node.expr.off+node.expr.width-1
    SpectreNetlistParser.RedTree.print_contents(io, node.ps, startof, endoff)
end

"""
    alter([io], ast; kwargs...)
    alter([io], ast, nt::ParamSim)
    alter([io], ast, nt::ParamLens)

Print a netlist with the given parameters substituted.
Parameters in subcircuits can be passed as named tuples.
"""
alter(node::SNode; kwargs...) = alter(stdout, node, values(kwargs))
alter(node::SNode, nt::ParamSim) = alter(stdout, node, nt.params)
alter(node::SNode, nt::ParamLens) = alter(stdout, node, getfield(nt, :nt))
alter(io::IO, node::SNode; kwargs...) = alter(io, node, values(kwargs))
alter(io::IO, node::SNode, nt::ParamSim) = alter(io, node, nt.params)
alter(io::IO, node::SNode, nt::ParamLens) = alter(io, node, getfield(nt, :nt))


struct SpectreParseError
    sa
end

#TODO not implemented yet
Base.show(io::IO, sap::SpectreParseError) = SpectreNetlistCSTParser.visit_errors(sap.sa; io)
