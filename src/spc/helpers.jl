# Helper utilities for MNA SPICE/Spectre codegen
# Extracted from spectre.jl during DAECompiler removal

using SpectreNetlistParser
using SpectreNetlistParser: SpectreNetlistCSTParser, SPICENetlistParser
using .SPICENetlistParser: SPICENetlistCSTParser
using .SpectreNetlistCSTParser: SpectreNetlistSource
using .SPICENetlistCSTParser: SPICENetlistSource

# Type aliases for AST nodes
const SNode = SpectreNetlistCSTParser.Node
const SC = SpectreNetlistCSTParser
const SP = SPICENetlistCSTParser

# Case conversion helpers for netlist identifiers
# SPICE is case-insensitive (lowercase), Spectre is case-sensitive
LString(s::SNode{<:SP.Terminal}) = lowercase(String(s))
LString(s::SNode{<:SP.AbstractASTNode}) = lowercase(String(s))
LString(s::SNode{<:SC.Terminal}) = String(s)
LString(s::SNode{<:SC.AbstractASTNode}) = String(s)
LString(s::AbstractString) = lowercase(s)
LString(s::Symbol) = lowercase(String(s))
LSymbol(s) = Symbol(LString(s))

# Parameter checking helper
function hasparam(params, name)
    params === nothing && return false
    for p in params
        if LString(p.name) == name
            return true
        end
    end
    return false
end

# Note: getparam(params, name, default=nothing) and binning_rx are defined in codegen.jl

#=== ParamLens system for hierarchical parameter access ===#

abstract type AbstractParamLens end

# Helper to update struct properties
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

# Canonicalize nested parameter tuples
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

# Compact nested params back to flat structure
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

# Identity lens - passes through unchanged
struct IdentityLens <: AbstractParamLens; end
Base.getproperty(lens::IdentityLens, ::Symbol; type=:unknown) = lens
(::IdentityLens)(;kwargs...) = values(kwargs)
(::IdentityLens)(val) = val

# Value lens - terminal node with a fixed value
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

function Base.getproperty(lens::ParamLens{T}, sym::Symbol; type=:unknown) where T
    nt = getfield(lens, :nt)
    nnt = get(nt, sym, (;))
    if !isa(nnt, NamedTuple)
        return ValLens(nnt)
    end
    isempty(nnt) && return IdentityLens()
    return ParamLens(nnt)
end

function (lens::ParamLens)(;kwargs...)
    nt = getfield(lens, :nt)
    hasfield(typeof(nt), :params) || return values(kwargs)
    merge(values(kwargs), nt.params)
end

(::ParamLens{typeof((;))})(val) = val
@generated function (lens::AbstractParamLens)(val)
    isprimitivetype(val) && error("Should have reached trivial lens before this point")
    Expr(:new, :(typeof(val)), (:(getproperty(lens, $(QuoteNode(name)))(getfield(val, $(QuoteNode(name))))) for name in fieldnames(val))...)
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

function Base.propertynames(obs::ParamObserver)
    return [fieldnames(ParamObserver)..., keys(getfield(obs, :params))...]
end

function Base.getproperty(obs::ParamObserver, sym::Symbol; type=nothing)
    if sym == :params
        return NamedTuple(getfield(obs, :params)[:params])
    elseif hasfield(ParamObserver, sym)
        return getfield(obs, sym)
    end
    dict = getfield(obs, :params)
    get!(dict, sym, ParamObserver(sym, type))
end

function (obs::ParamObserver)(;kwargs...)
    dict = get!(getfield(obs, :params), :params, Dict{Symbol, Number}())
    for (param, value) in kwargs
        get!(dict, param, value)
    end
    return (; (k=>dict[k] for k in keys(kwargs))...)
end

function Base.convert(to::Type{NamedTuple}, obs::ParamObserver)
    make_nt(getfield(obs, :params))
end

make_nt(x) = x
make_nt(x::ParamObserver) = make_nt(getfield(x, :params))
make_nt(dict::Dict) = (; (k => make_nt(v) for (k, v) in dict)...)

# Pretty print helper
function pretty_print(io, v, indent; header="")
    s = string(v)
    m = match(r"^NamedTuple\{.*?\}\((.+)\)", s)
    @assert m !== nothing s
    print(io, header, "(", m.captures[1], ")")
end
function pretty_print(io, dict::Dict, indent; header="")
    print(io, header, "(")
    first = true
    for (k, v) in dict
        if first
            first = false
        else
            print(io, ",")
        end
        print(io, "\n", " "^(indent+2))
        if v isa Dict
            pretty_print(io, v, indent+2; header="$k = ")
        elseif v isa ParamObserver
            pretty_print(io, getfield(v, :params), indent+2; header="$k::$(getfield(v, :type)) = ")
        else
            print(io, "$k = ", string(v))
        end
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", obs::ParamObserver; indent=0)
    print(io, "(ParamObserver) $(getfield(obs, :name))::$(getfield(obs, :type)) ")
    pretty_print(io, getfield(obs, :params), 0)
    print(io, "\n")
end

# Accessor macro for parameter paths like @param(observer.x1.rload)
macro param(path)
    esc(Expr(:., Expr(:., path.args[1], QuoteNode(:params)), path.args[2]))
end
