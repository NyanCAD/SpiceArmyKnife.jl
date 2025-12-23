using StaticArrays

struct CodegenState
    sema::SemaResult
end

# LString and LSymbol are defined in spectre.jl
# LineNumberNode is defined in SpectreNetlistCSTParser
# hasparam is defined in spectre.jl

function cg_net_name!(state::CodegenState, net)
    return LSymbol(net)
end

function cg_net_name!(state::CodegenState, net::Symbol)
    # *net# is an invalid identifier in both spice and julia, so we can use it without worrying about colliding
    # with spliced expressions
    is_ambiguous(state.sema, net) ? Symbol(string("*net#", net)) : net
end

function cg_model_name!(state::CodegenState, model)
    return LSymbol(model)
end

function cg_model_name!(state::CodegenState, model::Symbol)
    # *net# is an invalid identifier in both spice and julia, so we can use it without worrying about colliding
    # with spliced expressions
    is_ambiguous(state.sema, model) ? Symbol(string("*model#", model)) : model
end

#=============================================== Expressions ===================================================#
"""
    cg_expr!(state, expr::SNode)

Codegen a SPICE or Spectre expression `expr` to julia.
"""
function cg_expr! end

# Phase 0: NumberLiteral is now the leaf node (FloatLiteral/IntLiteral don't exist)
function cg_expr!(state::CodegenState, cs::SNode{SC.NumberLiteral})
    txt = String(cs)
    sf = d"1"
    if !isempty(txt) && txt[end] ∈ keys(spectre_magnitudes)
        sf = spectre_magnitudes[txt[end]]
        txt = txt[begin:end-1]
    end
    # Try to parse as integer first, then as float
    ret = tryparse(Int64, txt)
    if ret !== nothing
        return ret * Float64(sf)
    end
    ret = Base.parse(Dec64, txt)
    ret *= sf
    return Float64(ret)
end

using DecFP
const spectre_magnitudes = Dict(
    'T' => d"1e12",
    'G' => d"1e9",
    'M' => d"1e6",
    'K' => d"1e3",
    'k' => d"1e3",
    '_' => d"1",
    '%' => d"1e-2",
    'c' => d"1e-2",
    'm' => d"1e-3",
    'u' => d"1e-6",
    'n' => d"1e-9",
    'p' => d"1e-12",
    'f' => d"1e-15",
    'a' => d"1e-18",
);

const spice_magnitudes = Dict(
    "t" => d"1e12",
    "g" => d"1e9",
    "meg" => d"1e6",
    "k" => d"1e3",
    "m" => d"1e-3",
    "u" => d"1e-6",
    "mil" => d"25.4e-6",
    "n" => d"1e-9",
    "p" => d"1e-12",
    "f" => d"1e-15",
    "a" => d"1e-18",
);
# Match magnitude with optional trailing characters (mAmp, MegQux, etc.)
# Longest prefixes first to avoid "m" matching before "meg" or "mil"
const spice_magnitude_order = ["meg", "mil", "t", "g", "k", "m", "u", "n", "p", "f", "a"]
const spice_regex = Regex("($(join(spice_magnitude_order, "|")))")

const binning_rx = r"(.*)\.([0-9]+)"

# Phase 0: NumberLiteral is now the leaf node (FloatLiteral/IntLiteral don't exist)
function cg_expr!(state::CodegenState, cs::SNode{SP.NumberLiteral})
    txt = lowercase(String(cs))
    sf = d"1"

    # Find where the numeric part ends (scientific notation: digits, '.', 'e', '+', '-')
    num_end = 1
    for (i, c) in enumerate(txt)
        if isdigit(c) || c == '.' || c == 'e' || c == '+' || c == '-'
            num_end = i
        else
            break
        end
    end

    # Check if there's a magnitude suffix after the number
    suffix = txt[num_end+1:end]
    if !isempty(suffix)
        # Try to match a known magnitude at the START of the suffix
        for mag in spice_magnitude_order
            if startswith(suffix, mag)
                suffix_after = suffix[length(mag)+1:end]
                # Don't match single-letter magnitudes if they're part of a unit name
                is_valid = true
                if length(mag) == 1
                    if startswith(suffix_after, "mp") || startswith(suffix_after, "hm") ||
                       startswith(suffix_after, "arad") || startswith(suffix_after, "enry")
                        is_valid = false
                    end
                end
                if is_valid
                    sf = spice_magnitudes[mag]
                end
                break
            end
        end
    end

    # Extract just the numeric part
    txt = txt[begin:num_end]

    # Try to parse as integer first, then as float
    ret = tryparse(Int64, txt)
    if ret !== nothing
        return ret * Float64(sf)
    end
    ret = Base.parse(Dec64, txt)
    ret *= sf
    return Float64(ret)
end

function cg_expr!(state::CodegenState, cs::SNode{SP.JuliaEscape})
    (r, _) = Meta.parse(String(cs.body), 1; raise=false, greedy=true)
    return r
end

function cg_expr!(state::CodegenState, cs::Union{SNode{SC.BinaryExpression}, SNode{SP.BinaryExpression}})
    op = Symbol(cs.op)
    (lhs, rhs) = (cg_expr!(state, cs.lhs), cg_expr!(state, cs.rhs))
    if op == :(||)
        return Expr(:call, (|), lhs, rhs)
    elseif op == :(&&)
        return Expr(:call, (&), lhs, rhs)
    elseif op == Symbol("**")
        return Expr(:call, (^), lhs, rhs)
    elseif op == Symbol("^")
        return Expr(:call, (⊻), lhs, rhs)
    elseif op == Symbol("~^") || op == Symbol("^~")
        return Expr(:call, (~), Expr(:call, (⊻), lhs, rhs))
    else
        return Expr(:call, op, lhs, rhs)
    end
end

function cg_expr!(state::CodegenState, cs::Union{SNode{SC.UnaryOp}, SNode{SP.UnaryOp}})
    op = Symbol(cs.op)
    return Expr(:call, op, cg_expr!(state, cs.operand))
end

function cg_expr!(state::CodegenState, id::Symbol)
    if id == Symbol("true")
        true
    elseif id == Symbol("false")
        false
    elseif id == Symbol("\$time")
        Expr(:call, Symbol("\$time"))
    elseif id == Symbol("temper")
        Expr(:call, Symbol("temper"))
    else
        # TODO: Request parameter generation here.
        id
    end
end

function cg_expr!(state::CodegenState, stmt::Union{SNode{SC.FunctionCall}, SNode{SP.FunctionCall}})
    fname = LSymbol(stmt.id)
    id = lowercase(String(stmt.id))
    if id == "v"
        ckt_nodes = [cg_net_name!(state, n) for n in stmt.args]
        if length(ckt_nodes) == 1
            :($(ckt_nodes[1]).V)
        elseif length(stmt.args) == 2
            :($(ckt_nodes[1]).V - $(ckt_nodes[2]).V)
        end
    elseif isdefined(CedarSim.SpectreEnvironment, Symbol(id))
        args = map(x->cg_expr!(state, x.item), stmt.args)
        Expr(:call, GlobalRef(SpectreEnvironment, Symbol(id)), args...)
    else
        args = map(x->cg_expr!(state, x.item), stmt.args)
        Expr(:call, fname, args...)
    end
end

function cg_expr!(state::CodegenState, cs::Union{SNode{SC.Identifier}, SNode{SP.Identifier}})
    # TODO probably need to disambiguate stuff here
    id = LSymbol(cs)
    return cg_expr!(state, id)
end

function cg_expr!(state::CodegenState, cs::Union{SNode{SC.TernaryExpr}, SNode{SP.TernaryExpr}})
    return Expr(:if,
        cg_expr!(state, cs.condition),
        cg_expr!(state, cs.ifcase),
        cg_expr!(state, cs.elsecase))
end

cg_expr!(state::CodegenState, n::Union{SNode{SP.Brace}, SNode{SC.Parens}, SNode{SP.Parens}, SNode{SP.Prime}}) = cg_expr!(state, n.inner)

function cg_params!(state::CodegenState, params)
    ret = Expr[]
    for param in params
        push!(ret, Expr(:kw, LSymbol(param.name), cg_expr!(state, param.val)))
    end
    ret
end

"""
    spice_instance(to_julia, ports, name, model, parameters, val=nothing)

Create a spice instance with the given parameters.
This creates a `Named` object using `spicecall`,
and is used for all spice instances except subcircuits.
"""
function cg_spice_instance!(state::CodegenState, ports, name, model, param_exprs)
    port_exprs = map(ports) do port
        cg_net_name!(state, port)
    end
    :($(Named)($(spicecall)($model; $(param_exprs...)), $(LString(name)))($(port_exprs...)))
end

function cg_instance!(state::CodegenState, instance::SNode{SP.SubcktCall})
    ssema = resolve_subckt(state.sema, LSymbol(instance.model))
    implicit_params = Expr[Expr(:kw, name, cg_expr!(state, name)) for name in ssema.exposed_parameters]
    passed_parameters = OrderedDict{Symbol, SNode}()
    callee_codegen = CodegenState(ssema)
    s = gensym()
    ca = :(let $s=(;$(implicit_params...)); end)
    params = Symbol[]
    for passed_param in instance.params
        name = LSymbol(passed_param.name)
        # TODO: Pull up dependencies to this level
        def = cg_expr!(callee_codegen, passed_param.val)
        push!(ca.args[end].args, :($name = $def))
        push!(params, name)
    end
    push!(ca.args[end].args, Expr(:call, merge, s, Expr(:tuple, Expr(:parameters, params...))))
    params = ca
    models = Expr(:tuple, Expr(:parameters, [Expr(:kw, name, cg_expr!(state, name)) for name in ssema.exposed_models]...))
    subckts = Expr(:tuple, [resolve_subckt(state.sema, name).CktID for name in ssema.exposed_subckts]...)
    port_exprs = map(instance.nodes) do port
        cg_net_name!(state, port)
    end
    ret = :(SpCircuit{$(ssema.CktID), $subckts}($params, $models)($(port_exprs...)))
    ret
end

# Phase 0: VAModelCall doesn't exist in the parser

function cg_instance!(state::CodegenState, instance::SNode{SP.Resistor})
    # if params contains r or l, val is the model or nothing
    if hasparam(instance.params, "l") || hasparam(instance.params, "r")
        model = GlobalRef(SpectreEnvironment, :resistor)
        if instance.val !== nothing
            model = cg_model_name!(state, isntance.val)
        end
        return cg_spice_instance!(state, sema_nets(instance), instance.name, model, cg_params!(state, instance.params))
    else
        params = cg_params!(state, instance.params)
        push!(params, Expr(:kw, :r, cg_expr!(state, instance.val)))
        return cg_spice_instance!(state, sema_nets(instance), instance.name, GlobalRef(SpectreEnvironment, :resistor), params)
    end
end

function cg_instance!(state::CodegenState, instance::SNode{SP.Capacitor})
    return cg_spice_instance!(state, sema_nets(instance), instance.name, GlobalRef(SpectreEnvironment, :capacitor), cg_params!(state, instance.params))
end

function cg_instance!(state::CodegenState, instance::SNode{SP.MOSFET})
    return cg_spice_instance!(state, sema_nets(instance), instance.name, cg_model_name!(state, instance.model), cg_params!(state, instance.params))
end

function cg_instance!(state::CodegenState, instance::SNode{<:Union{SP.Voltage, SP.Current}})
    constructor = if instance isa SNode{SP.Voltage}
        GlobalRef(SpectreEnvironment, :vsource)
    elseif instance isa SNode{SP.Current}
        GlobalRef(SpectreEnvironment, :isource)
    else
        error(@show instance)
    end
    # TODO figure out the correct type for the current simulation
    kws = Expr[]
    for val in instance.vals
        if val isa SNode{SP.DCSource}
            jval = cg_expr!(state, val.dcval)
            kw = Expr(:kw, :dc, jval)
            push!(kws, kw)
        elseif val isa SNode{SP.ACSource} # ACSource
            jval = cg_expr!(state, val.acmag)
            kw = Expr(:kw, :ac, jval)
            push!(kws, kw)
        elseif val isa SNode{SP.TranSource} # TranSource
            fname = LSymbol(val.kw)
            fn = getproperty(SpectreEnvironment, fname)
            if fname == :pwl
                # TODO: This isn't the correct module to use
                trancall = Expr(:call, fn, Expr(:macrocall, StaticArrays.var"@SVector", LineNumberNode(val),
                    Expr(:vect, (cg_expr!(state, v) for v in val.values)...)))
            elseif fname == :sin
                trancall = Expr(:call, SpectreEnvironment.spsin, (cg_expr!(state, v) for v in val.values)...)
            else
                trancall = Expr(:call, fn, (cg_expr!(state, v) for v in val.values)...)
            end
            kw = Expr(:kw, :tran, trancall)
            push!(kws, kw)
        else
            @show val
            error("unhandled voltage value $(String(val))")
        end
    end
    return cg_spice_instance!(state, [instance.pos, instance.neg], instance.name, constructor, kws)
end

# devtype_param is defined in spectre.jl

function cg_model_def!(state::CodegenState, (model, modelref)::Pair{<:SNode, GlobalRef}, bins::Dict{Symbol, Vector{Symbol}})
    params = Any[]
    typ = LSymbol(model.typ)
    mosfet_type = typ in (:nmos, :pmos) ? typ : nothing
    level = nothing
    version = nothing
    for p in model.parameters
        name = LSymbol(p.name)
        if name == :type
            val = LSymbol(p.val)
            if val in (:p, :n)
                name = :devtype
                mosfet_type = val == :p ? :pmos : :nmos
                continue
            end
            # Default handling - no rewrite
        elseif name == :level
            # TODO
            level = parse(Float64, String(p.val))
            continue
        elseif name == :version
            version = parse(Float64, String(p.val))
            continue
        end
        val = cg_expr!(state, p.val)
        push!(params, Expr(:kw, name, Expr(:call, CedarSim.mknondefault, val)))
    end

    # some devices have a version parameter
    # while others have distinct models
    if version !== nothing && modelref.name in (:bsim4,)
        push!(params, Expr(:kw, :version, Expr(:call, CedarSim.mknondefault, version)))
    end

    if mosfet_type !== nothing
        param = devtype_param(modelref, mosfet_type)
        param !== nothing && push!(params, Expr(:kw, param[1], Expr(:call, CedarSim.mknondefault, param[2])))
    end

    m = match(binning_rx, LString(model.name))
    if m !== nothing
        push!(get!(bins, Symbol(m.captures[1]), Vector{Symbol}()),
            LSymbol(model.name))
    end

    lhs = Symbol(LString(model.name))

    @assert isdefined(modelref.mod, modelref.name)
    T = getglobal(modelref.mod, modelref.name)

    # Peek at the fieldnames of the model to find the correct case for all params.
    # This obviates the need for using `spicecall`, saving us a bunch of compilation work.
    # Essentially, we're inlining the generator for case_adjust_kwargs
    case_insensitive = Dict(Symbol(lowercase(String(kw))) => kw for kw in fieldnames(T))
    for i = 1:length(params)
        name = params[i].args[1]
        lname = Symbol(lowercase(String(name)))
        rname = get(case_insensitive, lname, nothing)
        if rname === nothing
            if lname in (:lmin, :lmax, :wmin, :wmax)
                # These are magic parameters for model binning. Treat them as
                # uppercase always.
                rname = Symbol(uppercase(String(name)))
            else
                rname = name
            end
        end
        params[i] = Expr(:kw,
            rname,
            params[i].args[2])
    end

    return :($(ParsedModel{T})(($modelref)(; $(params...))))
end

function codegen!(state::CodegenState)
    block = Expr(:block)
    ret = block
    # Codegen simulator options
    if haskey(state.sema.options, :scale) || haskey(state.sema.options, :gmin) || haskey(state.sema.options, :temp)
        block = Expr(:block)
        scale_expr = :(old_options.scale)
        if haskey(state.sema.options, :scale)
            scale_expr = :($(CedarSim.isdefault)(old_options.scale) ? $(cg_expr!(state, state.sema.options[:scale][end][2].val)) : $scale_expr)
        end
        gmin_expr  = :(old_options.gmin)
        if haskey(state.sema.options, :gmin)
            gmin_expr = :($(CedarSim.isdefault)(old_options.gmin) ? $(cg_expr!(state, state.sema.options[:gmin][end][2].val)) : $gmin_expr)
        end
        temp_expr  = :(old_options.temp)
        if haskey(state.sema.options, :temp)
            temp_expr = :($(CedarSim.isdefault)(old_options.temp) ? $(cg_expr!(state, state.sema.options[:temp][end][2].val)) : $temp_expr)
        end
        push!(ret.args, quote
            old_options = $(CedarSim.options)[]
            new_options = $(CedarSim.SimOptions)(; temp = $temp_expr, gmin = $gmin_expr, scale = $scale_expr)
            @Base.ScopedValues.with $(CedarSim.options)=>new_options $block
        end)
    end
    # Codegen nets
    for (net, _) in state.sema.nets
        net_name = cg_net_name!(state, net)
        push!(block.args, :($net_name = net($(QuoteNode(net)))))
    end
    # Implicit and explicit parameters
    if !isempty(state.sema.formal_parameters) || !isempty(state.sema.exposed_parameters)
        push!(block.args, :(var"*params#" = getfield($(Core.Argument(1)), :params)))
    end
    if !isempty(state.sema.exposed_models)
        push!(block.args, :(var"*models#" = getfield($(Core.Argument(1)), :models)))
    end
    for param in state.sema.exposed_parameters
        push!(block.args, :($param = getfield(var"*params#", $(QuoteNode(param)))))
    end
    # Codegen parameter defs
    params_in_order = collect(state.sema.params)
    cond_syms = Vector{Symbol}(undef, length(state.sema.conditionals))
    for n in state.sema.parameter_order
        if n <= length(params_in_order)
            (name, defs) = params_in_order[n]
            # Lexically later definitions override earlier ones, but only if they
            # are active.
            for def in defs
                cd = def[2]
                def = cg_expr!(state, cd.val.val)
                if name in state.sema.formal_parameters
                    expr = :($name = hasfield(typeof(var"*params#"), $(QuoteNode(name))) ? getfield(var"*params#", $(QuoteNode(name))) : $def)
                else
                    expr = :($name = $def)
                end
                if cd.cond != 0
                    cond = cond_syms[abs(cd.cond)]
                    cd.cond < 0 && (cond = :(!$cond))
                    expr = :($cond && $expr)
                end
                push!(block.args, expr)
            end
        else
            cond_idx = n - length(params_in_order)
            _, cd = state.sema.conditionals[cond_idx]
            s = cond_syms[cond_idx] = gensym()
            expr = cg_expr!(state, cd.val.body)
            if cd.cond != 0
                if cd.cond > 0
                    expr = :($(cond_syms[cd.cond]) && $expr)
                else
                    expr = :($(cond_syms[-cd.cond]) || $expr)
                end
            end
            push!(block.args, :($s = $expr))
        end
    end
    # Implicit and explicit models
    bins = Dict{Symbol, Vector{Symbol}}()
    for model in state.sema.exposed_models
        name = cg_model_name!(state, model)
        push!(block.args, :($name = getfield(var"*models#", $(QuoteNode(model)))))
    end
    for (model, defs) in state.sema.models
        name = cg_model_name!(state, model)
        model_def = cg_model_def!(state, defs[end][2].val, bins)
        push!(block.args, :($name = $model_def))
    end
    # Binned model aggregation
    for (name, this_bins) in bins
        name = cg_model_name!(state, name)
        push!(block.args, :($name = $(CedarSim.BinnedModel)($(GlobalRef(SpectreEnvironment, :var"$scale"))(), ($(this_bins...),))))
    end
    for (name, instances) in state.sema.instances
        if length(instances) == 1 && only(instances)[2].cond == 0
            (_, instance) = only(instances)
            instance = instance.val
            push!(block.args, LineNumberNode(instance))
            push!(block.args, cg_instance!(state, instance))
        else
            counter = gensym()
            push!(block.args, :($counter = 0))
            for (_, instance) in instances
                cond = cond_syms[abs(instance.cond)]
                instance.cond < 0 && (cond = :(!$cond))
                push!(block.args, Expr(:if, cond, Expr(:block,
                    LineNumberNode(instance.val),
                    cg_instance!(state, instance.val),
                    :($counter += 1)
                )))
            end
            push!(block.args, :($counter > 1 && error($("Multiple simultaneously active instances of $name"))))
        end
    end
    return ret
end

function codegen(scope::SemaResult)
    codegen!(CodegenState(scope))
end

#==============================================================================#
# MNA Codegen: Generate stamp! calls instead of Named(spicecall(...))
#
# Phase 4: SPICE Codegen for MNA Backend
#
# These functions generate MNA builder code from the same SemaResult.
# The generated function signature is:
#   function circuit_name(params, spec::MNASpec) -> MNAContext
#==============================================================================#

# Import MNA types for codegen
using ..MNA: MNAContext, MNASpec, get_node!, stamp!
using ..MNA: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
using ..MNA: VCVS, VCCS, CCVS, CCCS

# Helper to get a param value
function getparam(params, name, default=nothing)
    for p in params
        if LString(p.name) == name
            return p.val
        end
    end
    return default
end

"""
    cg_mna_instance!(state, instance)

Generate MNA stamp! call for a device instance.
Returns an expression that stamps the device into the context.
"""
function cg_mna_instance! end

"""
Generate stamp! call for a resistor.
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.Resistor})
    nets = sema_nets(instance)
    p = cg_net_name!(state, nets[1])
    n = cg_net_name!(state, nets[2])
    name = LString(instance.name)

    # Get resistance value
    r_expr = if hasparam(instance.params, "r")
        cg_expr!(state, getparam(instance.params, "r"))
    elseif instance.val !== nothing
        cg_expr!(state, instance.val)
    else
        # Calculate from rsh, l, w if present
        if hasparam(instance.params, "l") && hasparam(instance.params, "rsh")
            l_expr = cg_expr!(state, getparam(instance.params, "l"))
            w_expr = hasparam(instance.params, "w") ? cg_expr!(state, getparam(instance.params, "w")) : 1e-6
            rsh_expr = cg_expr!(state, getparam(instance.params, "rsh"))
            :($rsh_expr * $l_expr / $w_expr)
        else
            1000.0  # Default
        end
    end

    # Handle multiplicity
    m_expr = hasparam(instance.params, "m") ? cg_expr!(state, getparam(instance.params, "m")) : 1

    return quote
        let r_val = $r_expr, m_val = $m_expr
            # Parallel resistors: R_eff = R / m
            stamp!(Resistor(r_val / m_val; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
        end
    end
end

"""
Generate stamp! call for a capacitor.
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.Capacitor})
    nets = sema_nets(instance)
    p = cg_net_name!(state, nets[1])
    n = cg_net_name!(state, nets[2])
    name = LString(instance.name)

    # Get capacitance value
    c_expr = if hasparam(instance.params, "c")
        cg_expr!(state, getparam(instance.params, "c"))
    elseif instance.val !== nothing
        cg_expr!(state, instance.val)
    else
        1e-12  # Default 1pF
    end

    # Handle multiplicity
    m_expr = hasparam(instance.params, "m") ? cg_expr!(state, getparam(instance.params, "m")) : 1

    return quote
        let c_val = $c_expr, m_val = $m_expr
            # Parallel capacitors: C_eff = C * m
            stamp!(Capacitor(c_val * m_val; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
        end
    end
end

"""
Generate stamp! call for an inductor.
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.Inductor})
    nets = sema_nets(instance)
    p = cg_net_name!(state, nets[1])
    n = cg_net_name!(state, nets[2])
    name = LString(instance.name)

    # Get inductance value
    l_expr = if hasparam(instance.params, "l")
        cg_expr!(state, getparam(instance.params, "l"))
    elseif instance.val !== nothing
        cg_expr!(state, instance.val)
    else
        1e-6  # Default 1uH
    end

    return quote
        let l_val = $l_expr
            stamp!(Inductor(l_val; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
        end
    end
end

"""
Generate stamp! call for a voltage source.
"""
function cg_mna_instance!(state::CodegenState, ::Val{:mna}, instance::SNode{SP.Voltage})
    p = cg_net_name!(state, instance.pos)
    n = cg_net_name!(state, instance.neg)
    name = LString(instance.name)

    # Parse source values
    dc_val = 0.0
    for val in instance.vals
        if val isa SNode{SP.DCSource}
            dc_val = cg_expr!(state, val.dcval)
        end
    end

    return quote
        let v = $dc_val
            stamp!(VoltageSource(v; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
        end
    end
end

"""
Generate stamp! call for a current source.

SPICE convention: I1 n+ n- val means current flows from n+ to n- through the source.
This injects current into n- and extracts from n+.
MNA convention: stamp!(CurrentSource, ctx, p, n) injects into p, extracts from n.
So we swap: p=neg, n=pos.
"""
function cg_mna_instance!(state::CodegenState, ::Val{:mna}, instance::SNode{SP.Current})
    pos = cg_net_name!(state, instance.pos)
    neg = cg_net_name!(state, instance.neg)
    name = LString(instance.name)

    # Parse source values
    dc_val = 0.0
    for val in instance.vals
        if val isa SNode{SP.DCSource}
            dc_val = cg_expr!(state, val.dcval)
        end
    end

    # Swap nodes: MNA injects into first arg, SPICE injects into neg
    return quote
        let i = $dc_val
            stamp!(CurrentSource(i; name=$(QuoteNode(Symbol(name)))), ctx, $neg, $pos)
        end
    end
end

"""
Generate stamp! call for a VCVS (E element) - ControlledSource{:V,:V}.
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.ControlledSource{:V,:V}})
    nets = sema_nets(instance)
    out_p = cg_net_name!(state, nets[1])
    out_n = cg_net_name!(state, nets[2])
    in_p = cg_net_name!(state, nets[3])
    in_n = cg_net_name!(state, nets[4])
    name = LString(instance.name)

    # Get the voltage control node which contains the gain
    voltage_ctrl = instance.val
    @assert isa(voltage_ctrl, SNode{SP.VoltageControl})

    # Get the gain value from VoltageControl
    gain_expr = if voltage_ctrl.val !== nothing
        cg_expr!(state, voltage_ctrl.val)
    elseif voltage_ctrl.params !== nothing && hasparam(voltage_ctrl.params, "gain")
        cg_expr!(state, getparam(voltage_ctrl.params, "gain"))
    else
        1.0
    end

    return quote
        let gain = $gain_expr
            stamp!(VCVS(gain; name=$(QuoteNode(Symbol(name)))), ctx, $out_p, $out_n, $in_p, $in_n)
        end
    end
end

"""
Generate stamp! call for a VCCS (G element) - ControlledSource{:V,:C}.
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.ControlledSource{:V,:C}})
    nets = sema_nets(instance)
    out_p = cg_net_name!(state, nets[1])
    out_n = cg_net_name!(state, nets[2])
    in_p = cg_net_name!(state, nets[3])
    in_n = cg_net_name!(state, nets[4])
    name = LString(instance.name)

    # Get the voltage control node which contains the gm
    voltage_ctrl = instance.val
    @assert isa(voltage_ctrl, SNode{SP.VoltageControl})

    # Get the transconductance value from VoltageControl
    gm_expr = if voltage_ctrl.val !== nothing
        cg_expr!(state, voltage_ctrl.val)
    elseif voltage_ctrl.params !== nothing && (hasparam(voltage_ctrl.params, "gm") || hasparam(voltage_ctrl.params, "gain"))
        cg_expr!(state, getparam(voltage_ctrl.params, hasparam(voltage_ctrl.params, "gm") ? "gm" : "gain"))
    else
        1.0
    end

    return quote
        let gm = $gm_expr
            stamp!(VCCS(gm; name=$(QuoteNode(Symbol(name)))), ctx, $out_p, $out_n, $in_p, $in_n)
        end
    end
end

"""
Generate stamp! call for a CCVS (H element) - ControlledSource{:C,:V}.
Current-controlled voltage source: V(out) = rm * I(vname)
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.ControlledSource{:C,:V}})
    nets = sema_nets(instance)
    out_p = cg_net_name!(state, nets[1])
    out_n = cg_net_name!(state, nets[2])
    name = LString(instance.name)

    # Get the controlling current info from val
    current_ctrl = instance.val
    @assert isa(current_ctrl, SNode{SP.CurrentControl})

    # Get the voltage source name that provides the controlling current
    vname_sym = if current_ctrl.vnam !== nothing
        Symbol(:I_, LSymbol(current_ctrl.vnam))  # Current variable is named I_<vname>
    else
        error("CCVS $name requires a voltage source name for current sensing")
    end

    # Get the transresistance value
    rm_expr = if current_ctrl.val !== nothing
        cg_expr!(state, current_ctrl.val)
    elseif hasparam(current_ctrl.params, "rm") || hasparam(current_ctrl.params, "gain")
        cg_expr!(state, getparam(current_ctrl.params, hasparam(current_ctrl.params, "rm") ? "rm" : "gain"))
    else
        1.0
    end

    return quote
        let rm = $rm_expr
            # Get the current index of the referenced voltage source
            I_in_idx = get_current_idx(ctx, $(QuoteNode(vname_sym)))
            stamp!(CCVS(rm; name=$(QuoteNode(Symbol(name)))), ctx, $out_p, $out_n, I_in_idx)
        end
    end
end

"""
Generate stamp! call for a CCCS (F element) - ControlledSource{:C,:C}.
Current-controlled current source: I(out) = gain * I(vname)
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.ControlledSource{:C,:C}})
    nets = sema_nets(instance)
    out_p = cg_net_name!(state, nets[1])
    out_n = cg_net_name!(state, nets[2])
    name = LString(instance.name)

    # Get the controlling current info from val
    current_ctrl = instance.val
    @assert isa(current_ctrl, SNode{SP.CurrentControl})

    # Get the voltage source name that provides the controlling current
    vname_sym = if current_ctrl.vnam !== nothing
        Symbol(:I_, LSymbol(current_ctrl.vnam))  # Current variable is named I_<vname>
    else
        error("CCCS $name requires a voltage source name for current sensing")
    end

    # Get the current gain value
    gain_expr = if current_ctrl.val !== nothing
        cg_expr!(state, current_ctrl.val)
    elseif hasparam(current_ctrl.params, "gain")
        cg_expr!(state, getparam(current_ctrl.params, "gain"))
    else
        1.0
    end

    return quote
        let gain = $gain_expr
            # Get the current index of the referenced voltage source
            I_in_idx = get_current_idx(ctx, $(QuoteNode(vname_sym)))
            stamp!(CCCS(gain; name=$(QuoteNode(Symbol(name)))), ctx, $out_p, $out_n, I_in_idx)
        end
    end
end

"""
Generate MNA subcircuit call.

Explicit params from the netlist (e.g., `X1 vcc 0 myres factor=2`) become
kwargs to the builder call. The builder then passes these to the lens,
which merges them with any sweep overrides.
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.SubcktCall}, subckt_builders::Dict{Symbol, Symbol})
    ssema = resolve_subckt(state.sema, LSymbol(instance.model))

    callee_codegen = CodegenState(ssema)

    # Build kwargs from explicit parameters passed to the subcircuit call
    explicit_kwargs = Expr[]

    # Find Parameter children in the AST (SubcktCall exposes params as children)
    for child in SpectreNetlistParser.RedTree.children(instance)
        if child !== nothing && isa(child, SNode{SP.Parameter})
            name = LSymbol(child.name)
            def = cg_expr!(callee_codegen, child.val)
            push!(explicit_kwargs, Expr(:kw, name, def))
        end
    end

    # Port expressions
    port_exprs = [cg_net_name!(state, port) for port in instance.nodes]

    subckt_name = LSymbol(instance.model)
    instance_name = LSymbol(instance.name)
    builder_name = get(subckt_builders, subckt_name, Symbol(subckt_name, "_mna_builder"))

    # Generate code that:
    # 1. Navigates to subcircuit's portion of the lens via getproperty
    # 2. Calls builder with navigated lens and explicit params as kwargs
    # Note: lens_var is captured from the enclosing scope (var"*lens#" or lens)
    return quote
        let subckt_lens = getproperty(var"*lens#", $(QuoteNode(instance_name)))
            $builder_name(subckt_lens, spec, ctx, $(port_exprs...); $(explicit_kwargs...))
        end
    end
end

# Version for use in subcircuit context (lens is named `lens`)
function cg_mna_instance_subcircuit!(state::CodegenState, instance::SNode{SP.SubcktCall}, subckt_builders::Dict{Symbol, Symbol})
    ssema = resolve_subckt(state.sema, LSymbol(instance.model))

    callee_codegen = CodegenState(ssema)

    # Build kwargs from explicit parameters passed to the subcircuit call
    explicit_kwargs = Expr[]

    for child in SpectreNetlistParser.RedTree.children(instance)
        if child !== nothing && isa(child, SNode{SP.Parameter})
            name = LSymbol(child.name)
            def = cg_expr!(callee_codegen, child.val)
            push!(explicit_kwargs, Expr(:kw, name, def))
        end
    end

    port_exprs = [cg_net_name!(state, port) for port in instance.nodes]

    subckt_name = LSymbol(instance.model)
    instance_name = LSymbol(instance.name)
    builder_name = get(subckt_builders, subckt_name, Symbol(subckt_name, "_mna_builder"))

    return quote
        let subckt_lens = getproperty(lens, $(QuoteNode(instance_name)))
            $builder_name(subckt_lens, spec, ctx, $(port_exprs...); $(explicit_kwargs...))
        end
    end
end

#==============================================================================#
# Spectre Instance Codegen
#==============================================================================#

"""
Generate MNA stamp! call for a Spectre instance.

Spectre syntax: name (node1 node2 ...) master param1=val1 param2=val2 ...

Supported masters:
- resistor: r=value
- capacitor: c=value
- inductor: l=value
- vsource: dc=value, type=pwl/pulse/sine
- isource: dc=value, type=pwl/pulse/sine
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SC.Instance})
    master = lowercase(String(instance.master))
    nets = sema_nets(instance)
    name = LString(instance.name)

    if master == "resistor"
        length(nets) >= 2 || error("resistor requires 2 nodes")
        p = cg_net_name!(state, nets[1])
        n = cg_net_name!(state, nets[2])

        r_expr = if hasparam(instance.params, "r")
            cg_expr!(state, getparam(instance.params, "r"))
        else
            1000.0  # Default 1k
        end

        m_expr = hasparam(instance.params, "m") ? cg_expr!(state, getparam(instance.params, "m")) : 1

        return quote
            let r_val = $r_expr, m_val = $m_expr
                stamp!(Resistor(r_val / m_val; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        end

    elseif master == "capacitor"
        length(nets) >= 2 || error("capacitor requires 2 nodes")
        p = cg_net_name!(state, nets[1])
        n = cg_net_name!(state, nets[2])

        c_expr = if hasparam(instance.params, "c")
            cg_expr!(state, getparam(instance.params, "c"))
        else
            1e-12  # Default 1pF
        end

        m_expr = hasparam(instance.params, "m") ? cg_expr!(state, getparam(instance.params, "m")) : 1

        return quote
            let c_val = $c_expr, m_val = $m_expr
                stamp!(Capacitor(c_val * m_val; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        end

    elseif master == "inductor"
        length(nets) >= 2 || error("inductor requires 2 nodes")
        p = cg_net_name!(state, nets[1])
        n = cg_net_name!(state, nets[2])

        l_expr = if hasparam(instance.params, "l")
            cg_expr!(state, getparam(instance.params, "l"))
        else
            1e-6  # Default 1uH
        end

        m_expr = hasparam(instance.params, "m") ? cg_expr!(state, getparam(instance.params, "m")) : 1

        return quote
            let l_val = $l_expr, m_val = $m_expr
                stamp!(Inductor(l_val / m_val; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        end

    elseif master == "vsource"
        length(nets) >= 2 || error("vsource requires 2 nodes")
        p = cg_net_name!(state, nets[1])
        n = cg_net_name!(state, nets[2])

        dc_val = if hasparam(instance.params, "dc")
            cg_expr!(state, getparam(instance.params, "dc"))
        else
            0.0
        end

        # Check for transient source types
        src_type = hasparam(instance.params, "type") ? lowercase(String(getparam(instance.params, "type"))) : nothing

        if src_type == "pwl" && hasparam(instance.params, "wave")
            # PWL source with wave parameter
            wave_expr = cg_expr!(state, getparam(instance.params, "wave"))
            return quote
                stamp!(PWLVoltageSource($wave_expr; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        elseif src_type == "sine" || src_type == "sin"
            # Sinusoidal source
            vo = hasparam(instance.params, "vo") ? cg_expr!(state, getparam(instance.params, "vo")) : 0.0
            va = hasparam(instance.params, "va") ? cg_expr!(state, getparam(instance.params, "va")) : 1.0
            freq = hasparam(instance.params, "freq") ? cg_expr!(state, getparam(instance.params, "freq")) : 1e3
            return quote
                stamp!(SinVoltageSource($vo, $va, $freq; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        else
            # DC source
            return quote
                stamp!(VoltageSource($dc_val; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        end

    elseif master == "isource"
        length(nets) >= 2 || error("isource requires 2 nodes")
        # Spectre convention: isource (p n) means current flows from p to n internally
        # MNA convention: CurrentSource stamps into p and out of n
        p = cg_net_name!(state, nets[1])
        n = cg_net_name!(state, nets[2])

        dc_val = if hasparam(instance.params, "dc")
            cg_expr!(state, getparam(instance.params, "dc"))
        else
            0.0
        end

        # Check for transient source types
        src_type = hasparam(instance.params, "type") ? lowercase(String(getparam(instance.params, "type"))) : nothing

        if src_type == "pwl" && hasparam(instance.params, "wave")
            wave_expr = cg_expr!(state, getparam(instance.params, "wave"))
            return quote
                stamp!(PWLCurrentSource($wave_expr; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        else
            # DC source
            return quote
                stamp!(CurrentSource($dc_val; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        end

    elseif master == "vcvs"
        # Voltage-controlled voltage source
        length(nets) >= 4 || error("vcvs requires 4 nodes")
        p = cg_net_name!(state, nets[1])
        n = cg_net_name!(state, nets[2])
        cp = cg_net_name!(state, nets[3])
        cn = cg_net_name!(state, nets[4])

        gain = hasparam(instance.params, "gain") ? cg_expr!(state, getparam(instance.params, "gain")) : 1.0

        return quote
            stamp!(VCVS($gain; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n, $cp, $cn)
        end

    elseif master == "vccs"
        # Voltage-controlled current source
        length(nets) >= 4 || error("vccs requires 4 nodes")
        p = cg_net_name!(state, nets[1])
        n = cg_net_name!(state, nets[2])
        cp = cg_net_name!(state, nets[3])
        cn = cg_net_name!(state, nets[4])

        gm = hasparam(instance.params, "gm") ? cg_expr!(state, getparam(instance.params, "gm")) : 1.0

        return quote
            stamp!(VCCS($gm; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n, $cp, $cn)
        end

    else
        # Unknown master type - skip with warning comment
        return :(nothing)  # TODO: handle $(master)
    end
end

# Fallback for unhandled instance types - generate nothing
function cg_mna_instance!(state::CodegenState, instance)
    return :(nothing)  # Skip unimplemented devices
end

# Version without subckt_builders dict - just use default naming
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.SubcktCall})
    return cg_mna_instance!(state, instance, Dict{Symbol, Symbol}())
end

# Disambiguate Voltage/Current source calls
function cg_mna_instance!(state::CodegenState, instance::SNode{<:Union{SP.Voltage, SP.Current}})
    pos = cg_net_name!(state, instance.pos)
    neg = cg_net_name!(state, instance.neg)
    name = LString(instance.name)
    is_voltage = instance isa SNode{SP.Voltage}

    # SPICE convention: Iname n+ n- val means current flows from n+ to n- through source
    # This injects current into n- and extracts from n+
    # MNA convention: stamp!(CurrentSource, ctx, p, n) injects into p, extracts from n
    # So for current sources, swap the nodes
    p, n = is_voltage ? (pos, neg) : (neg, pos)

    # Parse source values - check for DC, AC, and transient
    dc_val = nothing
    tran_source = nothing

    for val in instance.vals
        if val isa SNode{SP.DCSource}
            dc_val = cg_expr!(state, val.dcval)
        elseif val isa SNode{SP.TranSource}
            tran_source = val
        end
        # AC sources not handled in MNA transient (yet)
    end

    # If we have a transient source, generate appropriate device
    if tran_source !== nothing
        fname = LSymbol(tran_source.kw)

        if fname == :pwl
            # PWL source: values are interleaved time-value pairs
            # PWL(t1 v1 t2 v2 ...)
            vals = [cg_expr!(state, v) for v in tran_source.values]

            if is_voltage
                return quote
                    let vals = [$(vals...)]
                        n_points = div(length(vals), 2)
                        times = vals[1:2:end]
                        values = vals[2:2:end]
                        stamp!(PWLVoltageSource(times, values; name=$(QuoteNode(Symbol(name)))),
                               ctx, $p, $n; t=spec.time, mode=spec.mode)
                    end
                end
            else
                return quote
                    let vals = [$(vals...)]
                        n_points = div(length(vals), 2)
                        times = vals[1:2:end]
                        values = vals[2:2:end]
                        stamp!(PWLCurrentSource(times, values; name=$(QuoteNode(Symbol(name)))),
                               ctx, $p, $n; t=spec.time, mode=spec.mode)
                    end
                end
            end

        elseif fname == :sin
            # SIN source: SIN(vo va freq [td theta phase])
            # SPICE order: vo, va, freq, td, theta, phase
            vals = [cg_expr!(state, v) for v in tran_source.values]
            n_vals = length(vals)

            vo_expr = n_vals >= 1 ? vals[1] : 0.0
            va_expr = n_vals >= 2 ? vals[2] : 0.0
            freq_expr = n_vals >= 3 ? vals[3] : 1.0
            td_expr = n_vals >= 4 ? vals[4] : 0.0
            theta_expr = n_vals >= 5 ? vals[5] : 0.0
            phase_expr = n_vals >= 6 ? vals[6] : 0.0

            if is_voltage
                return quote
                    let vo = $vo_expr, va = $va_expr, freq = $freq_expr,
                        td = $td_expr, theta = $theta_expr, phase = $phase_expr
                        stamp!(SinVoltageSource(vo, va, freq; td=td, theta=theta, phase=phase,
                                                name=$(QuoteNode(Symbol(name)))),
                               ctx, $p, $n; t=spec.time, mode=spec.mode)
                    end
                end
            else
                return quote
                    let io = $vo_expr, ia = $va_expr, freq = $freq_expr,
                        td = $td_expr, theta = $theta_expr, phase = $phase_expr
                        stamp!(SinCurrentSource(io, ia, freq; td=td, theta=theta, phase=phase,
                                                name=$(QuoteNode(Symbol(name)))),
                               ctx, $p, $n; t=spec.time, mode=spec.mode)
                    end
                end
            end

        elseif fname == :pulse
            # PULSE source: pulse(v1 v2 td tr tf pw period)
            # For now, implement as PWL approximation
            vals = [cg_expr!(state, v) for v in tran_source.values]
            n_vals = length(vals)

            v1_expr = n_vals >= 1 ? vals[1] : 0.0
            v2_expr = n_vals >= 2 ? vals[2] : 1.0
            td_expr = n_vals >= 3 ? vals[3] : 0.0
            tr_expr = n_vals >= 4 ? vals[4] : 1e-9
            tf_expr = n_vals >= 5 ? vals[5] : 1e-9
            pw_expr = n_vals >= 6 ? vals[6] : 1e-3
            per_expr = n_vals >= 7 ? vals[7] : 2e-3

            if is_voltage
                # Generate pulse as PWL for one period
                return quote
                    let v1 = $v1_expr, v2 = $v2_expr, td = $td_expr,
                        tr = $tr_expr, tf = $tf_expr, pw = $pw_expr, per = $per_expr
                        # PWL approximation of pulse for first period
                        times = [0.0, td, td+tr, td+tr+pw, td+tr+pw+tf, per]
                        values = [v1, v1, v2, v2, v1, v1]
                        stamp!(PWLVoltageSource(times, values; name=$(QuoteNode(Symbol(name)))),
                               ctx, $p, $n; t=spec.time, mode=spec.mode)
                    end
                end
            else
                return quote
                    let i1 = $v1_expr, i2 = $v2_expr, td = $td_expr,
                        tr = $tr_expr, tf = $tf_expr, pw = $pw_expr, per = $per_expr
                        times = [0.0, td, td+tr, td+tr+pw, td+tr+pw+tf, per]
                        values = [i1, i1, i2, i2, i1, i1]
                        stamp!(PWLCurrentSource(times, values; name=$(QuoteNode(Symbol(name)))),
                               ctx, $p, $n; t=spec.time, mode=spec.mode)
                    end
                end
            end

        else
            # Unknown transient source type - fall back to DC
            @warn "Unknown transient source type: $fname, using DC value"
            dc_val_actual = dc_val !== nothing ? dc_val : 0.0
            if is_voltage
                return quote
                    stamp!(VoltageSource($dc_val_actual; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
                end
            else
                return quote
                    stamp!(CurrentSource($dc_val_actual; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
                end
            end
        end
    end

    # No transient source - use DC value
    dc_val_actual = dc_val !== nothing ? dc_val : 0.0
    if is_voltage
        return quote
            let v = $dc_val_actual
                stamp!(VoltageSource(v; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        end
    else
        return quote
            let i = $dc_val_actual
                stamp!(CurrentSource(i; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
            end
        end
    end
end

"""
    codegen_mna!(state::CodegenState; skip_nets=Symbol[])

Generate MNA builder function body from semantic analysis result.
Returns code that builds an MNAContext with all devices stamped.

`skip_nets` specifies nets that should NOT have get_node! generated
(used for subcircuit ports which are passed in as arguments).

When `is_subcircuit=true`, params are function kwargs and lens is named `lens`.
When `is_subcircuit=false` (top-level), params come from the `params` argument.
"""
function codegen_mna!(state::CodegenState; skip_nets::Vector{Symbol}=Symbol[], is_subcircuit::Bool=false)
    block = Expr(:block)
    ret = block

    # Handle temperature option - update spec if temp is set
    if haskey(state.sema.options, :temp)
        temp_expr = cg_expr!(state, state.sema.options[:temp][end][2].val)
        push!(block.args, :(spec = MNASpec(temp=$temp_expr, mode=spec.mode)))
    end

    # Codegen nets - get_node! for each net (except ports passed as arguments)
    for (net, _) in state.sema.nets
        net_name = cg_net_name!(state, net)
        # Skip nets that are subcircuit ports - they're passed as function args
        if net_name in skip_nets
            continue
        end
        push!(block.args, :($net_name = get_node!(ctx, $(QuoteNode(net_name)))))
    end

    # Parameters from lens - set up lens for parameter access
    # For subcircuits, lens is a function argument named `lens`
    # For top-level, we wrap the `params` argument
    has_subcircuit_calls = any(state.sema.instances) do (name, insts)
        any(inst -> inst[2].val isa SNode{SP.SubcktCall}, insts)
    end
    needs_lens = !isempty(state.sema.formal_parameters) || !isempty(state.sema.exposed_parameters) ||
                 !isempty(state.sema.params) || has_subcircuit_calls

    # The lens variable name differs between subcircuit and top-level
    lens_var = is_subcircuit ? :lens : :var"*lens#"

    if needs_lens && !is_subcircuit
        # Top-level: wrap params argument in lens
        push!(block.args, :(var"*lens#" = params isa $(AbstractParamLens) ? params : $(ParamLens)(params)))
    end

    # NOTE: Don't pre-initialize exposed_parameters to 0.0!
    # The parameter_order loop below handles formal_parameters with proper defaults.

    # Codegen parameter defs
    params_in_order = collect(state.sema.params)
    cond_syms = Vector{Symbol}(undef, length(state.sema.conditionals))

    # If parameter_order is populated, use it; otherwise fall back to direct iteration
    param_indices = if isempty(state.sema.parameter_order)
        # Fall back: iterate through params in definition order
        1:length(params_in_order)
    else
        state.sema.parameter_order
    end

    for n in param_indices
        if n <= length(params_in_order)
            (name, defs) = params_in_order[n]
            for def in defs
                cd = def[2]
                def_expr = cg_expr!(state, cd.val.val)
                # In SPICE, any .param can be overridden from outside the subcircuit call
                # Use lens callable interface for ParamObserver support and parameter overrides
                if needs_lens
                    if is_subcircuit
                        # Subcircuit: params are function kwargs, use variable name in lens call
                        # lens(; name=name) where `name` is the function kwarg
                        expr = :($name = $lens_var(; $(Expr(:kw, name, name))).$name)
                    else
                        # Top-level: use default expression as kwarg to lens call
                        expr = :($name = $lens_var(; $(Expr(:kw, name, def_expr))).$name)
                    end
                else
                    expr = :($name = $def_expr)
                end
                if cd.cond != 0
                    cond = cond_syms[abs(cd.cond)]
                    cd.cond < 0 && (cond = :(!$cond))
                    expr = :($cond && $expr)
                end
                push!(block.args, expr)
            end
        else
            cond_idx = n - length(params_in_order)
            _, cd = state.sema.conditionals[cond_idx]
            s = cond_syms[cond_idx] = gensym()
            expr = cg_expr!(state, cd.val.body)
            if cd.cond != 0
                if cd.cond > 0
                    expr = :($(cond_syms[cd.cond]) && $expr)
                else
                    expr = :($(cond_syms[-cd.cond]) || $expr)
                end
            end
            push!(block.args, :($s = $expr))
        end
    end

    # Codegen device instances using MNA stamps
    # For SubcktCall, we need to use the appropriate function based on context
    function codegen_instance(inst)
        if inst isa SNode{SP.SubcktCall} && is_subcircuit
            cg_mna_instance_subcircuit!(state, inst, Dict{Symbol, Symbol}())
        else
            cg_mna_instance!(state, inst)
        end
    end

    for (name, instances) in state.sema.instances
        if length(instances) == 1 && only(instances)[2].cond == 0
            (_, instance) = only(instances)
            instance = instance.val
            push!(block.args, codegen_instance(instance))
        else
            # Handle conditional instances
            for (_, instance) in instances
                if instance.cond != 0
                    cond = cond_syms[abs(instance.cond)]
                    instance.cond < 0 && (cond = :(!$cond))
                    push!(block.args, Expr(:if, cond, codegen_instance(instance.val)))
                else
                    push!(block.args, codegen_instance(instance.val))
                end
            end
        end
    end

    return ret
end

"""
    codegen_mna(scope::SemaResult)

Generate MNA builder code from semantic analysis result.
"""
function codegen_mna(scope::SemaResult)
    codegen_mna!(CodegenState(scope))
end

"""
    extract_subcircuit_ports(sema::SemaResult) -> Vector{Symbol}

Extract port names from a subcircuit's AST.
Ports are the HierarchialNode children of the Subckt AST node.
"""
function extract_subcircuit_ports(sema::SemaResult)
    ports = Symbol[]
    subckt_ast = sema.ast
    for child in SpectreNetlistParser.RedTree.children(subckt_ast)
        if child !== nothing && isa(child, SNode{SP.HierarchialNode})
            port_name = LSymbol(child)
            push!(ports, port_name)
        end
    end
    return ports
end

"""
    codegen_mna_subcircuit(sema::SemaResult, subckt_name::Symbol)

Generate an MNA subcircuit builder function from semantic analysis.

The generated function has signature:
    function subckt_name_mna_builder(lens, spec, ctx, port1, port2, ...; param1=default1, ...) -> nothing

Explicit params from the subcircuit call are passed as kwargs. The builder
calls `lens(; param1=param1, ...)` to merge with any sweep overrides.
"""
function codegen_mna_subcircuit(sema::SemaResult, subckt_name::Symbol)
    state = CodegenState(sema)

    # Extract ports from AST
    subckt_ports = extract_subcircuit_ports(sema)

    # Build port arguments - subcircuit ports become function parameters
    port_args = [Symbol("port_", i) for i in 1:length(subckt_ports)]

    # Map internal port names to function parameters
    port_mappings = Expr[:($internal_name = $arg)
        for (internal_name, arg) in zip(subckt_ports, port_args)]

    # Build kwargs for subcircuit params with their defaults
    # These allow the caller to override via kwargs, then lens merges with sweep params
    param_kwargs = Expr[]
    for (name, defs) in sema.params
        if !isempty(defs)
            # Use the first definition's default value
            def = first(defs)[2]
            if def.cond == 0  # non-conditional
                def_expr = cg_expr!(state, def.val.val)
                push!(param_kwargs, Expr(:kw, name, def_expr))
            end
        end
    end

    # Generate body, skipping get_node! for ports (they're passed as args)
    # is_subcircuit=true means params are function kwargs and lens is named `lens`
    body = codegen_mna!(state; skip_nets=subckt_ports, is_subcircuit=true)

    builder_name = Symbol(subckt_name, "_mna_builder")

    return quote
        function $(builder_name)(lens, spec::$(MNASpec), ctx::$(MNAContext), $(port_args...); $(param_kwargs...))
            # Map ports to internal names
            $(port_mappings...)
            $body
            return nothing
        end
    end
end

"""
    make_mna_circuit(ast; circuit_name=:circuit)

Generate an MNA builder function from a SPICE/Spectre AST.

The generated function has signature:
    function circuit_name(params, spec::MNASpec) -> MNAContext

# Example
```julia
ast = SpectreNetlistParser.SPICENetlistParser.SPICENetlistCSTParser.parse(spice_code)
code = make_mna_circuit(ast)
circuit_fn = eval(code)
ctx = circuit_fn((R1=1000.0,), MNASpec())
sys = MNA.assemble!(ctx)
sol = MNA.solve_dc(sys)
```
"""
function make_mna_circuit(ast; circuit_name::Symbol=:circuit)
    # Run semantic analysis (use sema() not sema_file_or_section to get parameter_order)
    sema_result = sema(ast)
    state = CodegenState(sema_result)

    # Generate subcircuit builders first
    subckt_defs = Expr[]
    for (name, subckt_list) in sema_result.subckts
        if !isempty(subckt_list)
            # Take the first (non-conditional) subcircuit definition
            _, subckt_sema = first(subckt_list)
            subckt_def = codegen_mna_subcircuit(subckt_sema.val, name)
            push!(subckt_defs, subckt_def)
        end
    end

    # Generate the body
    body = codegen_mna!(state)

    # Wrap in function definition
    return quote
        # Subcircuit builders
        $(subckt_defs...)

        # Main circuit builder
        function $(circuit_name)(params, spec::$(MNASpec)=$(MNASpec)())
            ctx = $(MNAContext)()
            $body
            return ctx
        end
    end
end
