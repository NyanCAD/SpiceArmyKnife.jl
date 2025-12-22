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
const spice_regex = Regex("($(join(keys(spice_magnitudes), "|")))\$")

const binning_rx = r"(.*)\.([0-9]+)"

# Phase 0: NumberLiteral is now the leaf node (FloatLiteral/IntLiteral don't exist)
function cg_expr!(state::CodegenState, cs::SNode{SP.NumberLiteral})
    txt = lowercase(String(cs))
    sf = d"1"
    m = match(spice_regex, txt)
    if m !== nothing && haskey(spice_magnitudes, m.match)
        sf = spice_magnitudes[m.match]
        txt = txt[begin:end-length(m.match)]
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
"""
function cg_mna_instance!(state::CodegenState, ::Val{:mna}, instance::SNode{SP.Current})
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
        let i = $dc_val
            stamp!(CurrentSource(i; name=$(QuoteNode(Symbol(name)))), ctx, $p, $n)
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

    gain_expr = if hasparam(instance.params, "gain")
        cg_expr!(state, getparam(instance.params, "gain"))
    elseif instance.val !== nothing
        cg_expr!(state, instance.val)
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

    gm_expr = if hasparam(instance.params, "gm") || hasparam(instance.params, "gain")
        cg_expr!(state, getparam(instance.params, hasparam(instance.params, "gm") ? "gm" : "gain"))
    elseif instance.val !== nothing
        cg_expr!(state, instance.val)
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
Generate MNA subcircuit call.
"""
function cg_mna_instance!(state::CodegenState, instance::SNode{SP.SubcktCall}, subckt_builders::Dict{Symbol, Symbol})
    ssema = resolve_subckt(state.sema, LSymbol(instance.model))

    # Build parameter expressions
    implicit_params = Expr[Expr(:kw, name, cg_expr!(state, name)) for name in ssema.exposed_parameters]

    callee_codegen = CodegenState(ssema)
    s = gensym()
    ca = :(let $s=(;$(implicit_params...)); end)
    params = Symbol[]
    for passed_param in instance.params
        name = LSymbol(passed_param.name)
        def = cg_expr!(callee_codegen, passed_param.val)
        push!(ca.args[end].args, :($name = $def))
        push!(params, name)
    end
    push!(ca.args[end].args, Expr(:call, merge, s, Expr(:tuple, Expr(:parameters, params...))))
    params_expr = ca

    # Port expressions
    port_exprs = [cg_net_name!(state, port) for port in instance.nodes]

    subckt_name = LSymbol(instance.model)
    builder_name = get(subckt_builders, subckt_name, Symbol(subckt_name, "_mna_builder"))

    return quote
        let subckt_params = $params_expr
            # Call subcircuit builder with inherited context
            $builder_name(subckt_params, spec, ctx, $(port_exprs...))
        end
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
    p = cg_net_name!(state, instance.pos)
    n = cg_net_name!(state, instance.neg)
    name = LString(instance.name)
    is_voltage = instance isa SNode{SP.Voltage}

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
    codegen_mna!(state::CodegenState)

Generate MNA builder function body from semantic analysis result.
Returns code that builds an MNAContext with all devices stamped.
"""
function codegen_mna!(state::CodegenState)
    block = Expr(:block)
    ret = block

    # Handle temperature option - update spec if temp is set
    if haskey(state.sema.options, :temp)
        temp_expr = cg_expr!(state, state.sema.options[:temp][end][2].val)
        push!(block.args, :(spec = MNASpec(temp=$temp_expr, mode=spec.mode)))
    end

    # Codegen nets - get_node! for each net
    for (net, _) in state.sema.nets
        net_name = cg_net_name!(state, net)
        push!(block.args, :($net_name = get_node!(ctx, $(QuoteNode(net_name)))))
    end

    # Parameters from lens/params
    if !isempty(state.sema.formal_parameters) || !isempty(state.sema.exposed_parameters)
        push!(block.args, :(var"*params#" = params isa ParamLens ? getfield(params, :nt) : params))
        push!(block.args, :(var"*params#" = hasfield(typeof(var"*params#"), :params) ? getfield(var"*params#", :params) : var"*params#"))
    end

    for param in state.sema.exposed_parameters
        push!(block.args, :($param = hasfield(typeof(var"*params#"), $(QuoteNode(param))) ? getfield(var"*params#", $(QuoteNode(param))) : 0.0))
    end

    # Codegen parameter defs
    params_in_order = collect(state.sema.params)
    cond_syms = Vector{Symbol}(undef, length(state.sema.conditionals))
    for n in state.sema.parameter_order
        if n <= length(params_in_order)
            (name, defs) = params_in_order[n]
            for def in defs
                cd = def[2]
                def_expr = cg_expr!(state, cd.val.val)
                if name in state.sema.formal_parameters
                    expr = :($name = hasfield(typeof(var"*params#"), $(QuoteNode(name))) ? getfield(var"*params#", $(QuoteNode(name))) : $def_expr)
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
    for (name, instances) in state.sema.instances
        if length(instances) == 1 && only(instances)[2].cond == 0
            (_, instance) = only(instances)
            instance = instance.val
            push!(block.args, cg_mna_instance!(state, instance))
        else
            # Handle conditional instances
            for (_, instance) in instances
                if instance.cond != 0
                    cond = cond_syms[abs(instance.cond)]
                    instance.cond < 0 && (cond = :(!$cond))
                    push!(block.args, Expr(:if, cond, cg_mna_instance!(state, instance.val)))
                else
                    push!(block.args, cg_mna_instance!(state, instance.val))
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
    codegen_mna_subcircuit(sema::SemaResult, subckt_name::Symbol)

Generate an MNA subcircuit builder function from semantic analysis.

The generated function has signature:
    function subckt_name_mna_builder(params, spec, ctx, port1, port2, ...) -> nothing

It stamps devices directly into the passed context.
"""
function codegen_mna_subcircuit(sema::SemaResult, subckt_name::Symbol)
    state = CodegenState(sema)
    body = codegen_mna!(state)

    # Build port arguments - subcircuit ports become function parameters
    port_args = [Symbol("port_", i) for i in 1:length(sema.ports)]

    # Map internal port names to function parameters
    port_mappings = Expr[:($internal_name = $arg)
        for (internal_name, arg) in zip(sema.ports, port_args)]

    builder_name = Symbol(subckt_name, "_mna_builder")

    return quote
        function $(builder_name)(params, spec::$(MNASpec), ctx::$(MNAContext), $(port_args...))
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
    # Run semantic analysis
    sema = sema_file(ast)
    state = CodegenState(sema)

    # Generate subcircuit builders first
    subckt_defs = Expr[]
    for (name, subckt_list) in sema.subckts
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
