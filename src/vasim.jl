using VerilogAParser
using AbstractTrees
using AbstractTrees: parent, nodevalue
using VerilogAParser.VerilogACSTParser:
    ContributionStatement, AnalogSeqBlock, AnalogBlock,
    InOutDeclaration, NetDeclaration, ParameterDeclaration, AliasParameterDeclaration,
    VerilogModule, Literal, BinaryExpression, BPFC,
    IdentifierPrimary, @case, BranchDeclaration,
    AnalogFunctionDeclaration,
    IntRealDeclaration, IntRealVarDecl, AnalogStatement,
    AnalogConditionalBlock, AnalogVariableAssignment, AnalogProceduralAssignment,
    Parens, AnalogIf, AnalogFor, AnalogWhile, AnalogRepeat, UnaryOp, Function,
    AnalogSystemTaskEnable, StringLiteral,
    CaseStatement, FunctionCall, TernaryExpr,
    FloatLiteral, ChunkTree, virtrange,
    filerange, LineNumbers, compute_line,
    SystemIdentifier, Node, Identifier, IdentifierConcatItem,
    IdentifierPart, Attributes
using VerilogAParser.VerilogATokenize:
    Kind, INPUT, OUTPUT, INOUT, REAL, INTEGER, is_scale_factor
using Combinatorics
using ForwardDiff
using ForwardDiff: Dual

# Phase 0: Use stubs instead of DAECompiler
@static if CedarSim.USE_DAECOMPILER
    using DAECompiler
    using DAECompiler: variable, equation!, observed!
else
    using ..DAECompilerStubs: ddt, variable, equation!, observed!
end

const VAT = VerilogAParser.VerilogATokenize

const VANode = VerilogAParser.VerilogACSTParser.Node
struct SimTag; end
ForwardDiff.:(≺)(::Type{<:ForwardDiff.Tag}, ::Type{SimTag}) = true
ForwardDiff.:(≺)(::Type{SimTag}, ::Type{<:ForwardDiff.Tag}) = false

# Phase 0: Guard DAECompiler.ddt extension
@static if CedarSim.USE_DAECOMPILER
    function DAECompiler.ddt(dual::ForwardDiff.Dual{SimTag})
        ForwardDiff.Dual{SimTag}(ddt(dual.value), map(ddt, dual.partials.values))
    end
end

function eisa(e::VANode{S}, T::Type) where {S}
    S <: T
end
formof(e::VANode{S}) where {S} = S

@enum BranchKind CURRENT VOLTAGE

struct VAFunction
    arg_order::Vector{Symbol}
    inout_decls::Dict{Symbol, Symbol}
end

struct Scope
    parameters::Set{Symbol}
    node_order::Vector{Symbol}
    ninternal_nodes::Int
    branch_order::Vector{Pair{Symbol}}
    used_branches::Set{Pair{Symbol}}
    var_types::Dict{Symbol, Union{Type{Int}, Type{Float64}}}
    all_functions::Dict{Symbol, VAFunction}
    undefault_ids::Bool
    ddx_order::Vector{Symbol}
end
Scope() = Scope(Set{Symbol}(), Vector{Symbol}(), 0, Vector{Pair{Symbol}}(), Set{Pair{Symbol}}(),
    Dict{Symbol, Union{Type{Int}, Type{Float64}}}(), Dict{Symbol, VAFunction}(),
    false, Vector{Symbol}())

struct Pins
    p::Any
    n::Any
end
Base.reverse(p::Pins) = Pins(p.n, p.p)
Base.getindex(p::Pins, i::Integer) = i == 1 ? p.p : i == 2 ? p.n : throw(BoundsError(p, i))

struct BranchContribution
    kind::BranchKind
    pins::Pins
    value::Any
end

function Base.LineNumberNode(n::VANode)
    print_vr = (n.startof+n.expr.off):(n.startof+n.expr.off+n.expr.width-1)
    tc = AbstractTrees.TreeCursor(ChunkTree(n.ps))
    leaf = VerilogAParser.VerilogACSTParser.findleafbyvirtrange(Leaves(tc), first(print_vr))

    # Compute line number
    vr = virtrange(leaf)
    fr = filerange(leaf)

    startoff_file = first(fr) + (n.startof + n.expr.off - first(vr))

    sf = n.ps.srcfiles[leaf.fidx]
    lsf = sf.lineinfo

    lno_first = compute_line(lsf, startoff_file)
    LineNumberNode(lno_first, Symbol(sf.path))
end


function (::Scope)(cs::VANode{Literal})
    Meta.parse(String(cs))
end

const sf_mapping = Dict(
    'T' => 1e12,
    'G' => 1e9,
    'M' => 1e6,
    'K' => 1e3,
    'k' => 1e3,
    'm' => 1e-3,
    'u' => 1e-6,
    'n' => 1e-9,
    'p' => 1e-12,
    'f' => 1e-15,
    'a' => 1e-18,
);

function (::Scope)(cs::VANode{FloatLiteral})
    txt = String(cs)
    sf = nothing
    if is_scale_factor(txt[end])
        sf = txt[end]
        txt = txt[1:end-1]
    end
    ret = Base.parse(Float64, txt)
    if sf !== nothing
        ret *= sf_mapping[sf]
    end
    return ret
end

function (to_julia::Scope)(cs::VANode{ContributionStatement})
    bpfc = cs.lvalue
    kind = Symbol(bpfc.id)
    # TODO: Look at disciplines
    if kind == :I
        kind = CURRENT
    elseif kind == :V
        kind = VOLTAGE
    else
        return :(error("Unknown branch contribution kind"))
    end

    refs = map(bpfc.references) do ref
        Symbol(assemble_id_string(ref.item))
    end

    if length(refs) == 1
        node = refs[1]
        svar = Symbol("branch_state_", node, "_0")
        eqvar = Symbol("branch_value_", node, "_0")
        push!(to_julia.used_branches, node => Symbol("0"))
        return quote
            if $svar != $kind
                $eqvar = 0.0
                $svar = $kind
            end
            $eqvar += $(ForwardDiff.value)($(SimTag),$(to_julia(cs.assign_expr)))
        end
    elseif length(refs) == 2
        (id1, id2) = refs

        idx = findfirst(to_julia.branch_order) do branch
            branch == (id1 => id2) || branch == (id2 => id1)
        end
        @assert idx !== nothing
        branch = to_julia.branch_order[idx]
        push!(to_julia.used_branches, branch)
        reversed = branch == (id2 => id1)

        s = gensym()

        svar = Symbol("branch_state_", branch[1], "_", branch[2])
        eqvar = Symbol("branch_value_", branch[1], "_", branch[2])
        return @nolines quote
            if $svar != $kind
                $eqvar = 0.0
                $svar = $kind
            end
            $s = $(to_julia(cs.assign_expr))
            $eqvar += $(ForwardDiff.value)($(SimTag), $(reversed ? :(-$s) : s))
        end
    end
end

function (to_julia::Scope)(bpfc::VANode{BPFC})
    Expr(:call, Symbol(bpfc.id), map(x->Symbol(x.item), bpfc.references)...)
end

function assemble_id_string(id)
    if isa(id, Node{SystemIdentifier})
        return String(id)
    elseif isa(id, Node{Identifier})
        return join(assemble_id_string(c) for c in children(id))
    elseif isa(id, Node{IdentifierPart})
        s = String(id)
        id.escaped && (s = 2[2:end])
        return s
    elseif isa(id, Node{IdentifierConcatItem})
        return assemble_id_string(id.id)
    else
        error(typeof(id))
    end
end

function (scope::Scope)(ip::VANode{IdentifierPrimary})
    id = Symbol(assemble_id_string(ip.id))
    if scope.undefault_ids
        id = Expr(:call, undefault, id)
    end
    id
end

function (scope::Scope)(ip::VANode{SystemIdentifier})
    id = Symbol(ip)
    # Is this really the right place for these? These are kind function calls
    # without arguments in the spec.
    if id == Symbol("\$temperature")
        return Expr(:call, id)
    else
        error()
    end
end

function (to_julia::Scope)(cs::VANode{BinaryExpression})
    op = Symbol(cs.op)
    #if op in (:(||), :(&&))
    #    return Expr(op, to_julia(cs.lhs), to_julia(cs.rhs))
    if op == :(||)
        return Expr(:call, (|), to_julia(cs.lhs), to_julia(cs.rhs))
    elseif op == :(&&)
        return Expr(:call, (&), to_julia(cs.lhs), to_julia(cs.rhs))
    else
        return Expr(:call, op, to_julia(cs.lhs), to_julia(cs.rhs))
    end
end

function (to_julia::Scope)(asb::VANode{AnalogSeqBlock})
    if asb.decl !== nothing
        block_var_types = copy(to_julia.var_types)

        for decl in asb.decl.decls
            item = decl.item
            @case formof(item) begin
                IntRealDeclaration => begin
                    T = kw_to_T(item.kw.kw)
                    for ident in item.idents
                        # ident.item is IntRealVarDecl with id, eq, init fields
                        vardecl = ident.item
                        name = Symbol(assemble_id_string(vardecl.id))
                        block_var_types[name] = T
                    end
                end
            end
        end

        to_julia_block = Scope(to_julia.parameters,
            to_julia.node_order, to_julia.ninternal_nodes, to_julia.branch_order,
            to_julia.used_branches, block_var_types, to_julia.all_functions,
            to_julia.undefault_ids, to_julia.ddx_order)
    else
        to_julia_block = to_julia
    end
    ret = Expr(:block)
    last_lno = nothing
    for stmt in asb.stmts
        lno = LineNumberNode(stmt)
        if lno != last_lno
            push!(ret.args, lno)
        end
        push!(ret.args, to_julia_block(stmt))
    end
    ret
end

abstract type SemaError <: CedarException; end

struct DuplicateDef <: SemaError
    name::Symbol
end

struct NoArguments <: SemaError; end
struct MissingTypeDecl <: SemaError
    name::Symbol
end


function validate_parameters(type_decls, inout_decls)
    isempty(inout_decls) && throw(NoArguments())
    for key in keys(inout_decls)
        haskey(type_decls, key) || throw(MissingTypeDecl(key))
    end
end

kw_to_T(kw::Kind) = kw === REAL ? Float64 : Int

(to_julia::Scope)(stmt::VANode{AnalogStatement}) = to_julia(stmt.stmt)

function (to_julia::Scope)(stmt::VANode{AnalogConditionalBlock})
    aif = stmt.aif
    function if_body_to_julia(ifstmt)
        if formof(ifstmt) == AnalogSeqBlock
            return to_julia(ifstmt)
        else
            return Expr(:block, LineNumberNode(ifstmt), to_julia(ifstmt))
        end
    end

    ifex = ex = Expr(:if, to_julia(aif.condition), if_body_to_julia(aif.stmt))
    for case in stmt.elsecases
        if formof(case.stmt) == AnalogIf
            elif = case.stmt
            newex = Expr(:elseif, to_julia(elif.condition), if_body_to_julia(elif.stmt))
            push!(ex.args, newex)
            ex = newex
        else
            push!(ex.args, if_body_to_julia(case.stmt))
        end
    end
    ifex
end

function (to_julia::Scope)(stmt::VANode{AnalogFor})
    body = to_julia(stmt.stmt)
    push!(body.args, to_julia(stmt.update_stmt))
    while_expr = Expr(:while, to_julia(stmt.cond_expr), body)
    Expr(:block, to_julia(stmt.init_stmt), while_expr)
end

function (to_julia::Scope)(stmt::VANode{AnalogWhile})
    body = to_julia(stmt.stmt)
    Expr(:while, to_julia(stmt.cond_expr), body)
end

function (to_julia::Scope)(stmt::VANode{AnalogRepeat})
    body = to_julia(stmt.stmt)
    Expr(:for, :(_ = 1:$(stmt.num_repeat)), body)
end

function (to_julia::Scope)(stmt::VANode{UnaryOp})
    return Expr(:call, Symbol(stmt.op), to_julia(stmt.operand))
end

function (to_julia::Scope)(stmt::VANode{FunctionCall})
    fname = Symbol(stmt.id)
    if fname == Symbol("\$param_given")
        id = Symbol(stmt.args[1].item)
        return Expr(:call, !, Expr(:call, isdefault,
            Expr(:call, getfield, Symbol("#self#"), QuoteNode(id))))
    end

    # TODO: Rather than hardcoding this, this should look at the list of
    # defined disciplines
    if fname == :V
        function Vref(id)
            id_idx = findfirst(==(id), to_julia.ddx_order)
            if id_idx !== nothing
                return Expr(:call, Dual{SimTag}, id,
                        Expr(:(...), ntuple(i->i == id_idx ? 1.0 : 0.0, length(to_julia.ddx_order)))
                )
            else
                return id
            end
        end

        @assert length(stmt.args) in (1,2)
        id1 = Symbol(stmt.args[1].item)
        id2 = length(stmt.args) > 1 ? Symbol(stmt.args[2].item) : nothing

        if id2 === nothing
            push!(to_julia.used_branches, id1 => Symbol("0"))
            return Vref(id1)
        else
            idx = findfirst(to_julia.branch_order) do branch
                branch == (id1 => id2) || branch == (id2 => id1)
            end
            @assert idx !== nothing
            branch = to_julia.branch_order[idx]
            push!(to_julia.used_branches, branch)
            return :($(Vref(id1)) - $(Vref(id2)))
        end
    elseif fname == :I
        @assert length(stmt.args) == 2
        id1 = Symbol(stmt.args[1].item)
        id2 = Symbol(stmt.args[2].item)

        idx = findfirst(to_julia.branch_order) do branch
            branch == (id1 => id2) || branch == (id2 => id1)
        end
        @assert idx !== nothing

        branch = to_julia.branch_order[idx]
        reversed = branch == (id2 => id1)
        push!(to_julia.used_branches, branch)

        ex = :(error("TODO"))
        reversed && (ex = :(-$ex))
        return ex
    elseif fname == :ddx
        item = stmt.args[2].item
        @assert formof(item) == FunctionCall
        @assert Symbol(item.id) == :V
        if length(item.args) == 1
            probe = Symbol(item.args[1].item)
            id_idx = findfirst(==(probe), to_julia.ddx_order)
            return :(let x = $(to_julia(stmt.args[1].item))
                $(isa)(x, $(Dual)) ? (@inbounds $(ForwardDiff.partials)($(SimTag), x, $id_idx)) : 0.0
            end)
        else
            probe1 = Symbol(item.args[1].item)
            id1_idx = findfirst(==(probe1), to_julia.ddx_order)
            probe2 = Symbol(item.args[2].item)
            id2_idx = findfirst(==(probe2), to_julia.ddx_order)
            return :(let x = $(to_julia(stmt.args[1].item)),
                        dx1 = $(isa)(x, $(Dual)) ? (@inbounds $(ForwardDiff.partials)($(SimTag), x, $id1_idx)) : 0.0,
                        dx2 = $(isa)(x, $(Dual)) ? (@inbounds $(ForwardDiff.partials)($(SimTag), x, $id2_idx)) : 0.0
                (dx1-dx2)/2
            end)
        end
    elseif fname ∈ (:white_noise, :flicker_noise)
        args = map(x->to_julia(x.item), stmt.args)
        ϵ = Symbol(:ϵ, gensym(last(args)))
        args[end] = QuoteNode(ϵ)
        return Expr(:call, fname, :dscope, args...)
    end

    vaf = get(to_julia.all_functions, fname, nothing)
    if vaf !== nothing
        # Call to a Verilog-A defined function
        args = map(x->to_julia(x.item), stmt.args)
        in_args = Any[]
        out_args = Any[]

        if length(args) != length(vaf.arg_order)
            # TODO: Fancy diagnostic
            return Expr(:call, error, "Wrong number of arguments to function $fname ($args, $(vaf.arg_order)")
        end

        for (arg, vaarg) in zip(args, vaf.arg_order)
            kind = vaf.inout_decls[vaarg]
            if kind == :output || kind == :inout
                isa(arg, Symbol) || return Expr(:call, error, "Output argument $vaarg to function $fname must be a symbol. Got $arg.")
                push!(out_args, arg)
            end
            if kind == :input || kind == :inout
                push!(in_args, arg)
            end
        end
        ret = Expr(:call, fname, in_args...)
        if length(out_args) != 0
            s = gensym()
            ret = @nolines quote
                ($s, ($(out_args...),)) = $ret
                $s
            end
        end
        return ret
    end

    return Expr(:call, fname, map(x->to_julia(x.item), stmt.args)...)
end

function (to_julia::Scope)(stmt::VANode{AnalogProceduralAssignment})
    return to_julia(stmt.assign)
end
function (to_julia::Scope)(stmt::VANode{AnalogVariableAssignment})
    assignee = Symbol(stmt.lvalue)
    varT = get(to_julia.var_types, assignee, nothing)
    assignee === nothing && cedarerror("Undeclared variable: $assignee")

    eq = stmt.eq.op

    op = eq == VAT.EQ             ?  :(=) :
         eq == VAT.STAR_STAR_EQ   ?  :(=) : # Extra handling below
         eq == VAT.PLUS_EQ        ? :(+=) :
         eq == VAT.MINUS_EQ       ? :(-=) :
         eq == VAT.STAR_EQ        ? :(*=) :
         eq == VAT.SLASH_EQ       ? :(/=) :
         eq == VAT.PERCENT_EQ     ? :(%=) :
         eq == VAT.AND_EQ         ? :(&=) :
         eq == VAT.OR_EQ          ? :(|=) :
         eq == VAT.XOR_EQ         ? :(^=) :
         eq == VAT.LBITSHIFT_EQ   ? :(<<=) :
         eq == VAT.RBITSHIFT_EQ   ? :(>>=) :
         eq == VAT.LBITSHIFT_A_EQ ? :(=) : # Extra handling below
         eq == VAT.RBITSHIFT_A_EQ ? :(>>>=) :
         cedarerror("Unsupported assignment operator: $eq")

    req = Expr(op, assignee, varT === nothing ? Expr(:call, error, "Unknown type for variable $assignee") :
        Expr(:call, VerilogAEnvironment.vaconvert, varT, to_julia(stmt.rvalue)))

    if eq == VAT.STAR_STAR_EQ
        req = Expr(op, assignee, Expr(:call, var"**", req.args...))
    elseif eq === VAT.LBITSHIFT_A_EQ
        req = Expr(op, assignee, Expr(:call, var"<<<", req.args...))
    end

    return req
end

function (to_julia::Scope)(stmt::VANode{Parens})
    return to_julia(stmt.inner)
end

function (to_julia::Scope)(cs::VANode{TernaryExpr})
    return Expr(:if, to_julia(cs.condition), to_julia(cs.ifcase), to_julia(cs.elsecase))
end


function (to_julia::Scope)(fd::VANode{AnalogFunctionDeclaration})
    type_decls = Dict{Symbol, Any}()
    inout_decls = Dict{Symbol, Symbol}()
    fname = Symbol(fd.id)
    var_types = Dict{Symbol, Union{Type{Int}, Type{Float64}}}()
    # 4.7.1
    # The analog_function_type specifies the return value of the function;
    # its use is optional. type can be a real or an integer; if unspecified,
    # the default is real.
    rt = fd.fty === nothing ? Real : kw_to_T(fd.fty.kw)
    var_types[fname] = rt
    arg_order = Symbol[]
    for decl in fd.items
        item = decl.item
        @case formof(item) begin
            InOutDeclaration => begin
                kind = item.kw.kw === INPUT ?  :input :
                       item.kw.kw === OUTPUT ? :output :
                                               :inout
                for name in item.portnames
                    ns = Symbol(name.item)
                    haskey(inout_decls, ns) && throw(DuplicateDef(name))
                    # NB: Don't think this is technically forbidden by the standard
                    ns == fname && throw(DuplicateDef(ns))
                    inout_decls[ns] = kind
                    push!(arg_order, ns)
                end
            end
            IntRealDeclaration => begin
                T = kw_to_T(item.kw.kw)
                for ident in item.idents
                    # ident.item is IntRealVarDecl with id, eq, init fields
                    vardecl = ident.item
                    ns = Symbol(assemble_id_string(vardecl.id))
                    haskey(type_decls, ns) && throw(DuplicateDef(ns))
                    ns == fname && throw(DuplicateDef(ns))
                    var_types[ns] = T
                end
            end
            _ => cedarerror("Unknown function declaration item")
        end
    end

    to_julia_internal = Scope(to_julia.parameters, to_julia.node_order,
        to_julia.ninternal_nodes, to_julia.branch_order, to_julia.used_branches, var_types,
        to_julia.all_functions, to_julia.undefault_ids, to_julia.ddx_order)

    validate_parameters(var_types, inout_decls)
    out_args = [k for k in arg_order if inout_decls[k] in (:output, :inout)]
    in_args = [k for k in arg_order if inout_decls[k] in (:input, :inout)]
    rt_decl = length(out_args) == 0 ? fname : :(($fname, ($(out_args...),)))

    to_julia.all_functions[fname] = VAFunction(arg_order, inout_decls)

    localize_vars = Any[]
    for var in keys(var_types)
        var in arg_order && continue
        push!(localize_vars, :(local $var))
    end

    return @nolines quote
        @inline function $fname($(in_args...))
            $(localize_vars...)
            local $fname = $(VerilogAEnvironment.vaconvert)($rt, 0)
            $(to_julia_internal(fd.stmt))
            return $rt_decl
        end
    end
end

function (to_julia::Scope)(stmt::VANode{StringLiteral})
    return String(stmt)[2:end-1]
end

const systemtaskenablemap = Dict{Symbol, Function}(
    Symbol("\$ln")=>Base.log, Symbol("\$log10")=>log10, Symbol("\$exp")=>exp, Symbol("\$sqrt")=>sqrt,
    Symbol("\$sin")=>sin, Symbol("\$cos")=>cos, Symbol("\$tan")=>tan,
    Symbol("\$asin")=>asin, Symbol("\$acos")=>acos, Symbol("\$atan")=>atan, Symbol("\$atan2")=>atan,
    Symbol("\$sinh")=>sinh, Symbol("\$cos")=>cosh, Symbol("\$tan")=>tanh,
    Symbol("\$asinh")=>asinh, Symbol("\$acosh")=>acosh, Symbol("\$atanh")=>atanh,
    Symbol("\$error")=>error)
function (to_julia::Scope)(stmt::VANode{AnalogSystemTaskEnable})
    if formof(stmt.task) == FunctionCall
        fc = stmt.task
        fname = Symbol(fc.id)
        args = map(x->to_julia(x.item), fc.args)
        if fname in keys(systemtaskenablemap)
            return Expr(:call, systemtaskenablemap[fname], args...)
        elseif fname == Symbol("\$strobe")
            # TODO: Need to guard this with convergence conditions
            return nothing # Expr(:call, println, args...)
        elseif fname == Symbol("\$warning")
            warn = GlobalRef(Base, Symbol("@warn"))
            return nothing # Expr(:macrocall, warn, LineNumberNode(stmt), args..., :(maxlog=1))
        else
            cedarerror("Verilog task unimplmented: $fname")
        end
    else
        error()
    end
end

function (to_julia::Scope)(stmt::VANode{CaseStatement})
    s = gensym()
    first = true
    expr = nothing
    default_case = nothing
    for case in stmt.cases
        if isa(case.conds, Node)
            # Default case
            default_case = to_julia(case.item)
        else
            conds = map(cond->:($s == $(to_julia(cond.item))), case.conds)
            cond = length(conds) == 1 ? conds[1] : Expr(:(||), conds...)
            ex = Expr(first ? :if : :elseif, cond, to_julia(case.item))
            if first
                expr = ex
                first = false
            else
                push!(expr.args, ex)
            end
        end
    end
    default_case !== nothing && (push!(expr.args, default_case))
    Expr(:block, :($s = $(to_julia(stmt.switch))), expr)
end

function (to_julia::Scope)(stmt::VANode{Attributes})
    res = Dict{Symbol, Any}()
    for attr in stmt.specs
        item = attr.item
        res[Symbol(item.name)] = to_julia(item.val)
    end
    return res
end

function pins(vm::VANode{VerilogModule})
    plist = vm.port_list
    plist === nothing && return []
    pins = Symbol[]
    mapreduce(vcat, plist.ports) do port_decl
        Symbol(port_decl.item)
    end
end

using Base.Meta
using Core.Compiler: SlotNumber

function find_ddx!(ddx_order::Vector{Symbol}, va::VANode)
    for stmt in AbstractTrees.PreOrderDFS(va)
        if stmt isa VANode{FunctionCall} && Symbol(stmt.id) == :ddx
            item = stmt.args[2].item
            @assert formof(item) == FunctionCall
            @assert Symbol(item.id) == :V
            for arg in item.args
                name = Symbol(arg.item)
                !in(name, ddx_order) && push!(ddx_order, Symbol(arg.item))
            end
        end
    end
end

function make_spice_device(vm::VANode{VerilogModule})
    ps = pins(vm)
    modname = String(vm.id)
    symname = Symbol(modname)
    ret = Expr(:block,
        #map(ps) do p
        #    :(@named $p = Pin(analysis))
        #end...
    )
    struct_fields = Any[]
    defaults = Any[]
    parameter_names = Set{Symbol}()
    to_julia_global = Scope()
    find_ddx!(to_julia_global.ddx_order, vm)
    to_julia_defaults = Scope(to_julia_global.parameters,
    to_julia_global.node_order, to_julia_global.ninternal_nodes,
        to_julia_global.branch_order, to_julia_global.used_branches,
        to_julia_global.var_types, to_julia_global.all_functions,
        true, to_julia_global.ddx_order)
    internal_nodes = Vector{Symbol}()
    var_types = Dict{Symbol, Union{Type{Int}, Type{Float64}}}()
    aliases = Dict{Symbol, Symbol}()
    observables = Dict{Symbol, Symbol}()

    # First to a pre-pass to figure out the scope context
    for child in vm.items
        item = child.item
        @case formof(item) begin
            # Not represented on the julia side for now
            InOutDeclaration => nothing
            NetDeclaration => begin
                for net in item.net_names
                    id = Symbol(assemble_id_string(net.item))
                    if !(id in ps)
                        # Internal node
                        push!(internal_nodes, id)
                    end
                end
            end
            ParameterDeclaration => begin
                for param in item.params
                    param = param.item
                    pT = Float64
                    if item.ptype !== nothing
                        pT = kw_to_T(item.ptype.kw)
                    end
                    paramname = String(assemble_id_string(param.id))
                    paramname_lc = lowercase(paramname)
                    paramsym = Symbol(paramname)
                    push!(parameter_names, paramsym)
                    # TODO: The CMC Verilog-A models use an attribute to
                    # distinguish between model and instance parameters
                    # Use symbol for type to avoid embedding type objects in AST
                    pT_sym = pT === Float64 ? :Float64 : :Int
                    # Build field expr without quoting to avoid hygiene issues
                    type_annotation = Expr(:(::), paramsym, Expr(:curly, :(CedarSim.DefaultOr), pT_sym))
                    default_val = Expr(:block, LineNumberNode(param.default_expr), to_julia_defaults(param.default_expr))
                    field_expr = Expr(:(=), type_annotation, default_val)
                    push!(struct_fields, field_expr)
                    var_types[Symbol(paramname)] = pT
                    #push!(ret.args,
                    #    :(@parameters $(Symbol(paramsym)))
                    #)
                    #push!(defaults,
                    #    :($(Symbol(paramsym)) => instance.model.$(paramsym))
                    #)
                    #push!(parameter_names, paramsym)
                    # TODO: SPICE likes these lowercase, so we generate both,
                    # but in Verilog-A they're case sensitive, so shouldn't
                    # interact.
                    #if paramname != paramname_lc
                    #    push!(ret.args, :($(Symbol(paramname)) = $(Symbol(paramname_lc))))
                    #end
                end
            end
            AliasParameterDeclaration => begin
                param = item
                paramsym = Symbol(assemble_id_string(param.id))
                targetsym = Symbol(assemble_id_string(param.value))
                push!(parameter_names, paramsym)
                aliases[paramsym] = targetsym
            end
            IntRealDeclaration => begin
                attr = child.attrs !== nothing && to_julia_global(child.attrs)
                observe = attr isa Dict && length(attr) == 1 && haskey(attr, :desc)
                T = kw_to_T(item.kw.kw)
                for ident in item.idents
                    # ident.item is IntRealVarDecl with id, eq, init fields
                    vardecl = ident.item
                    name = Symbol(assemble_id_string(vardecl.id))
                    var_types[name] = T
                    if observe
                        observables[name] = Symbol(attr[:desc])
                    end
                end
            end
        end
    end

    node_order = [ps; internal_nodes; Symbol("0")]
    to_julia = Scope(parameter_names,  node_order, length(internal_nodes),
        collect(map(x->Pair(x...), combinations(node_order, 2))),
        Set{Pair{Symbol}}(),
        var_types,
        Dict{Symbol, VAFunction}(), false,
        to_julia_global.ddx_order)
    lno = nothing
    for child in vm.items
        item = child.item
        @case formof(item) begin
            # Not represented on the julia side for now
            InOutDeclaration => nothing
            IntRealDeclaration => nothing
            NetDeclaration => nothing # Handled above
            BranchDeclaration => nothing
            ParameterDeclaration => nothing # Handled above
            AliasParameterDeclaration => nothing # Handled above
            AnalogFunctionDeclaration => begin
                push!(ret.args, to_julia(item))
            end
            AnalogBlock => begin
                lno = LineNumberNode(item.stmt)
                push!(ret.args, to_julia(item.stmt))
            end
            _ => cedarerror("Unrecognized statement $child")
        end
    end

    # Phase 0: Use local variable/equation!/observed! references (stubs or DAECompiler)
    internal_nodeset = map(enumerate(internal_nodes)) do (n, id)
        @nolines quote
            $id = $(variable)($DScope(dscope, $(QuoteNode(Symbol("V($id)")))))
        end
    end

    all_branch_order = filter(branch->branch in to_julia.used_branches, to_julia.branch_order)
    branch_state = map(all_branch_order) do (a, b)
        svar = Symbol("branch_state_", a, "_", b)
        eqvar = Symbol("branch_value_", a, "_", b)
        @nolines quote
            $svar = $(CURRENT)
            $eqvar = 0.0
        end
    end

    internal_currents = Any[(Symbol("I($a, $b)") for (a, b) in all_branch_order)...]

    internal_currents_def = Expr(:block,
        (@nolines quote
            $v = $(variable)($DScope(dscope,$(QuoteNode(v))))
        end for v in internal_currents)...
    )

    function current_sum(node)
        Expr(:call, +, map(Iterators.filter(branch->node in branch[2], enumerate(all_branch_order))) do (n, (a,b))
            ex = internal_currents[n]
            # Positive currents flow out of devices, into nodes, so I(a, b)'s contribution to the a KCL
            # is -I(a, b).
            node == a ? :(-$ex) : ex
        end...)
    end

    internal_node_kcls = map(enumerate(internal_nodes)) do (n, node)
        @nolines :($(equation!)($(current_sum(node)),
            $DScope(dscope, $(QuoteNode(Symbol("KCL($node)"))))))
    end

    internal_eqs = map(enumerate(all_branch_order)) do (n, (a,b))
        svar = Symbol("branch_state_", a, "_", b)
        eqvar = Symbol("branch_value_", a, "_", b)
        if b == Symbol("0")
            @nolines :($(equation!)($svar == $(CURRENT) ? $(internal_currents[n]) - $eqvar : $a - $eqvar,
                $DScope(dscope, $(QuoteNode(Symbol("Branch($a)"))))))
        else
            @nolines :($(equation!)($svar == $(CURRENT) ? $(internal_currents[n]) - $eqvar : ($a - $b) - $eqvar,
                $DScope(dscope, $(QuoteNode(Symbol("Branch($a, $b)"))))))
        end
    end

    argnames = map(p->Symbol("#port_", p), ps)
    external_eqs = map(argnames, map(current_sum, ps)) do a, c
        :($(kcl!)($(a), $c))
    end

    obs_def = (:($o = 0.0) for o in keys(observables))
    obs_expr = (:($(observed!)($var,
            $DScope(dscope, $(QuoteNode(name))))) for (var, name) in observables)

    arg_assign = map(ps, argnames) do p, a
        :($p = $a.V)
    end

    params_to_locals = map(collect(to_julia.parameters)) do id
        :($id = $(Expr(:call, undefault, Expr(:call, getfield, Symbol("#self#"), QuoteNode(id)))))
    end

    sim = @nolines :(function (var"#self#"::$symname)($(argnames...); dscope=$(GenScope)($(debug_scope)[], $(QuoteNode(symname))))
        $(obs_def...)
        $lno
        $(arg_assign...)
        $(internal_currents_def)
        $(params_to_locals...)
        $(internal_nodeset...)
        $(branch_state...)
        $ret
        $(internal_node_kcls...)
        $(external_eqs...)
        $(internal_eqs...)
        $(obs_expr...)
        return ()
    end)

    Expr(:toplevel,
        :(VerilogAEnvironment.CedarSim.@kwdef struct $symname <: VerilogAEnvironment.VAModel
            $(struct_fields...)
        end),
        sim,
    )
end


#==============================================================================#
# MNA Device Generation (Phase 5)
#
# Generates stamp! methods for Verilog-A devices instead of DAECompiler code.
# Uses s-dual approach for automatic resist/react separation.
#==============================================================================#

"""
    make_mna_device(vm::VANode{VerilogModule})

Generate MNA-compatible Julia code for a Verilog-A module.

Unlike `make_spice_device` which generates code for DAECompiler,
this generates `stamp!` methods that work with MNAContext directly.

# Generated Code Structure
```julia
@kwdef struct DeviceName <: VAModel
    param1::DefaultOr{Float64} = default1
    ...
end

function MNA.stamp!(dev::DeviceName, ctx::MNAContext, p::Int, n::Int;
                    t::Real=0.0, mode::Symbol=:dcop)
    # Parameter extraction
    param1 = undefault(dev.param1)

    # Contribution function (captures parameters)
    function contrib(Vpn)
        # VA analog block translated to Julia
        # ddt(x) becomes va_ddt(x)
        Vpn / param1  # Example: resistor
    end

    # Stamp contribution (uses AD for Jacobian)
    stamp_current_contribution!(ctx, p, n, contrib, zeros(max(p, n)))
end
```
"""
function make_mna_device(vm::VANode{VerilogModule})
    ps = pins(vm)
    modname = String(vm.id)
    symname = Symbol(modname)

    # Collect struct fields and parameters
    struct_fields = Any[]
    parameter_names = Set{Symbol}()
    param_defaults = Dict{Symbol, Any}()  # Store parameter default expressions
    to_julia_global = Scope()
    find_ddx!(to_julia_global.ddx_order, vm)

    to_julia_defaults = Scope(to_julia_global.parameters,
        to_julia_global.node_order, to_julia_global.ninternal_nodes,
        to_julia_global.branch_order, to_julia_global.used_branches,
        to_julia_global.var_types, to_julia_global.all_functions,
        true, to_julia_global.ddx_order)

    internal_nodes = Vector{Symbol}()
    var_types = Dict{Symbol, Union{Type{Int}, Type{Float64}}}()
    var_inits = Dict{Symbol, Any}()  # Store variable initialization expressions
    aliases = Dict{Symbol, Symbol}()

    # Pre-pass: collect parameters and nodes
    for child in vm.items
        item = child.item
        @case formof(item) begin
            InOutDeclaration => nothing
            NetDeclaration => begin
                for net in item.net_names
                    id = Symbol(assemble_id_string(net.item))
                    if !(id in ps)
                        push!(internal_nodes, id)
                    end
                end
            end
            ParameterDeclaration => begin
                for param in item.params
                    param = param.item
                    pT = Float64
                    if item.ptype !== nothing
                        pT = kw_to_T(item.ptype.kw)
                    end
                    paramname = String(assemble_id_string(param.id))
                    paramsym = Symbol(paramname)
                    push!(parameter_names, paramsym)
                    # Extract default value from param.default_expr
                    if param.default_expr !== nothing
                        # Parse the default expression using the scope
                        param_defaults[paramsym] = to_julia_defaults(param.default_expr)
                    else
                        param_defaults[paramsym] = 0.0  # Fallback
                    end
                    # Use simplest possible field - just type annotation, no default
                    # @kwdef will use the type's default constructor
                    field_expr = Expr(:(::), paramsym, :(CedarSim.DefaultOr{Float64}))
                    push!(struct_fields, field_expr)
                    var_types[Symbol(paramname)] = pT
                end
            end
            AliasParameterDeclaration => begin
                param = item
                paramsym = Symbol(assemble_id_string(param.id))
                targetsym = Symbol(assemble_id_string(param.value))
                push!(parameter_names, paramsym)
                aliases[paramsym] = targetsym
            end
            IntRealDeclaration => begin
                T = kw_to_T(item.kw.kw)
                for ident in item.idents
                    # ident.item is IntRealVarDecl with id, eq, init fields
                    vardecl = ident.item
                    name = Symbol(assemble_id_string(vardecl.id))
                    var_types[name] = T
                    # Capture initialization expression if present
                    if vardecl.init !== nothing
                        var_inits[name] = to_julia_defaults(vardecl.init)
                    end
                end
            end
        end
    end

    # Second pre-pass: collect named branches (branch declarations like "branch (pos, neg) br;")
    named_branches = Dict{Symbol, Pair{Symbol,Symbol}}()
    for child in vm.items
        item = child.item
        @case formof(item) begin
            BranchDeclaration => begin
                # Extract references (nodes in parentheses)
                refs = Symbol[]
                for ref in item.references
                    push!(refs, Symbol(assemble_id_string(ref.item)))
                end
                @assert length(refs) == 2 "Branch declaration must have exactly 2 nodes"
                pos_node, neg_node = refs[1], refs[2]

                # Extract branch identifiers
                for bid in item.ids
                    branch_name = Symbol(assemble_id_string(bid.item.id))
                    named_branches[branch_name] = pos_node => neg_node
                end
            end
            _ => nothing
        end
    end

    # Build scope for code generation
    node_order = [ps; internal_nodes; Symbol("0")]
    to_julia_mna = MNAScope(parameter_names, node_order, length(internal_nodes),
        collect(map(x->Pair(x...), combinations(node_order, 2))),
        Set{Pair{Symbol}}(),
        var_types,
        Dict{Symbol, VAFunction}(), false,
        to_julia_global.ddx_order,
        named_branches)

    # Generate analog block code
    analog_body = Expr(:block)
    contributions = Any[]
    function_defs = Any[]

    for child in vm.items
        item = child.item
        @case formof(item) begin
            InOutDeclaration => nothing
            IntRealDeclaration => nothing
            NetDeclaration => nothing
            BranchDeclaration => nothing
            ParameterDeclaration => nothing
            AliasParameterDeclaration => nothing
            AnalogFunctionDeclaration => begin
                push!(function_defs, to_julia_mna(item))
            end
            AnalogBlock => begin
                # Collect contributions from analog block
                mna_collect_contributions!(contributions, to_julia_mna, item.stmt)
            end
            _ => nothing
        end
    end

    # Generate parameter extraction
    params_to_locals = map(collect(parameter_names)) do id
        :($id = $(undefault)(dev.$id))
    end

    # Generate variable declarations for non-parameter local vars
    # Use initialization expression if provided, otherwise default to zero
    local_var_decls = Any[]
    for (name, T) in var_types
        if !(name in parameter_names)
            init_expr = get(var_inits, name, :(zero($T)))
            push!(local_var_decls, :(local $name::$T = $init_expr))
        end
    end

    # Generate stamp method using unified n-terminal approach
    # (works for any number of terminals, including 2)
    port_args = ps
    stamp_method = generate_mna_stamp_method_nterm(
        symname, ps, port_args, internal_nodes, params_to_locals, local_var_decls,
        function_defs, contributions, to_julia_mna)

    # Build struct and constructor directly without @kwdef to avoid macro hygiene issues
    # that rename field symbols in baremodule contexts

    # Filter out alias parameters from struct fields (aliases don't need storage)
    real_params = filter(p -> !haskey(aliases, p), parameter_names)

    # 1. Build plain struct definition (only real parameters, not aliases)
    struct_body = Expr(:block)
    for paramsym in real_params
        push!(struct_body.args, Expr(:(::), paramsym, :(CedarSim.DefaultOr{Float64})))
    end
    struct_def = Expr(:struct, false,
        Expr(:<:, symname, :(VerilogAEnvironment.VAModel)),
        struct_body)

    # 2. Build keyword constructor that mimics @kwdef
    # Constructor accepts both real params and aliases
    # Aliases forward their value to the target parameter
    if !isempty(real_params) || !isempty(aliases)
        # Build keyword parameter list with defaults
        kw_params = Expr(:parameters)
        call_args = Any[]

        # Add real parameters
        for paramsym in real_params
            # Each parameter: paramsym = mkdefault(default_value)
            # Use the actual default value from the VA parameter declaration
            default_val = get(param_defaults, paramsym, 0.0)
            push!(kw_params.args, Expr(:kw, paramsym, :(CedarSim.mkdefault($default_val))))
        end

        # Add alias parameters (they default to nothing, meaning "use target's value")
        for (alias_sym, _) in aliases
            push!(kw_params.args, Expr(:kw, alias_sym, :nothing))
        end

        # Build call args: for each real param, check if an alias was provided
        for paramsym in real_params
            # Find aliases that target this parameter
            targeting_aliases = [a for (a, t) in aliases if t == paramsym]
            if isempty(targeting_aliases)
                # No aliases, just use the parameter directly
                push!(call_args, Expr(:call, :(VerilogAEnvironment.vaconvert),
                    :(CedarSim.notdefault(fieldtype($symname, $(QuoteNode(paramsym))))),
                    paramsym))
            else
                # Has aliases - use alias value if provided, otherwise use parameter
                # Build: something(alias1, alias2, ..., paramsym) where something picks first non-nothing
                alias_expr = paramsym
                for alias_sym in reverse(targeting_aliases)
                    alias_expr = :($alias_sym !== nothing ? $alias_sym : $alias_expr)
                end
                push!(call_args, Expr(:call, :(VerilogAEnvironment.vaconvert),
                    :(CedarSim.notdefault(fieldtype($symname, $(QuoteNode(paramsym))))),
                    alias_expr))
            end
        end

        # Build constructor function
        constructor = Expr(:function,
            Expr(:call, symname, kw_params),
            Expr(:call, symname, call_args...))
    else
        constructor = nothing
    end

    # 3. Build getproperty override to support alias access
    # Base.getproperty(dev::TypeName, s::Symbol) = s == :alias ? getfield(dev, :target) : getfield(dev, s)
    getproperty_override = nothing
    if !isempty(aliases)
        # Build the if-elseif chain from inside out (rightmost first)
        # Start with the final else clause
        alias_list = collect(aliases)
        alias_checks = :(getfield(dev, s))  # Default: direct field access

        # Build chain from last alias to first
        for i in length(alias_list):-1:1
            (alias_sym, target_sym) = alias_list[i]
            check = :(s == $(QuoteNode(alias_sym)))
            result = :(getfield(dev, $(QuoteNode(target_sym))))
            alias_checks = Expr(:if, check, result, alias_checks)
        end

        getproperty_override = :(function Base.getproperty(dev::$symname, s::Symbol)
            $alias_checks
        end)
    end

    result_args = Any[struct_def]
    if constructor !== nothing
        push!(result_args, constructor)
    end
    if getproperty_override !== nothing
        push!(result_args, getproperty_override)
    end
    push!(result_args, stamp_method)

    Expr(:toplevel, result_args...)
end

"""
MNA-specific scope that translates VA constructs for MNA stamping.
"""
struct MNAScope
    parameters::Set{Symbol}
    node_order::Vector{Symbol}
    ninternal_nodes::Int
    branch_order::Vector{Pair{Symbol}}
    used_branches::Set{Pair{Symbol}}
    var_types::Dict{Symbol, Union{Type{Int}, Type{Float64}}}
    all_functions::Dict{Symbol, VAFunction}
    undefault_ids::Bool
    ddx_order::Vector{Symbol}
    named_branches::Dict{Symbol, Pair{Symbol,Symbol}}  # Maps branch name -> (pos, neg) nodes
end

# Forward basic translations to Scope
(s::MNAScope)(x::VANode{Literal}) = Scope()(x)
(s::MNAScope)(x::VANode{FloatLiteral}) = Scope()(x)

function (scope::MNAScope)(ip::VANode{IdentifierPrimary})
    Symbol(assemble_id_string(ip.id))
end

function (to_julia::MNAScope)(cs::VANode{BinaryExpression})
    op = Symbol(cs.op)
    if op == :(||)
        return Expr(:call, (|), to_julia(cs.lhs), to_julia(cs.rhs))
    elseif op == :(&&)
        return Expr(:call, (&), to_julia(cs.lhs), to_julia(cs.rhs))
    else
        return Expr(:call, op, to_julia(cs.lhs), to_julia(cs.rhs))
    end
end

function (to_julia::MNAScope)(stmt::VANode{UnaryOp})
    return Expr(:call, Symbol(stmt.op), to_julia(stmt.operand))
end

function (to_julia::MNAScope)(stmt::VANode{Parens})
    return to_julia(stmt.inner)
end

function (to_julia::MNAScope)(cs::VANode{TernaryExpr})
    return Expr(:if, to_julia(cs.condition), to_julia(cs.ifcase), to_julia(cs.elsecase))
end

function (to_julia::MNAScope)(stmt::VANode{FunctionCall})
    fname = Symbol(stmt.id)

    if fname == :V
        # Voltage access - return variable name that will be replaced with Vpn
        @assert length(stmt.args) in (1, 2)
        id1 = Symbol(stmt.args[1].item)

        # Check if this is a named branch (e.g., V(br) where br is a branch)
        if length(stmt.args) == 1 && haskey(to_julia.named_branches, id1)
            # Named branch: V(br) -> V_pos - V_neg
            branch_nodes = to_julia.named_branches[id1]
            pos_node, neg_node = branch_nodes.first, branch_nodes.second
            push!(to_julia.used_branches, pos_node => neg_node)
            if neg_node == Symbol("0")
                return pos_node
            else
                return :($pos_node - $neg_node)
            end
        end

        id2 = length(stmt.args) > 1 ? Symbol(stmt.args[2].item) : Symbol("0")
        push!(to_julia.used_branches, id1 => id2)

        if id2 == Symbol("0")
            return id1
        else
            return :($id1 - $id2)
        end
    elseif fname == :I
        # Current access
        @assert length(stmt.args) in (1, 2)

        if length(stmt.args) == 1
            # I(br) - current through named branch
            branch_name = Symbol(stmt.args[1].item)
            if haskey(to_julia.named_branches, branch_name)
                # Named branch: I(br) returns the branch current variable
                # The current variable is accessed via a special symbol that will be
                # replaced with the actual current index in the generated code
                branch_nodes = to_julia.named_branches[branch_name]
                push!(to_julia.used_branches, branch_nodes.first => branch_nodes.second)
                # Return a symbol that represents the branch current
                # This will be provided as a variable in the generated stamp function
                return Symbol("_I_branch_", branch_name)
            else
                return :(error("I() with single argument requires a named branch"))
            end
        else
            # I(a, b) - not directly supported in contribution-based stamping
            return :(error("I(a,b) probe not supported in MNA contribution"))
        end
    elseif fname == :ddt
        # Time derivative - use va_ddt
        return Expr(:call, :va_ddt, to_julia(stmt.args[1].item))
    elseif fname == :ddx
        # Partial derivative - ddx(expr, V(a,b)) returns ∂expr/∂V(a,b)
        # For n-terminal MNA devices, duals are indexed by node_order (port positions)
        item = stmt.args[2].item
        @assert formof(item) == FunctionCall
        @assert Symbol(item.id) == :V
        if length(item.args) == 1
            probe = Symbol(item.args[1].item)
            # Use node_order for partial index (duals indexed by port position)
            id_idx = findfirst(==(probe), to_julia.node_order)
            return :(let x = $(to_julia(stmt.args[1].item))
                isa(x, Dual) ? @inbounds(ForwardDiff.partials(x, $id_idx)) : 0.0
            end)
        else
            probe1 = Symbol(item.args[1].item)
            id1_idx = findfirst(==(probe1), to_julia.node_order)
            probe2 = Symbol(item.args[2].item)
            id2_idx = findfirst(==(probe2), to_julia.node_order)
            # ∂expr/∂V(a,b) = (∂expr/∂V_a - ∂expr/∂V_b) / 2
            # This works because V(a,b) = V_a - V_b, so:
            # ∂expr/∂V_a = ∂expr/∂V(a,b) and ∂expr/∂V_b = -∂expr/∂V(a,b)
            return :(let x = $(to_julia(stmt.args[1].item)),
                        dx1 = isa(x, Dual) ? @inbounds(ForwardDiff.partials(x, $id1_idx)) : 0.0,
                        dx2 = isa(x, Dual) ? @inbounds(ForwardDiff.partials(x, $id2_idx)) : 0.0
                (dx1 - dx2) / 2
            end)
        end
    elseif fname == Symbol("\$temperature")
        return :(spec.temp + 273.15)  # Convert to Kelvin
    elseif fname == Symbol("\$vt")
        return :((spec.temp + 273.15) * 8.617333262e-5)  # kT/q
    elseif fname == Symbol("\$param_given")
        # Check if a parameter was explicitly specified (not using default)
        id = Symbol(stmt.args[1].item)
        return Expr(:call, :!, Expr(:call, CedarSim.isdefault,
            Expr(:., :dev, QuoteNode(id))))
    elseif fname == Symbol("\$simparam")
        # Simulator parameter access - $simparam("name") or $simparam("name", default)
        if stmt.args[1].item isa VANode{StringLiteral}
            param_str = String(stmt.args[1].item)[2:end-1]  # Strip quotes
        else
            param_str = String(stmt.args[1].item)
        end
        param_sym = Symbol(param_str)
        if length(stmt.args) == 1
            # No default - error if not found
            return :(hasproperty(spec, $(QuoteNode(param_sym))) ?
                     getproperty(spec, $(QuoteNode(param_sym))) :
                     error("Unknown simparam: " * $param_str))
        else
            # With default value
            default_expr = to_julia(stmt.args[2].item)
            return :(hasproperty(spec, $(QuoteNode(param_sym))) ?
                     getproperty(spec, $(QuoteNode(param_sym))) :
                     $default_expr)
        end
    elseif fname == :analysis
        # Analysis type check - returns true if current analysis matches the string
        # Mapping:
        #   "dc" or "static" -> spec.mode == :dcop
        #   "tran" or "transient" -> spec.mode == :tran
        #   "ac" -> spec.mode == :ac
        #   "nodeset" -> false (not supported)
        @assert length(stmt.args) == 1 "analysis() takes exactly one argument"
        analysis_str = to_julia(stmt.args[1].item)
        if analysis_str isa String
            analysis_sym = analysis_str
        else
            # If it's not a constant string, evaluate at runtime
            return :(
                let atype = $analysis_str
                    if atype == "dc" || atype == "static"
                        spec.mode == :dcop
                    elseif atype == "tran" || atype == "transient"
                        spec.mode == :tran
                    elseif atype == "ac"
                        spec.mode == :ac
                    else
                        false
                    end
                end
            )
        end
        # For constant strings, generate simpler code
        if analysis_sym == "dc" || analysis_sym == "static"
            return :(spec.mode == :dcop)
        elseif analysis_sym == "tran" || analysis_sym == "transient"
            return :(spec.mode == :tran)
        elseif analysis_sym == "ac"
            return :(spec.mode == :ac)
        else
            return false
        end
    elseif fname == Symbol("\$limit")
        # $limit(voltage, limiter_fn, ...) - voltage limiting for Newton convergence
        # In our MNA implementation, we can optionally apply limiting or just return the voltage
        # For now, we simply return the voltage value without limiting
        # This allows the model to run, though convergence may be slower
        voltage_expr = to_julia(stmt.args[1].item)
        return voltage_expr
    end

    # Noise functions - return 0 in MNA (noise not simulated in DC/transient)
    if fname in (:white_noise, :flicker_noise)
        return 0.0
    end

    # Check for VA-defined function
    vaf = get(to_julia.all_functions, fname, nothing)
    if vaf !== nothing
        # Call to a Verilog-A defined function - handle output/inout parameters
        args = map(x -> to_julia(x.item), stmt.args)
        in_args = Any[]
        out_args = Any[]

        if length(args) != length(vaf.arg_order)
            return Expr(:call, error, "Wrong number of arguments to function $fname ($args, $(vaf.arg_order))")
        end

        for (arg, vaarg) in zip(args, vaf.arg_order)
            kind = vaf.inout_decls[vaarg]
            if kind == :output || kind == :inout
                isa(arg, Symbol) || return Expr(:call, error, "Output argument $vaarg to function $fname must be a symbol. Got $arg.")
                push!(out_args, arg)
            end
            if kind == :input || kind == :inout
                push!(in_args, arg)
            end
        end
        ret = Expr(:call, fname, in_args...)
        if length(out_args) != 0
            s = gensym()
            ret = @nolines quote
                ($s, ($(out_args...),)) = $ret
                $s
            end
        end
        return ret
    end

    # Default: pass through function call
    # Strip $ prefix for system functions (e.g., $pow -> pow, $ln -> ln)
    fname_str = String(fname)
    if startswith(fname_str, "\$")
        fname = Symbol(fname_str[2:end])
    end
    return Expr(:call, fname, map(x -> to_julia(x.item), stmt.args)...)
end

function (to_julia::MNAScope)(stmt::VANode{AnalogVariableAssignment})
    assignee = Symbol(stmt.lvalue)
    varT = get(to_julia.var_types, assignee, Float64)

    eq = stmt.eq.op
    op = eq == VAT.EQ ? :(=) :
         eq == VAT.PLUS_EQ ? :(+=) :
         eq == VAT.MINUS_EQ ? :(-=) :
         eq == VAT.STAR_EQ ? :(*=) :
         eq == VAT.SLASH_EQ ? :(/=) :
         :(=)

    return Expr(op, assignee,
        Expr(:call, VerilogAEnvironment.vaconvert, varT, to_julia(stmt.rvalue)))
end

function (to_julia::MNAScope)(stmt::VANode{AnalogProceduralAssignment})
    return to_julia(stmt.assign)
end

function (to_julia::MNAScope)(asb::VANode{AnalogSeqBlock})
    ret = Expr(:block)
    for stmt in asb.stmts
        push!(ret.args, to_julia(stmt))
    end
    ret
end

(to_julia::MNAScope)(stmt::VANode{AnalogStatement}) = to_julia(stmt.stmt)

function (to_julia::MNAScope)(stmt::VANode{AnalogConditionalBlock})
    aif = stmt.aif
    function if_body_to_julia(ifstmt)
        if formof(ifstmt) == AnalogSeqBlock
            return to_julia(ifstmt)
        else
            return Expr(:block, to_julia(ifstmt))
        end
    end

    # Convert VA condition to boolean - in VA, integers are truthy (non-zero = true)
    function va_condition_to_bool(cond_expr)
        # Wrap in !iszero() to convert any numeric type to Bool
        :(!(iszero($(to_julia(cond_expr)))))
    end

    ifex = ex = Expr(:if, va_condition_to_bool(aif.condition), if_body_to_julia(aif.stmt))
    for case in stmt.elsecases
        if formof(case.stmt) == AnalogIf
            elif = case.stmt
            newex = Expr(:elseif, va_condition_to_bool(elif.condition), if_body_to_julia(elif.stmt))
            push!(ex.args, newex)
            ex = newex
        else
            push!(ex.args, if_body_to_julia(case.stmt))
        end
    end
    ifex
end

# Handle system identifiers like $mfactor
function (to_julia::MNAScope)(ip::VANode{SystemIdentifier})
    id = Symbol(ip)
    if id == Symbol("\$mfactor")
        # Device multiplicity - default to 1.0
        return :(hasproperty(spec, :mfactor) ? spec.mfactor : 1.0)
    elseif id == Symbol("\$temperature")
        return :(spec.temp + 273.15)
    else
        # For other system identifiers, return as function call
        return Expr(:call, id)
    end
end

# Handle analog system task enable (e.g., $warning, $strobe)
function (to_julia::MNAScope)(stmt::VANode{AnalogSystemTaskEnable})
    if formof(stmt.task) == FunctionCall
        fc = stmt.task
        fname = Symbol(fc.id)
        args = map(x -> to_julia(x.item), fc.args)
        if fname == Symbol("\$warning")
            # Warnings are suppressed in MNA simulation
            return nothing
        elseif fname == Symbol("\$strobe")
            return nothing
        elseif fname == Symbol("\$error")
            return Expr(:call, :error, args...)
        elseif fname == Symbol("\$discontinuity")
            # Discontinuity markers are no-ops in MNA
            return nothing
        else
            # Default: treat as regular function call (strip $ prefix)
            fname_str = String(fname)
            if startswith(fname_str, "\$")
                fname = Symbol(fname_str[2:end])
            end
            return Expr(:call, fname, args...)
        end
    else
        return nothing
    end
end

# Handle string literals
function (to_julia::MNAScope)(stmt::VANode{StringLiteral})
    return String(stmt)[2:end-1]  # Strip quotes
end

# Handle analog for loops
function (to_julia::MNAScope)(stmt::VANode{AnalogFor})
    body = to_julia(stmt.stmt)
    push!(body.args, to_julia(stmt.update_stmt))
    # Convert VA condition to boolean
    cond = :(!(iszero($(to_julia(stmt.cond_expr)))))
    while_expr = Expr(:while, cond, body)
    Expr(:block, to_julia(stmt.init_stmt), while_expr)
end

# Handle analog while loops
function (to_julia::MNAScope)(stmt::VANode{AnalogWhile})
    body = to_julia(stmt.stmt)
    # Convert VA condition to boolean
    cond = :(!(iszero($(to_julia(stmt.cond_expr)))))
    Expr(:while, cond, body)
end

# Handle analog repeat loops
function (to_julia::MNAScope)(stmt::VANode{AnalogRepeat})
    body = to_julia(stmt.stmt)
    Expr(:for, :(_ = 1:$(stmt.num_repeat)), body)
end

# Handle contribution statements inside conditionals
# When a contribution is inside an if-block, we generate inline stamping code
function (to_julia::MNAScope)(cs::VANode{ContributionStatement})
    bpfc = cs.lvalue
    kind_sym = Symbol(bpfc.id)
    kind = kind_sym == :I ? :current : kind_sym == :V ? :voltage : :unknown

    refs = map(bpfc.references) do ref
        Symbol(assemble_id_string(ref.item))
    end

    p_sym = refs[1]
    n_sym = length(refs) > 1 ? refs[2] : Symbol("0")
    push!(to_julia.used_branches, p_sym => n_sym)

    # Get the node variable names (using the convention from generate_mna_stamp_method_nterm)
    # Handle ground node (Symbol("0")) specially - it maps to integer 0
    p_idx = p_sym == Symbol("0") ? nothing : findfirst(==(p_sym), to_julia.node_order)
    n_idx = n_sym == Symbol("0") ? nothing : findfirst(==(n_sym), to_julia.node_order)
    # Ground node uses literal 0; other nodes use their parameter symbol
    p_node = p_idx === nothing ? 0 : Symbol("_node_", p_sym)
    n_node = n_idx === nothing ? 0 : Symbol("_node_", n_sym)

    # For voltage contributions (V(a,b) <+ expr), we need different handling
    if kind == :voltage
        # Voltage contribution: V(p,n) <+ value means we enforce V_p - V_n = value
        # This is typically done with a voltage source pattern
        # For now, generate inline stamping (simplified)
        expr = to_julia(cs.assign_expr)
        return quote
            # Voltage contribution V($p_sym, $n_sym) <+ $expr
            let v_contrib = Float64($expr)
                # For V <+ 0, this is a short circuit - stamp high conductance
                # This is a simplified implementation; proper handling needs voltage source stamping
                if abs(v_contrib) < 1e-15
                    # Short circuit: stamp high conductance
                    let G_short = 1e12
                        CedarSim.MNA.stamp_G!(ctx, $p_node, $p_node, G_short)
                        CedarSim.MNA.stamp_G!(ctx, $p_node, $n_node, -G_short)
                        CedarSim.MNA.stamp_G!(ctx, $n_node, $p_node, -G_short)
                        CedarSim.MNA.stamp_G!(ctx, $n_node, $n_node, G_short)
                    end
                end
            end
        end
    end

    # Current contribution: generate inline contribution stamping
    # This is used when contributions are inside conditionals
    expr = to_julia(cs.assign_expr)
    n_all_nodes = length(to_julia.node_order)

    # p_node and n_node already set above (handles ground node correctly)

    # For contributions inside conditionals, we evaluate and stamp inline
    return quote
        # Contribution I($p_sym, $n_sym) <+ ...
        let I_branch = $expr
            # Extract value and partials from the dual
            if I_branch isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
                # Pure reactive
                I_val = ForwardDiff.value(ForwardDiff.value(I_branch))
                q_val = ForwardDiff.value(ForwardDiff.partials(I_branch, 1))
            elseif I_branch isa ForwardDiff.Dual
                _val = ForwardDiff.value(I_branch)
                if _val isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
                    I_val = ForwardDiff.value(ForwardDiff.value(_val))
                    q_val = ForwardDiff.value(ForwardDiff.partials(_val, 1))
                else
                    I_val = _val isa ForwardDiff.Dual ? ForwardDiff.value(_val) : Float64(_val)
                    q_val = 0.0
                end
            else
                I_val = I_branch isa Real ? Float64(I_branch) : 0.0
                q_val = 0.0
            end

            # For now, stamp simplified Jacobian based on known nodes
            # This is approximate - proper handling requires full partials extraction
            # like in generate_mna_stamp_method_nterm
            if $p_node != 0
                CedarSim.MNA.stamp_b!(ctx, $p_node, -I_val)
            end
            if $n_node != 0
                CedarSim.MNA.stamp_b!(ctx, $n_node, I_val)
            end
        end
    end
end

# Handle case statements
function (to_julia::MNAScope)(stmt::VANode{CaseStatement})
    s = gensym()
    first = true
    expr = nothing
    default_case = nothing
    for case in stmt.cases
        if isa(case.conds, Node)
            # Default case
            default_case = to_julia(case.item)
        else
            conds = map(cond -> :($s == $(to_julia(cond.item))), case.conds)
            cond = length(conds) == 1 ? conds[1] : Expr(:(||), conds...)
            ex = Expr(first ? :if : :elseif, cond, to_julia(case.item))
            if first
                expr = ex
                first = false
            else
                push!(expr.args, ex)
            end
        end
    end
    if expr === nothing
        return default_case !== nothing ? default_case : Expr(:block)
    end
    default_case !== nothing && push!(expr.args, default_case)
    Expr(:block, :($s = $(to_julia(stmt.switch))), expr)
end

function (to_julia::MNAScope)(fd::VANode{AnalogFunctionDeclaration})
    # Similar to Scope version but uses MNAScope for body
    type_decls = Dict{Symbol, Any}()
    inout_decls = Dict{Symbol, Symbol}()
    fname = Symbol(fd.id)
    var_types = Dict{Symbol, Union{Type{Int}, Type{Float64}}}()
    rt = fd.fty === nothing ? Real : kw_to_T(fd.fty.kw)
    var_types[fname] = rt
    arg_order = Symbol[]

    for decl in fd.items
        item = decl.item
        @case formof(item) begin
            InOutDeclaration => begin
                kind = item.kw.kw === INPUT ? :input :
                       item.kw.kw === OUTPUT ? :output :
                       :inout
                for name in item.portnames
                    ns = Symbol(name.item)
                    inout_decls[ns] = kind
                    push!(arg_order, ns)
                end
            end
            IntRealDeclaration => begin
                T = kw_to_T(item.kw.kw)
                for ident in item.idents
                    # ident.item is IntRealVarDecl with id, eq, init fields
                    vardecl = ident.item
                    name = Symbol(assemble_id_string(vardecl.id))
                    var_types[name] = T
                end
            end
        end
    end

    to_julia_internal = MNAScope(to_julia.parameters, to_julia.node_order,
        to_julia.ninternal_nodes, to_julia.branch_order, to_julia.used_branches, var_types,
        to_julia.all_functions, to_julia.undefault_ids, to_julia.ddx_order,
        to_julia.named_branches)

    in_args = [k for k in arg_order if inout_decls[k] in (:input, :inout)]
    out_args = [k for k in arg_order if inout_decls[k] in (:output, :inout)]
    rt_decl = length(out_args) == 0 ? fname : :(($fname, ($(out_args...),)))

    to_julia.all_functions[fname] = VAFunction(arg_order, inout_decls)

    localize_vars = Any[]
    for var in keys(var_types)
        var in arg_order && continue
        push!(localize_vars, :(local $var))
    end

    return @nolines quote
        @inline function $fname($(in_args...))
            $(localize_vars...)
            local $fname = VerilogAEnvironment.vaconvert($rt, 0)
            $(to_julia_internal(fd.stmt))
            return $rt_decl
        end
    end
end

"""
Collect contribution statements and regular statements from analog block for MNA stamping.
"""
function mna_collect_contributions!(contributions, to_julia::MNAScope, stmt)
    if stmt isa VANode{ContributionStatement}
        push!(contributions, mna_translate_contribution(to_julia, stmt))
    elseif stmt isa VANode{AnalogSeqBlock}
        for s in stmt.stmts
            mna_collect_contributions!(contributions, to_julia, s)
        end
    elseif stmt isa VANode{AnalogStatement}
        mna_collect_contributions!(contributions, to_julia, stmt.stmt)
    elseif stmt isa VANode{AnalogConditionalBlock}
        # For conditional blocks, we need to handle them specially
        # For now, add the whole translated block
        push!(contributions, (kind=:conditional, expr=to_julia(stmt)))
    elseif stmt isa VANode{AnalogVariableAssignment}
        # Regular assignments (e.g., cdrain = R*V(g,s)**2)
        push!(contributions, (kind=:assignment, expr=to_julia(stmt)))
    elseif stmt isa VANode{AnalogProceduralAssignment}
        # Procedural assignments in analog block (e.g., cdrain = R*V(g,s)**2;)
        push!(contributions, (kind=:assignment, expr=to_julia(stmt)))
    end
end

"""
Translate a contribution statement for MNA.
"""
function mna_translate_contribution(to_julia::MNAScope, cs::VANode{ContributionStatement})
    bpfc = cs.lvalue
    kind_sym = Symbol(bpfc.id)
    kind = kind_sym == :I ? :current : kind_sym == :V ? :voltage : :unknown

    refs = map(bpfc.references) do ref
        Symbol(assemble_id_string(ref.item))
    end

    if length(refs) == 1
        node = refs[1]
        # Check if this is a named branch (e.g., V(br) or I(br) where br is a branch)
        if haskey(to_julia.named_branches, node)
            branch_nodes = to_julia.named_branches[node]
            pos_node, neg_node = branch_nodes.first, branch_nodes.second
            push!(to_julia.used_branches, pos_node => neg_node)
            # Return with branch_name to indicate this is a named branch contribution
            return (kind=kind, p=pos_node, n=neg_node, expr=to_julia(cs.assign_expr),
                    branch_name=node, is_branch=true)
        else
            push!(to_julia.used_branches, node => Symbol("0"))
            return (kind=kind, p=node, n=Symbol("0"), expr=to_julia(cs.assign_expr),
                    branch_name=nothing, is_branch=false)
        end
    elseif length(refs) == 2
        (id1, id2) = refs
        push!(to_julia.used_branches, id1 => id2)
        return (kind=kind, p=id1, n=id2, expr=to_julia(cs.assign_expr),
                branch_name=nothing, is_branch=false)
    end

    return (kind=:unknown, expr=:(error("Invalid contribution")),
            branch_name=nothing, is_branch=false)
end

"""
Generate stamp! method for n-terminal device (potentially with internal nodes).

For n-terminal devices with internal nodes, we use a vector-valued dual approach:
1. Allocate internal nodes using alloc_internal_node! (done once per context)
2. Create duals with partials for each node voltage (terminals + internal)
3. Evaluate the contribution expression
4. Extract ∂I/∂V_k for each node k and stamp into G matrix

# Arguments
- `symname`: Module/device name symbol
- `ps`: Port/terminal symbols (e.g., [:p, :n])
- `port_args`: Port argument symbols for the stamp method
- `internal_nodes`: Internal node symbols declared in the module
- `params_to_locals`: Parameter extraction expressions
- `local_var_decls`: Local variable declarations
- `function_defs`: VA function definitions
- `contributions`: Branch contribution tuples
- `to_julia`: MNAScope for code translation
"""
function generate_mna_stamp_method_nterm(symname, ps, port_args, internal_nodes, params_to_locals,
                                          local_var_decls, function_defs, contributions,
                                          to_julia)
    n_ports = length(port_args)
    n_internal = length(internal_nodes)
    n_all_nodes = n_ports + n_internal

    # Create unique node parameter names (prefixed to avoid conflict with voltage vars)
    # Terminal nodes come from function arguments
    node_params = [Symbol("_node_", p) for p in port_args]
    # Internal node indices will be allocated at runtime
    internal_node_params = [Symbol("_node_", n) for n in internal_nodes]
    # All node symbols for dual creation (terminals + internal)
    all_node_syms = [port_args; internal_nodes]
    all_node_params = [node_params; internal_node_params]

    # Build the contribution evaluation body - includes local vars and expressions
    # that compute the current contributions
    contrib_eval = Expr(:block)
    # For n-terminal devices:
    # 1. Don't use `local` - variables need to be visible in outer scope for stamp_code
    # 2. Use parametric type based on first port dual so Float64 variables can hold Duals
    # 3. Integer variables stay as Int (for control flow - booleans, counters)
    first_port = port_args[1]
    for decl in local_var_decls
        # Convert `local name::T = init_expr` appropriately
        # Float64 vars: promote to Dual-compatible type
        # Int vars: keep as scalar (used for booleans/control flow)
        if decl.head == :local
            inner = decl.args[1]
            if inner isa Expr && inner.head == :(=)
                lhs = inner.args[1]
                rhs = inner.args[2]
                if lhs isa Expr && lhs.head == :(::)
                    name = lhs.args[1]
                    var_type = lhs.args[2]  # Type annotation (Int or Float64)

                    # Check if this is an integer type (control flow variable)
                    is_integer_type = var_type == :Int || var_type == Int

                    if is_integer_type
                        # Integer variables: keep as scalar, don't promote
                        push!(contrib_eval.args, :($name = $rhs))
                    elseif rhs isa Expr && rhs.head == :call && rhs.args[1] == :zero
                        # zero() call - use zero(typeof(...)) for type compatibility
                        push!(contrib_eval.args, :($name = zero(typeof($first_port))))
                    else
                        # Float64: preserve init value, add zero for type promotion
                        push!(contrib_eval.args, :($name = $rhs + zero(typeof($first_port))))
                    end
                else
                    # If no type annotation, still strip `local`
                    push!(contrib_eval.args, inner)
                end
            elseif inner isa Expr && inner.head == :(::)
                # Just type annotation, no initialization
                name = inner.args[1]
                var_type = inner.args[2]
                is_integer_type = var_type == :Int || var_type == Int
                if is_integer_type
                    push!(contrib_eval.args, :($name = 0))
                else
                    push!(contrib_eval.args, :($name = zero(typeof($first_port))))
                end
            else
                # Plain assignment inside local - just use the assignment
                push!(contrib_eval.args, inner)
            end
        else
            push!(contrib_eval.args, decl)
        end
    end

    # Collect current contributions by branch, and voltage contributions for named branches
    branch_contribs = Dict{Tuple{Symbol,Symbol}, Vector{Any}}()
    voltage_branch_contribs = Dict{Symbol, NamedTuple}()  # branch_name -> (p, n, exprs)

    for c in contributions
        if c.kind == :current
            branch = (c.p, c.n)
            if !haskey(branch_contribs, branch)
                branch_contribs[branch] = Any[]
            end
            push!(branch_contribs[branch], c.expr)
        elseif c.kind == :voltage && hasproperty(c, :is_branch) && c.is_branch
            # Voltage contribution to a named branch (e.g., V(br) <+ expr for inductor)
            branch_name = c.branch_name
            if !haskey(voltage_branch_contribs, branch_name)
                voltage_branch_contribs[branch_name] = (p=c.p, n=c.n, exprs=Any[])
            end
            push!(voltage_branch_contribs[branch_name].exprs, c.expr)
        elseif c.kind == :conditional
            push!(contrib_eval.args, c.expr)
        elseif c.kind == :assignment
            # Regular assignment (e.g., cdrain = R*V(g,s)**2)
            push!(contrib_eval.args, c.expr)
        end
    end

    # Allocate current variables for named branches with voltage contributions
    branch_current_alloc = Expr(:block)
    branch_current_vars = Dict{Symbol, Symbol}()  # branch_name -> current_var_name
    for (branch_name, _) in voltage_branch_contribs
        I_var = Symbol("_I_branch_", branch_name, "_idx")
        alloc_name = Symbol(symname, "_I_", branch_name)
        push!(branch_current_alloc.args,
            :($I_var = CedarSim.MNA.alloc_current!(ctx, $(QuoteNode(alloc_name)))))
        branch_current_vars[branch_name] = I_var
    end

    # Generate stamping code for each unique branch - UNROLL loops at codegen time
    # Now handles both terminal nodes and internal nodes
    stamp_code = Expr(:block)
    for ((p_sym, n_sym), exprs) in branch_contribs
        # Look up node indices in combined list (terminals + internal)
        # Handle ground node (Symbol("0")) specially - it maps to integer 0
        p_idx = p_sym == Symbol("0") ? nothing : findfirst(==(p_sym), all_node_syms)
        n_idx = n_sym == Symbol("0") ? nothing : findfirst(==(n_sym), all_node_syms)
        # Ground node uses literal 0; other nodes use their parameter symbol
        p_node = p_idx === nothing ? 0 : all_node_params[p_idx]
        n_node = n_idx === nothing ? 0 : all_node_params[n_idx]

        # Sum all contributions to this branch
        sum_expr = length(exprs) == 1 ? exprs[1] : Expr(:call, :+, exprs...)

        # Generate unrolled stamping code
        # The result can be:
        # 1. Dual{Nothing, Float64, N} - pure resistive (no ddt)
        # 2. Dual{ContributionTag, Dual{Nothing}, 1} - pure reactive (only ddt terms)
        # 3. Dual{Nothing, Dual{ContributionTag}, N} - mixed resistive+reactive
        #    (when adding Dual{Nothing} + Dual{ContributionTag}, Nothing becomes outer)
        branch_stamp = quote
            # Evaluate the branch current
            I_branch = $sum_expr

            # Handle nested dual structure for mixed resistive/reactive contributions
            # See mna_ad_stamping.md and evaluate_contribution in contrib.jl
            if I_branch isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
                # Case 2: ContributionTag is outer - pure reactive (e.g., C*ddt(V))
                I_resist = ForwardDiff.value(I_branch)   # Inner dual for resistive I
                I_react = ForwardDiff.partials(I_branch, 1)  # Inner dual for charge q

                # Extract scalar value and node partials from resistive part
                I_val = ForwardDiff.value(I_resist)
                $([:($(Symbol("dI_dV", k)) = ForwardDiff.partials(I_resist, $k)) for k in 1:n_all_nodes]...)

                # Extract scalar value and node partials from reactive part (charge)
                q_val = ForwardDiff.value(I_react)
                $([:($(Symbol("dq_dV", k)) = ForwardDiff.partials(I_react, $k)) for k in 1:n_all_nodes]...)
            elseif I_branch isa ForwardDiff.Dual
                # Dual{Nothing} is outer - check if value contains ContributionTag
                _val = ForwardDiff.value(I_branch)
                if _val isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
                    # Case 3: Mixed resistive+reactive (e.g., V/R + C*ddt(V))
                    # value(I_branch) = Dual{ContributionTag}(inner_I, inner_q)
                    # partials(I_branch, k) = Dual{ContributionTag}(dI/dVk, dq/dVk)
                    inner_resist = ForwardDiff.value(_val)
                    inner_react = ForwardDiff.partials(_val, 1)

                    # Extract I value and q value from innermost level
                    I_val = inner_resist isa ForwardDiff.Dual ? ForwardDiff.value(inner_resist) : Float64(inner_resist)
                    q_val = inner_react isa ForwardDiff.Dual ? ForwardDiff.value(inner_react) : Float64(inner_react)

                    # Extract dI/dVk from partials - each partial is Dual{ContributionTag}
                    $([quote
                        _part_k = ForwardDiff.partials(I_branch, $k)
                        if _part_k isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
                            $(Symbol("dI_dV", k)) = ForwardDiff.value(ForwardDiff.value(_part_k))
                        else
                            $(Symbol("dI_dV", k)) = _part_k isa ForwardDiff.Dual ? ForwardDiff.value(_part_k) : Float64(_part_k)
                        end
                    end for k in 1:n_all_nodes]...)

                    # Extract dq/dVk from inner_react partials
                    $([:($(Symbol("dq_dV", k)) = inner_react isa ForwardDiff.Dual ? ForwardDiff.partials(inner_react, $k) : 0.0) for k in 1:n_all_nodes]...)
                else
                    # Case 1: Pure resistive (e.g., V/R)
                    I_val = _val isa ForwardDiff.Dual ? ForwardDiff.value(_val) : Float64(_val)
                    $([:($(Symbol("dI_dV", k)) = ForwardDiff.partials(I_branch, $k)) for k in 1:n_all_nodes]...)
                    q_val = 0.0
                    $([:($(Symbol("dq_dV", k)) = 0.0) for k in 1:n_all_nodes]...)
                end
            else
                # Case 0: Scalar result (e.g., Ids=0 in cutoff)
                I_val = I_branch isa Real ? Float64(I_branch) : 0.0
                $([:($(Symbol("dI_dV", k)) = 0.0) for k in 1:n_all_nodes]...)
                q_val = 0.0
                $([:($(Symbol("dq_dV", k)) = 0.0) for k in 1:n_all_nodes]...)
            end
        end

        # Stamp resistive Jacobians into G matrix
        # MNA sign convention: I(p,n) flows from p to n
        # G[p,k] = +dI/dVk (current leaving p)
        # G[n,k] = -dI/dVk (current entering n)
        for k in 1:n_all_nodes
            k_node = all_node_params[k]
            push!(branch_stamp.args, quote
                if $p_node != 0 && $k_node != 0
                    CedarSim.MNA.stamp_G!(ctx, $p_node, $k_node, $(Symbol("dI_dV", k)))
                end
                if $n_node != 0 && $k_node != 0
                    CedarSim.MNA.stamp_G!(ctx, $n_node, $k_node, -$(Symbol("dI_dV", k)))
                end
            end)
        end

        # Stamp reactive Jacobians (capacitances) into C matrix
        # Same sign convention as G matrix
        for k in 1:n_all_nodes
            k_node = all_node_params[k]
            push!(branch_stamp.args, quote
                if $p_node != 0 && $k_node != 0
                    CedarSim.MNA.stamp_C!(ctx, $p_node, $k_node, $(Symbol("dq_dV", k)))
                end
                if $n_node != 0 && $k_node != 0
                    CedarSim.MNA.stamp_C!(ctx, $n_node, $k_node, -$(Symbol("dq_dV", k)))
                end
            end)
        end

        # Stamp RHS: Ieq = I_val - sum(dI/dVk * Vk)
        # MNA sign convention: b[p] -= Ieq, b[n] += Ieq
        # (matches stamp_contribution! in contrib.jl)
        ieq_terms = Any[:I_val]
        for k in 1:n_all_nodes
            push!(ieq_terms, :(- $(Symbol("dI_dV", k)) * $(Symbol("V_", k))))
        end
        ieq_expr = Expr(:call, :+, ieq_terms...)

        # RHS stamping using Newton companion model
        # Must match the 2-term convention in contrib.jl:
        #   stamp_b!(ctx, p, -b_companion)
        #   stamp_b!(ctx, n, +b_companion)
        # where b_companion = I_val - dI/dVp*Vp - dI/dVn*Vn = Ieq
        push!(branch_stamp.args, quote
            let Ieq = $ieq_expr
                if $p_node != 0
                    CedarSim.MNA.stamp_b!(ctx, $p_node, -Ieq)
                end
                if $n_node != 0
                    CedarSim.MNA.stamp_b!(ctx, $n_node, Ieq)
                end
            end
        end)

        push!(stamp_code.args, branch_stamp)
    end

    # Generate internal node allocation code (runs once per stamp! call)
    # This allocates matrix/vector entries for internal nodes
    internal_node_alloc = Expr(:block)
    for (i, (int_sym, int_param)) in enumerate(zip(internal_nodes, internal_node_params))
        # Allocate internal node using alloc_internal_node!
        # The name is unique per device instance using module name + node name
        alloc_name = Symbol(symname, "_", int_sym)
        push!(internal_node_alloc.args,
            :($int_param = CedarSim.MNA.alloc_internal_node!(ctx, $(QuoteNode(alloc_name)))))
    end

    # Generate voltage extraction for all nodes (terminals + internal)
    # V_1..V_n_ports are for terminal nodes
    # V_(n_ports+1)..V_n_all_nodes are for internal nodes
    voltage_extraction = Expr(:block)
    # Terminal nodes (from function arguments)
    for i in 1:n_ports
        np = node_params[i]
        push!(voltage_extraction.args,
            :($(Symbol("V_", i)) = $np == 0 ? 0.0 : (isempty(x) ? 0.0 : x[$np])))
    end
    # Internal nodes (from alloc_internal_node!)
    for i in 1:n_internal
        idx = n_ports + i
        inp = internal_node_params[i]
        push!(voltage_extraction.args,
            :($(Symbol("V_", idx)) = isempty(x) || $inp > length(x) ? 0.0 : x[$inp]))
    end

    # Generate dual creation for all nodes (terminals + internal)
    # Each node gets a dual with identity partials: ∂V_i/∂V_k = δ_ik
    dual_creation = Expr(:block)
    for i in 1:n_all_nodes
        node_sym = all_node_syms[i]
        # Create dual with partials: (0,...,1,...,0) where 1 is at position i
        partials_tuple = Expr(:tuple, [k == i ? 1.0 : 0.0 for k in 1:n_all_nodes]...)
        push!(dual_creation.args,
            :($node_sym = Dual{Nothing}($(Symbol("V_", i)), $partials_tuple...)))
    end

    # Generate branch current extraction for named branches
    branch_current_extraction = Expr(:block)
    for (branch_name, I_var) in branch_current_vars
        I_sym = Symbol("_I_branch_", branch_name)
        push!(branch_current_extraction.args,
            :($I_sym = isempty(x) || $I_var > length(x) ? 0.0 : x[$I_var]))
    end

    # Generate voltage contribution stamping for named branches
    voltage_stamp_code = Expr(:block)
    for (branch_name, vc) in voltage_branch_contribs
        I_var = branch_current_vars[branch_name]
        p_sym, n_sym = vc.p, vc.n
        exprs = vc.exprs

        # Look up node indices
        p_idx = p_sym == Symbol("0") ? nothing : findfirst(==(p_sym), all_node_syms)
        n_idx = n_sym == Symbol("0") ? nothing : findfirst(==(n_sym), all_node_syms)
        p_node = p_idx === nothing ? 0 : all_node_params[p_idx]
        n_node = n_idx === nothing ? 0 : all_node_params[n_idx]

        # Sum all voltage contributions
        sum_expr = length(exprs) == 1 ? exprs[1] : Expr(:call, :+, exprs...)

        # Generate stamping code for voltage contribution
        # V(br) <+ expr means V_p - V_n = expr
        # With current variable I_br:
        # - KCL: G[p, I_br] = 1, G[n, I_br] = -1 (current flows from p to n)
        # - Voltage constraint: G[I_br, p] = 1, G[I_br, n] = -1, b[I_br] = expr
        #
        # For inductor: V = L*ddt(I) uses va_ddt which creates Dual{ContributionTag}
        # - value = resistive part
        # - partials(1) = reactive part (stamps into C[I_br, I_br])

        v_stamp = quote
            # Evaluate voltage contribution
            V_contrib = $sum_expr

            # Stamp KCL: current I flows from p to n
            if $p_node != 0
                CedarSim.MNA.stamp_G!(ctx, $p_node, $I_var, 1.0)
            end
            if $n_node != 0
                CedarSim.MNA.stamp_G!(ctx, $n_node, $I_var, -1.0)
            end

            # Voltage constraint: V_p - V_n = V_contrib
            CedarSim.MNA.stamp_G!(ctx, $I_var, $p_node, 1.0)
            CedarSim.MNA.stamp_G!(ctx, $I_var, $n_node, -1.0)

            # Handle reactive (ddt) contributions
            if V_contrib isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
                # Contains ddt() terms
                V_resist = ForwardDiff.value(V_contrib)
                V_react = ForwardDiff.partials(V_contrib, 1)

                # V_resist is the resistive voltage part (e.g., R*I)
                # V_react is the reactive coefficient (e.g., L from L*ddt(I) = L*s*I)
                V_resist_val = V_resist isa ForwardDiff.Dual ? ForwardDiff.value(V_resist) : Float64(V_resist)
                V_react_val = V_react isa ForwardDiff.Dual ? ForwardDiff.value(V_react) : Float64(V_react)

                # Stamp RHS with resistive voltage
                CedarSim.MNA.stamp_b!(ctx, $I_var, V_resist_val)

                # Stamp C matrix for reactive part: V = L*dI/dt
                # Voltage equation: V_p - V_n - L*dI/dt = 0
                # This stamps -L into C[I_var, I_var]
                CedarSim.MNA.stamp_C!(ctx, $I_var, $I_var, -V_react_val)
            else
                # Pure resistive voltage
                V_val = V_contrib isa ForwardDiff.Dual ? ForwardDiff.value(V_contrib) : Float64(V_contrib)
                CedarSim.MNA.stamp_b!(ctx, $I_var, V_val)
            end
        end

        push!(voltage_stamp_code.args, v_stamp)
    end

    # Build the stamp method
    # Terminal nodes come from function parameters; internal nodes are allocated dynamically
    quote
        function CedarSim.MNA.stamp!(dev::$symname, ctx::CedarSim.MNA.MNAContext,
                                     $([:($np::Int) for np in node_params]...);
                                     t::Real=0.0, mode::Symbol=:dcop, x::AbstractVector=Float64[],
                                     spec::CedarSim.MNA.MNASpec=CedarSim.MNA.MNASpec())
            $(params_to_locals...)
            $(function_defs...)

            # Allocate internal nodes (idempotent - returns existing index if already allocated)
            $internal_node_alloc

            # Allocate current variables for named branches with voltage contributions
            $branch_current_alloc

            # Get operating point voltages (Float64) - used for RHS linearization
            $voltage_extraction

            # Get operating point currents for named branches
            $branch_current_extraction

            # Create duals with partials for each node voltage (terminals + internal)
            # dual[i] = Dual(V_i, (k==1 ? 1 : 0), (k==2 ? 1 : 0), ...)
            $dual_creation

            # Evaluate contribution expressions with duals
            $contrib_eval

            # Stamp current contributions
            $stamp_code

            # Stamp voltage contributions for named branches (e.g., inductor V = L*dI/dt)
            $voltage_stamp_code

            return nothing
        end
    end
end

"""
    make_mna_module(va::VANode)

Generate an MNA-compatible module from a parsed Verilog-A file.
"""
function make_mna_module(va::VANode)
    vamod = va.stmts[end]
    s = Symbol(String(vamod.id), "_module")
    typename = Symbol(vamod.id)

    # Get the device definition (returns Expr(:toplevel, struct_def, constructor, stamp_method))
    device_expr = CedarSim.make_mna_device(vamod)

    Expr(:toplevel, :(baremodule $s
        using Base: AbstractVector, Real, Symbol, Float64, Int, isempty, max, zeros, zero, length
        using Base: hasproperty, getproperty, getfield, error, !==, iszero, abs
        import Base  # For getproperty override in aliasparam
        import ..CedarSim
        using ..CedarSim.VerilogAEnvironment
        using ..CedarSim.MNA: va_ddt, stamp_current_contribution!, MNAContext, MNASpec, alloc_internal_node!, alloc_current!
        using ForwardDiff: Dual, value, partials
        import ForwardDiff
        export $typename
        $(device_expr.args...)
    end), :(using .$s))
end

struct VAFile
    file::String
end
Base.String(vaf::VAFile) = vaf.file
Base.abspath(file::VAFile) = VAFile(Base.abspath(file.file))
Base.isfile(file::VAFile) = Base.isfile(file.file)
Base.isabspath(file::VAFile) = Base.isabspath(file.file)
Base.findfirst(str::String, file::VAFile) = Base.findfirst(str, file.file)
Base.joinpath(str::String, file::VAFile) = VAFile(Base.joinpath(str, file.file))
Base.normpath(file::VAFile) = VAFile(Base.normpath(file.file))
export VAFile, @va_str

function make_module(va::VANode)
    vamod = va.stmts[end]
    s = Symbol(String(vamod.id), "_module")
    Expr(:toplevel, :(baremodule $s
        using ..CedarSim.VerilogAEnvironment
        export $(Symbol(vamod.id))
        $(CedarSim.make_spice_device(vamod))
    end), :(using .$s))
end

function parse_and_eval_vafile(mod::Module, file::VAFile)
    va = VerilogAParser.parsefile(file.file)
    if va.ps.errored
        cedarthrow(LoadError(file.file, 0, VAParseError(va)))
    else
        Core.eval(mod, make_mna_module(va))
    end
    return va.ps.srcfiles
end

function Base.include(mod::Module, file::VAFile)
    parse_and_eval_vafile(mod, file)
    return nothing
end

macro va_str(str)
    va = VerilogAParser.parse(IOBuffer(str))
    if va.ps.errored
        cedarthrow(LoadError("va_str", 0, VAParseError(va)))
    else
        # Use runtime eval to handle module definitions which must be at top level.
        # QuoteNode prevents any hygiene transformations on the AST.
        expr = make_mna_module(va)
        :(Core.eval($__module__, $(QuoteNode(expr))))
    end
end

struct VAParseError
    va
end

Base.show(io::IO, vap::VAParseError) = VerilogAParser.VerilogACSTParser.visit_errors(vap.va; io)
