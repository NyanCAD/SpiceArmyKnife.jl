# MNA Backend for Verilog-A Code Generation
# Ports vasim.jl to generate MNA-compatible residual functions instead of DAECompiler equations

using VerilogAParser
using AbstractTrees
using AbstractTrees: parent, nodevalue
using VerilogAParser.VerilogACSTParser:
    ContributionStatement, AnalogSeqBlock, AnalogBlock,
    InOutDeclaration, NetDeclaration, ParameterDeclaration, AliasParameterDeclaration,
    VerilogModule, Literal, BinaryExpression, BPFC,
    IdentifierPrimary, @case, BranchDeclaration,
    AnalogFunctionDeclaration,
    IntRealDeclaration, AnalogStatement,
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

const MNA_VAT = VerilogAParser.VerilogATokenize
const MNA_VANode = VerilogAParser.VerilogACSTParser.Node

# Tag type for ForwardDiff to track ddx derivatives
struct MNASimTag end
ForwardDiff.:(≺)(::Type{<:ForwardDiff.Tag}, ::Type{MNASimTag}) = true
ForwardDiff.:(≺)(::Type{MNASimTag}, ::Type{<:ForwardDiff.Tag}) = false

# Export the tag for generated code
export MNASimTag

export @mna_va_str, MNAVAFile, mna_va_load, MNABranchKind, MNA_CURRENT, MNA_VOLTAGE

# Verilog-A compatible math functions (exported for use in generated code)
# These are referenced via fully qualified names in generated code
const pow = ^
const ln = log
const sqrt = Base.sqrt
const abs = Base.abs
const exp = Base.exp
const log10 = Base.log10
const sin = Base.sin
const cos = Base.cos
const tan = Base.tan
const asin = Base.asin
const acos = Base.acos
const atan = Base.atan
const sinh = Base.sinh
const cosh = Base.cosh
const tanh = Base.tanh
const asinh = Base.asinh
const acosh = Base.acosh
const atanh = Base.atanh
const hypot = Base.hypot
const min = Base.min
const max = Base.max
const floor = Base.floor
const ceil = Base.ceil

# Verilog-A limit function (clamping)
limexp(x) = x > 80.0 ? exp(80.0) * (1 + x - 80.0) : exp(x)  # Limited exponential to avoid overflow

# Export Verilog-A math functions for generated code
export pow, ln, limexp

#=
MNA VA Code Generation Strategy:

For a Verilog-A module like:
    module resistor(p, n);
        parameter real R = 1000;
        analog I(p,n) <+ V(p,n)/R;
    endmodule

We generate:
1. A struct holding parameters
2. A stamp! function that adds nonlinear element callbacks
3. The callback computes currents from voltages and returns residual contributions

For internal nodes, we add them as additional MNA unknowns.
=#

@enum MNABranchKind MNA_CURRENT MNA_VOLTAGE

"""
    MNAVAFunction

Holds information about a Verilog-A defined function.
"""
struct MNAVAFunction
    arg_order::Vector{Symbol}
    inout_decls::Dict{Symbol, Symbol}
end

"""
    MNAScope

Scope for MNA code generation from Verilog-A.
Similar to Scope in vasim.jl but generates MNA-compatible code.
"""
struct MNAScope
    parameters::Set{Symbol}
    node_order::Vector{Symbol}      # All nodes: external pins + internal + "0" (ground)
    ninternal_nodes::Int
    branch_order::Vector{Pair{Symbol}}
    used_branches::Set{Pair{Symbol}}
    var_types::Dict{Symbol, Union{Type{Int}, Type{Float64}}}
    all_functions::Dict{Symbol, MNAVAFunction}
    undefault_ids::Bool
    ddx_order::Vector{Symbol}       # Nodes used in ddx() calls for ForwardDiff tracking
    ddt_exprs::Vector{Any}          # Expressions inside ddt() calls for transient tracking
end

MNAScope() = MNAScope(Set{Symbol}(), Vector{Symbol}(), 0, Vector{Pair{Symbol}}(),
    Set{Pair{Symbol}}(), Dict{Symbol, Union{Type{Int}, Type{Float64}}}(),
    Dict{Symbol, MNAVAFunction}(), false, Vector{Symbol}(), Vector{Any}())

"""
    find_ddx!(ddx_order::Vector{Symbol}, va::MNA_VANode)

Scan a Verilog-A AST node for ddx() calls and collect the probe nodes.
Used to determine which nodes need ForwardDiff tracking.
"""
function find_ddx!(ddx_order::Vector{Symbol}, va::MNA_VANode)
    for stmt in AbstractTrees.PreOrderDFS(va)
        if stmt isa MNA_VANode{FunctionCall} && Symbol(stmt.id) == :ddx
            item = stmt.args[2].item
            @assert mna_formof(item) == FunctionCall
            @assert Symbol(item.id) == :V
            for arg in item.args
                name = Symbol(mna_assemble_id_string(arg.item))
                !in(name, ddx_order) && push!(ddx_order, name)
            end
        end
    end
end

"""
    find_ddt!(ddt_count::Ref{Int}, va::MNA_VANode)

Scan a Verilog-A AST node for ddt() calls and count them.
Each ddt() call gets a unique charge state index for transient simulation.
Returns the number of ddt() calls found.
"""
function find_ddt!(ddt_count::Ref{Int}, va::MNA_VANode)
    for stmt in AbstractTrees.PreOrderDFS(va)
        if stmt isa MNA_VANode{FunctionCall} && Symbol(stmt.id) == :ddt
            ddt_count[] += 1
        end
    end
end

# Scale factor mapping
const mna_sf_mapping = Dict(
    'T' => 1e12, 'G' => 1e9, 'M' => 1e6, 'K' => 1e3, 'k' => 1e3,
    'm' => 1e-3, 'u' => 1e-6, 'n' => 1e-9, 'p' => 1e-12, 'f' => 1e-15, 'a' => 1e-18,
)

mna_formof(e::MNA_VANode{S}) where {S} = S

function mna_assemble_id_string(id)
    if isa(id, Node{SystemIdentifier})
        return String(id)
    elseif isa(id, Node{Identifier})
        return join(mna_assemble_id_string(c) for c in AbstractTrees.children(id))
    elseif isa(id, Node{IdentifierPart})
        s = String(id)
        id.escaped && (s = s[2:end])
        return s
    elseif isa(id, Node{IdentifierConcatItem})
        return mna_assemble_id_string(id.id)
    else
        return String(id)
    end
end

mna_kw_to_T(kw::Kind) = kw === REAL ? Float64 : Int

# Literal parsing
function (::MNAScope)(cs::MNA_VANode{Literal})
    Meta.parse(String(cs))
end

function (::MNAScope)(cs::MNA_VANode{FloatLiteral})
    txt = String(cs)
    sf = nothing
    if !isempty(txt) && is_scale_factor(txt[end])
        sf = txt[end]
        txt = txt[1:end-1]
    end
    ret = Base.parse(Float64, txt)
    if sf !== nothing && haskey(mna_sf_mapping, sf)
        ret *= mna_sf_mapping[sf]
    end
    return ret
end

# Contribution statement - the heart of VA device modeling
# I(p,n) <+ expr  means: add current 'expr' flowing from p to n
# V(p,n) <+ expr  means: set voltage V(p) - V(n) = expr (requires branch variable)
function (to_julia::MNAScope)(cs::MNA_VANode{ContributionStatement})
    bpfc = cs.lvalue
    kind_sym = Symbol(bpfc.id)

    # Use fully qualified names for macro hygiene
    if kind_sym == :I
        kind_expr = :(CedarSim.MNA_CURRENT)
    elseif kind_sym == :V
        kind_expr = :(CedarSim.MNA_VOLTAGE)
    else
        return :(error("Unknown branch contribution kind: $kind_sym"))
    end

    refs = map(bpfc.references) do ref
        Symbol(mna_assemble_id_string(ref.item))
    end

    if length(refs) == 1
        # Single node reference: I(node) or V(node) - relative to ground
        node = refs[1]
        svar = Symbol("branch_state_", node, "_0")
        eqvar = Symbol("branch_value_", node, "_0")
        push!(to_julia.used_branches, node => Symbol("0"))
        return quote
            if $svar != $kind_expr
                $eqvar = 0.0
                $svar = $kind_expr
            end
            # Use ForwardDiff.value to extract scalar from Dual (for ddx support)
            $eqvar += $(ForwardDiff.value)($(CedarSim.MNASimTag), $(to_julia(cs.assign_expr)))
        end
    elseif length(refs) == 2
        # Two node reference: I(p,n) or V(p,n)
        (id1, id2) = refs

        idx = findfirst(to_julia.branch_order) do branch
            branch == (id1 => id2) || branch == (id2 => id1)
        end
        @assert idx !== nothing "Branch ($id1, $id2) not found in branch_order"
        branch = to_julia.branch_order[idx]
        push!(to_julia.used_branches, branch)
        reversed = branch == (id2 => id1)

        s = gensym()
        svar = Symbol("branch_state_", branch[1], "_", branch[2])
        eqvar = Symbol("branch_value_", branch[1], "_", branch[2])

        return quote
            if $svar != $kind_expr
                $eqvar = 0.0
                $svar = $kind_expr
            end
            # Use ForwardDiff.value to extract scalar from Dual (for ddx support)
            $s = $(ForwardDiff.value)($(CedarSim.MNASimTag), $(to_julia(cs.assign_expr)))
            $eqvar += $(reversed ? :(-$s) : s)
        end
    end
end

# Identifier primary (variable/parameter reference)
function (scope::MNAScope)(ip::MNA_VANode{IdentifierPrimary})
    id = Symbol(mna_assemble_id_string(ip.id))
    if scope.undefault_ids
        id = Expr(:call, :mna_undefault, id)
    end
    id
end

# System identifier ($temperature, etc.)
function (scope::MNAScope)(ip::MNA_VANode{SystemIdentifier})
    id = Symbol(ip)
    if id == Symbol("\$temperature")
        return :(_temperature)
    else
        return id
    end
end

# Binary expression
function (to_julia::MNAScope)(cs::MNA_VANode{BinaryExpression})
    op = Symbol(cs.op)
    if op == :(||)
        return Expr(:call, (|), to_julia(cs.lhs), to_julia(cs.rhs))
    elseif op == :(&&)
        return Expr(:call, (&), to_julia(cs.lhs), to_julia(cs.rhs))
    else
        return Expr(:call, op, to_julia(cs.lhs), to_julia(cs.rhs))
    end
end

# Sequential block (begin...end)
function (to_julia::MNAScope)(asb::MNA_VANode{AnalogSeqBlock})
    if asb.decl !== nothing
        block_var_types = copy(to_julia.var_types)
        for decl in asb.decl.decls
            item = decl.item
            @case mna_formof(item) begin
                IntRealDeclaration => begin
                    T = mna_kw_to_T(item.kw.kw)
                    for name in item.idents
                        block_var_types[Symbol(name.item)] = T
                    end
                end
            end
        end
        to_julia_block = MNAScope(to_julia.parameters, to_julia.node_order,
            to_julia.ninternal_nodes, to_julia.branch_order, to_julia.used_branches,
            block_var_types, to_julia.all_functions, to_julia.undefault_ids, to_julia.ddx_order,
            to_julia.ddt_exprs)
    else
        to_julia_block = to_julia
    end

    ret = Expr(:block)
    for stmt in asb.stmts
        push!(ret.args, to_julia_block(stmt))
    end
    ret
end

# Analog statement wrapper
(to_julia::MNAScope)(stmt::MNA_VANode{AnalogStatement}) = to_julia(stmt.stmt)

# Conditional block (if/else)
function (to_julia::MNAScope)(stmt::MNA_VANode{AnalogConditionalBlock})
    aif = stmt.aif
    function if_body_to_julia(ifstmt)
        if mna_formof(ifstmt) == AnalogSeqBlock
            return to_julia(ifstmt)
        else
            return Expr(:block, to_julia(ifstmt))
        end
    end

    ifex = ex = Expr(:if, to_julia(aif.condition), if_body_to_julia(aif.stmt))
    for case in stmt.elsecases
        if mna_formof(case.stmt) == AnalogIf
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

# For loop
function (to_julia::MNAScope)(stmt::MNA_VANode{AnalogFor})
    body = to_julia(stmt.stmt)
    push!(body.args, to_julia(stmt.update_stmt))
    while_expr = Expr(:while, to_julia(stmt.cond_expr), body)
    Expr(:block, to_julia(stmt.init_stmt), while_expr)
end

# While loop
function (to_julia::MNAScope)(stmt::MNA_VANode{AnalogWhile})
    body = to_julia(stmt.stmt)
    Expr(:while, to_julia(stmt.cond_expr), body)
end

# Repeat loop
function (to_julia::MNAScope)(stmt::MNA_VANode{AnalogRepeat})
    body = to_julia(stmt.stmt)
    Expr(:for, :(_ = 1:$(stmt.num_repeat)), body)
end

# Unary operator
function (to_julia::MNAScope)(stmt::MNA_VANode{UnaryOp})
    op = Symbol(stmt.op)
    operand = to_julia(stmt.operand)
    # In Verilog-A, ! (logical not) works on integers where 0 is false, non-zero is true
    # Julia's ! only works on Bool, so we need to convert
    if op == :!
        return Expr(:call, :!, Expr(:call, :!=, operand, 0))
    elseif op == :~
        # Bitwise not - need to use Int conversion
        return Expr(:call, :~, Expr(:call, :Int, operand))
    else
        return Expr(:call, op, operand)
    end
end

# Function call
function (to_julia::MNAScope)(stmt::MNA_VANode{FunctionCall})
    fname = Symbol(stmt.id)

    # $param_given check
    if fname == Symbol("\$param_given")
        id = Symbol(stmt.args[1].item)
        return Expr(:call, :(!), Expr(:call, :mna_isdefault,
            Expr(:call, :getfield, :_self, QuoteNode(id))))
    end

    # V() probe - returns voltage
    # If the node is in ddx_order, wrap in ForwardDiff Dual for derivative tracking
    if fname == :V
        # Helper to wrap a node voltage in Dual if it's in ddx_order
        function Vref(id)
            id_idx = findfirst(==(id), to_julia.ddx_order)
            if id_idx !== nothing
                # Wrap in Dual with partial = 1 at this node's position
                n_ddx = length(to_julia.ddx_order)
                partials = ntuple(i -> i == id_idx ? 1.0 : 0.0, n_ddx)
                return :($(ForwardDiff.Dual){$(CedarSim.MNASimTag)}($id, $(partials...)))
            else
                return id
            end
        end

        @assert length(stmt.args) in (1, 2)
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

    # I() probe - returns current (not commonly used in MNA residual computation)
    elseif fname == :I
        @assert length(stmt.args) == 2
        id1 = Symbol(stmt.args[1].item)
        id2 = Symbol(stmt.args[2].item)
        # For MNA, branch currents are typically computed, not probed
        # Return a placeholder - this needs special handling for voltage sources
        return :(error("I() probe not yet supported in MNA backend"))

    # ddt() - time derivative (for transient)
    # Companion model approach (like ngspice/OpenVAF):
    # ddt(Q) ≈ alpha * (Q - Q_prev) where alpha = 1/dt (Backward Euler)
    # This returns the current contribution from the charge dynamics
    elseif fname == :ddt
        expr_code = to_julia(stmt.args[1].item)

        # Assign charge index (1-based, incremented for each ddt call)
        push!(to_julia.ddt_exprs, expr_code)
        charge_idx = length(to_julia.ddt_exprs)

        # Generated code:
        # - Evaluate Q expression
        # - Store Q value for later history update
        # - Return alpha * (Q - Q_prev) as the companion model current
        # - For DC: alpha = 0, so ddt = 0
        # - For transient: ddt ≈ (Q - Q_prev) / dt
        return quote
            let _q_expr = $expr_code
                # Extract scalar value
                _Q_new = $(ForwardDiff.value)($(CedarSim.MNASimTag), _q_expr)
                _react_charges[$charge_idx] = _Q_new
                # Get previous charge value (0 for first step)
                _Q_prev = _charge_prev[$charge_idx]
                # Companion model: I = alpha * (Q - Q_prev)
                _alpha * (_Q_new - _Q_prev)
            end
        end

    # ddx() - derivative with respect to a node voltage
    # ddx(expr, V(node)) returns d(expr)/d(V(node))
    elseif fname == :ddx
        item = stmt.args[2].item
        @assert mna_formof(item) == FunctionCall
        @assert Symbol(item.id) == :V

        if length(item.args) == 1
            # ddx(expr, V(node)) - single node
            probe = Symbol(mna_assemble_id_string(item.args[1].item))
            id_idx = findfirst(==(probe), to_julia.ddx_order)
            expr_code = to_julia(stmt.args[1].item)
            return :(let _x = $expr_code
                isa(_x, $(ForwardDiff.Dual)) ?
                    (@inbounds $(ForwardDiff.partials)($(CedarSim.MNASimTag), _x, $id_idx)) : 0.0
            end)
        else
            # ddx(expr, V(node1, node2)) - differential voltage
            probe1 = Symbol(mna_assemble_id_string(item.args[1].item))
            probe2 = Symbol(mna_assemble_id_string(item.args[2].item))
            id1_idx = findfirst(==(probe1), to_julia.ddx_order)
            id2_idx = findfirst(==(probe2), to_julia.ddx_order)
            expr_code = to_julia(stmt.args[1].item)
            return :(let _x = $expr_code,
                        _dx1 = isa(_x, $(ForwardDiff.Dual)) ?
                            (@inbounds $(ForwardDiff.partials)($(CedarSim.MNASimTag), _x, $id1_idx)) : 0.0,
                        _dx2 = isa(_x, $(ForwardDiff.Dual)) ?
                            (@inbounds $(ForwardDiff.partials)($(CedarSim.MNASimTag), _x, $id2_idx)) : 0.0
                (_dx1 - _dx2) / 2
            end)
        end

    # Noise functions - ignored for DC
    elseif fname ∈ (:white_noise, :flicker_noise)
        return 0.0
    end

    # Check for VA-defined function
    vaf = get(to_julia.all_functions, fname, nothing)
    if vaf !== nothing
        args = map(x->to_julia(x.item), stmt.args)
        in_args = Any[]
        out_args = Any[]

        if length(args) != length(vaf.arg_order)
            return Expr(:call, :error, "Wrong number of arguments to function $fname")
        end

        for (arg, vaarg) in zip(args, vaf.arg_order)
            kind = vaf.inout_decls[vaarg]
            if kind == :output || kind == :inout
                isa(arg, Symbol) || return Expr(:call, :error, "Output argument must be a symbol")
                push!(out_args, arg)
            end
            if kind == :input || kind == :inout
                push!(in_args, arg)
            end
        end
        ret = Expr(:call, fname, in_args...)
        if length(out_args) != 0
            s = gensym()
            ret = quote
                ($s, ($(out_args...),)) = $ret
                $s
            end
        end
        return ret
    end

    # Standard math functions - map to CedarSim qualified names
    # This ensures the functions are found when code is eval'd in Main
    va_math_functions = Dict(
        :pow => :(CedarSim.pow),
        :ln => :(CedarSim.ln),
        :limexp => :(CedarSim.limexp),
        :sqrt => :sqrt,
        :abs => :abs,
        :exp => :exp,
        :log => :log,
        :log10 => :log10,
        :sin => :sin,
        :cos => :cos,
        :tan => :tan,
        :asin => :asin,
        :acos => :acos,
        :atan => :atan,
        :atan2 => :atan,  # Julia's atan(y, x)
        :sinh => :sinh,
        :cosh => :cosh,
        :tanh => :tanh,
        :asinh => :asinh,
        :acosh => :acosh,
        :atanh => :atanh,
        :hypot => :hypot,
        :min => :min,
        :max => :max,
        :floor => :floor,
        :ceil => :ceil,
    )

    qualified_fname = get(va_math_functions, fname, fname)
    return Expr(:call, qualified_fname, map(x->to_julia(x.item), stmt.args)...)
end

# Variable assignment
function (to_julia::MNAScope)(stmt::MNA_VANode{AnalogProceduralAssignment})
    return to_julia(stmt.assign)
end

function (to_julia::MNAScope)(stmt::MNA_VANode{AnalogVariableAssignment})
    assignee = Symbol(stmt.lvalue)
    varT = get(to_julia.var_types, assignee, Float64)

    eq = stmt.eq.op
    op = eq == MNA_VAT.EQ             ?  :(=) :
         eq == MNA_VAT.STAR_STAR_EQ   ?  :(=) :
         eq == MNA_VAT.PLUS_EQ        ? :(+=) :
         eq == MNA_VAT.MINUS_EQ       ? :(-=) :
         eq == MNA_VAT.STAR_EQ        ? :(*=) :
         eq == MNA_VAT.SLASH_EQ       ? :(/=) :
         eq == MNA_VAT.PERCENT_EQ     ? :(%=) :
         eq == MNA_VAT.AND_EQ         ? :(&=) :
         eq == MNA_VAT.OR_EQ          ? :(|=) :
         eq == MNA_VAT.XOR_EQ         ? :(^=) :
         error("Unsupported assignment operator: $eq")

    req = Expr(op, assignee, Expr(:call, :mna_vaconvert, varT, to_julia(stmt.rvalue)))

    if eq == MNA_VAT.STAR_STAR_EQ
        req = Expr(op, assignee, Expr(:call, :^, assignee, to_julia(stmt.rvalue)))
    end

    return req
end

# Parentheses
function (to_julia::MNAScope)(stmt::MNA_VANode{Parens})
    return to_julia(stmt.inner)
end

# Ternary expression
function (to_julia::MNAScope)(cs::MNA_VANode{TernaryExpr})
    return Expr(:if, to_julia(cs.condition), to_julia(cs.ifcase), to_julia(cs.elsecase))
end

# Function declaration
function (to_julia::MNAScope)(fd::MNA_VANode{AnalogFunctionDeclaration})
    type_decls = Dict{Symbol, Any}()
    inout_decls = Dict{Symbol, Symbol}()
    fname = Symbol(fd.id)
    var_types = Dict{Symbol, Union{Type{Int}, Type{Float64}}}()

    rt = fd.fty === nothing ? Float64 : mna_kw_to_T(fd.fty.kw)
    var_types[fname] = rt
    arg_order = Symbol[]

    for decl in fd.items
        item = decl.item
        @case mna_formof(item) begin
            InOutDeclaration => begin
                kind = item.kw.kw === INPUT ?  :input :
                       item.kw.kw === OUTPUT ? :output :
                                               :inout
                for name in item.portnames
                    ns = Symbol(name.item)
                    inout_decls[ns] = kind
                    push!(arg_order, ns)
                end
            end
            IntRealDeclaration => begin
                T = mna_kw_to_T(item.kw.kw)
                for name in item.idents
                    var_types[Symbol(name.item)] = T
                end
            end
        end
    end

    to_julia_internal = MNAScope(to_julia.parameters, to_julia.node_order,
        to_julia.ninternal_nodes, to_julia.branch_order, to_julia.used_branches,
        var_types, to_julia.all_functions, to_julia.undefault_ids, to_julia.ddx_order,
        to_julia.ddt_exprs)

    out_args = [k for k in arg_order if inout_decls[k] in (:output, :inout)]
    in_args = [k for k in arg_order if inout_decls[k] in (:input, :inout)]
    rt_decl = length(out_args) == 0 ? fname : :(($fname, ($(out_args...),)))

    to_julia.all_functions[fname] = MNAVAFunction(arg_order, inout_decls)

    localize_vars = Any[]
    for var in keys(var_types)
        var in arg_order && continue
        push!(localize_vars, :(local $var))
    end

    return quote
        @inline function $fname($(in_args...))
            $(localize_vars...)
            local $fname = mna_vaconvert($rt, 0)
            $(to_julia_internal(fd.stmt))
            return $rt_decl
        end
    end
end

# String literal
function (to_julia::MNAScope)(stmt::MNA_VANode{StringLiteral})
    return String(stmt)[2:end-1]
end

# System task enable ($ln, $exp, etc.)
const mna_systemtaskenablemap = Dict{Symbol, Function}(
    Symbol("\$ln")=>Base.log, Symbol("\$log10")=>log10, Symbol("\$exp")=>exp, Symbol("\$sqrt")=>sqrt,
    Symbol("\$sin")=>sin, Symbol("\$cos")=>cos, Symbol("\$tan")=>tan,
    Symbol("\$asin")=>asin, Symbol("\$acos")=>acos, Symbol("\$atan")=>atan, Symbol("\$atan2")=>atan,
    Symbol("\$sinh")=>sinh, Symbol("\$cosh")=>cosh, Symbol("\$tanh")=>tanh,
    Symbol("\$asinh")=>asinh, Symbol("\$acosh")=>acosh, Symbol("\$atanh")=>atanh,
    Symbol("\$abs")=>abs, Symbol("\$min")=>min, Symbol("\$max")=>max,
    Symbol("\$floor")=>floor, Symbol("\$ceil")=>ceil,
)

function (to_julia::MNAScope)(stmt::MNA_VANode{AnalogSystemTaskEnable})
    if mna_formof(stmt.task) == FunctionCall
        fc = stmt.task
        fname = Symbol(fc.id)
        args = map(x->to_julia(x.item), fc.args)
        if fname in keys(mna_systemtaskenablemap)
            return Expr(:call, mna_systemtaskenablemap[fname], args...)
        elseif fname in (Symbol("\$strobe"), Symbol("\$warning"), Symbol("\$display"))
            return nothing  # Ignore print statements for now
        else
            return :(error("Verilog task unimplemented: $fname"))
        end
    else
        return :(error("Unknown system task"))
    end
end

# Case statement
function (to_julia::MNAScope)(stmt::MNA_VANode{CaseStatement})
    s = gensym()
    first = true
    expr = nothing
    default_case = nothing
    for case in stmt.cases
        if isa(case.conds, Node)
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
    default_case !== nothing && expr !== nothing && push!(expr.args, default_case)
    expr === nothing ? Expr(:block) : Expr(:block, :($s = $(to_julia(stmt.switch))), expr)
end

# Attributes
function (to_julia::MNAScope)(stmt::MNA_VANode{Attributes})
    res = Dict{Symbol, Any}()
    for attr in stmt.specs
        item = attr.item
        res[Symbol(item.name)] = to_julia(item.val)
    end
    return res
end

# Get pins from module
function mna_pins(vm::MNA_VANode{VerilogModule})
    plist = vm.port_list
    plist === nothing && return Symbol[]
    mapreduce(vcat, plist.ports; init=Symbol[]) do port_decl
        [Symbol(port_decl.item)]
    end
end

#=
Main code generation function
=#

"""
    make_mna_va_device(vm::MNA_VANode{VerilogModule})

Generate MNA-compatible Julia code from a Verilog-A module.
Returns an Expr that defines a struct and stamp!/residual functions.
"""
function make_mna_va_device(vm::MNA_VANode{VerilogModule})
    ps = mna_pins(vm)
    modname = String(vm.id)
    symname = Symbol(modname)

    # Collect module information
    struct_fields = Any[]
    parameter_names = Set{Symbol}()
    internal_nodes = Vector{Symbol}()
    var_types = Dict{Symbol, Union{Type{Int}, Type{Float64}}}()
    aliases = Dict{Symbol, Symbol}()

    # Scan for ddx() calls to determine which nodes need derivative tracking
    ddx_order = Vector{Symbol}()
    find_ddx!(ddx_order, vm)

    # ddt_exprs will be populated during code generation
    ddt_exprs = Vector{Any}()

    # Create scope for default expression evaluation
    to_julia_defaults = MNAScope(Set{Symbol}(), Vector{Symbol}(), 0,
        Vector{Pair{Symbol}}(), Set{Pair{Symbol}}(),
        Dict{Symbol, Union{Type{Int}, Type{Float64}}}(),
        Dict{Symbol, MNAVAFunction}(), true, ddx_order, Vector{Any}())

    # First pass: collect declarations
    for child in vm.items
        item = child.item
        @case mna_formof(item) begin
            InOutDeclaration => nothing
            NetDeclaration => begin
                for net in item.net_names
                    id = Symbol(mna_assemble_id_string(net.item))
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
                        pT = mna_kw_to_T(item.ptype.kw)
                    end
                    paramname = String(mna_assemble_id_string(param.id))
                    paramsym = Symbol(paramname)
                    push!(parameter_names, paramsym)

                    # Get default value
                    default_val = to_julia_defaults(param.default_expr)
                    push!(struct_fields, :($paramsym::$pT = $default_val))
                    var_types[paramsym] = pT
                end
            end
            AliasParameterDeclaration => begin
                paramsym = Symbol(mna_assemble_id_string(item.id))
                targetsym = Symbol(mna_assemble_id_string(item.value))
                push!(parameter_names, paramsym)
                aliases[paramsym] = targetsym
            end
            IntRealDeclaration => begin
                T = mna_kw_to_T(item.kw.kw)
                for ident in item.idents
                    var_types[Symbol(String(ident.item))] = T
                end
            end
        end
    end

    # Create scope for analog block translation
    node_order = [ps; internal_nodes; Symbol("0")]
    to_julia = MNAScope(parameter_names, node_order, length(internal_nodes),
        collect(map(x->Pair(x...), combinations(node_order, 2))),
        Set{Pair{Symbol}}(),
        var_types,
        Dict{Symbol, MNAVAFunction}(), false, ddx_order, ddt_exprs)

    # Second pass: translate analog block and function declarations
    analog_code = Expr(:block)
    func_defs = Expr(:block)

    for child in vm.items
        item = child.item
        @case mna_formof(item) begin
            AnalogFunctionDeclaration => begin
                push!(func_defs.args, to_julia(item))
            end
            AnalogBlock => begin
                push!(analog_code.args, to_julia(item.stmt))
            end
            _ => nothing
        end
    end

    # Determine which branches are used
    all_branch_order = filter(branch -> branch in to_julia.used_branches, to_julia.branch_order)
    n_branches = length(all_branch_order)

    # Number of charge state variables (for ddt() calls)
    n_charges = length(to_julia.ddt_exprs)

    # Generate branch state/value initialization
    branch_init = map(all_branch_order) do (a, b)
        svar = Symbol("branch_state_", a, "_", b)
        eqvar = Symbol("branch_value_", a, "_", b)
        quote
            $svar = CedarSim.MNA_CURRENT
            $eqvar = 0.0
        end
    end

    # Generate branch name symbols for struct storage
    branch_syms = [Symbol("branch_", a, "_", b) for (a, b) in all_branch_order]

    # Generate node voltage assignments from solution vector
    # External pins get voltage from connected MNA nodes
    # Internal nodes get voltage from additional MNA unknowns
    voltage_assigns = Any[]
    for (i, p) in enumerate(ps)
        push!(voltage_assigns, :($p = _pin_voltages[$i]))
    end
    for (i, inode) in enumerate(internal_nodes)
        push!(voltage_assigns, :($inode = _internal_voltages[$i]))
    end

    # Generate parameter assignments from struct
    param_assigns = map(collect(parameter_names)) do id
        :($id = _self.$id)
    end

    # Generate local variable initializations (excluding parameters which are already set)
    var_inits = Any[]
    for (name, T) in var_types
        # Skip if this is a parameter (already assigned from struct)
        if name in parameter_names
            continue
        end
        if T == Float64
            push!(var_inits, :($name = 0.0))
        elseif T == Int
            push!(var_inits, :($name = 0))
        else
            push!(var_inits, :($name = zero($T)))
        end
    end

    # Generate current contributions from branch values
    # For each branch (a,b) with current contribution:
    #   - Current INTO node a = -I(a,b)
    #   - Current INTO node b = +I(a,b)
    current_contributions = Any[]
    for (a, b) in all_branch_order
        eqvar = Symbol("branch_value_", a, "_", b)
        svar = Symbol("branch_state_", a, "_", b)

        # Find node indices
        a_idx = findfirst(==(a), node_order)
        b_idx = findfirst(==(b), node_order)

        # Branch index in the device's _branch_indices array (1-based)
        branch_local_idx = findfirst(==(a => b), all_branch_order)

        push!(current_contributions, quote
            if $svar == CedarSim.MNA_CURRENT
                # Current contribution: I(a,b) flows from a to b
                # Current INTO a = -I(a,b), Current INTO b = +I(a,b)
                if $a_idx <= length(_pin_voltages)
                    _pin_currents[$a_idx] -= $eqvar
                elseif $a_idx <= length(_pin_voltages) + length(_internal_voltages)
                    _internal_currents[$a_idx - length(_pin_voltages)] -= $eqvar
                end
                if $(b != Symbol("0"))
                    if $b_idx <= length(_pin_voltages)
                        _pin_currents[$b_idx] += $eqvar
                    elseif $b_idx <= length(_pin_voltages) + length(_internal_voltages)
                        _internal_currents[$b_idx - length(_pin_voltages)] += $eqvar
                    end
                end
            else
                # Voltage contribution: V(a,b) = branch_value
                # The branch current I_br flows through this voltage constraint
                # For MNA: G[a, br] = +1, G[b, br] = -1 in the stamped formulation
                # Since F .-= residual, we need opposite signs to get same effect:
                # residual[a] = -I_br gives F[a] = ... - (-I_br) = ... + I_br
                # residual[b] = +I_br gives F[b] = ... - (+I_br) = ... - I_br
                _branch_idx = circuit.num_nodes + _self._branch_indices[$branch_local_idx]
                _I_br = x[_branch_idx]

                # Add branch current to KCL (note: signs are negated because F .-= residual)
                if $a_idx <= length(_pin_voltages)
                    _pin_currents[$a_idx] -= _I_br
                elseif $a_idx <= length(_pin_voltages) + length(_internal_voltages)
                    _internal_currents[$a_idx - length(_pin_voltages)] -= _I_br
                end
                if $(b != Symbol("0"))
                    if $b_idx <= length(_pin_voltages)
                        _pin_currents[$b_idx] += _I_br
                    elseif $b_idx <= length(_pin_voltages) + length(_internal_voltages)
                        _internal_currents[$b_idx - length(_pin_voltages)] += _I_br
                    end
                end

                # Voltage constraint goes into _branch_residuals
                # Since F .-= residual, we need residual = -(V_a - V_b - V_source)
                # so that F = -(-(V_a - V_b - V_source)) = V_a - V_b - V_source = 0 at solution
                _V_a = $a_idx <= length(_pin_voltages) ? _pin_voltages[$a_idx] :
                       _internal_voltages[$a_idx - length(_pin_voltages)]
                _V_b = $(b == Symbol("0")) ? 0.0 :
                       ($b_idx <= length(_pin_voltages) ? _pin_voltages[$b_idx] :
                        _internal_voltages[$b_idx - length(_pin_voltages)])
                _branch_residuals[$branch_local_idx] = $eqvar - (_V_a - _V_b)
            end
        end)
    end

    # Generate the full device code
    n_internal = length(internal_nodes)
    n_pins = length(ps)

    quote
        # Helper functions for VA compatibility
        # Handle ForwardDiff Dual numbers by extracting value first
        mna_vaconvert(::Type{Float64}, x::$(ForwardDiff.Dual)) = Float64($(ForwardDiff.value)(x))
        mna_vaconvert(::Type{Float64}, x) = Float64(x)
        mna_vaconvert(::Type{Int}, x::$(ForwardDiff.Dual)) = Int(round($(ForwardDiff.value)(x)))
        mna_vaconvert(::Type{Int}, x) = Int(round(x))
        mna_undefault(x) = x  # For now, just return value
        mna_isdefault(x) = false  # For now, assume not default
        mna_ddt(x) = 0.0  # For DC analysis, ddt = 0

        # Device struct - use fully qualified types for macro hygiene
        Base.@kwdef mutable struct $symname <: CedarSim.MNADevice
            # MNA connection info (filled in by constructor)
            _nets::Vector{CedarSim.MNANet} = CedarSim.MNANet[]
            _internal_net_indices::Vector{Int} = Int[]
            _branch_indices::Vector{Int} = Int[]  # Branch current indices for voltage contributions
            _charge_indices::Vector{Int} = Int[]  # Charge state indices for ddt() calls
            _name::Symbol = $(QuoteNode(symname))
            _n_charges::Int = $n_charges  # Number of charge state variables
            # Parameters
            $(struct_fields...)
        end

        # Constructor that connects to MNA circuit
        function $symname(circuit::CedarSim.MNACircuit, $(ps...); name::Symbol=$(QuoteNode(symname)), kwargs...)
            # Get/create nets for external pins
            nets = [CedarSim.get_net!(circuit, n) for n in [$(ps...)]]

            # Create internal nodes
            internal_indices = Int[]
            for i in 1:$n_internal
                internal_name = Symbol(name, :_, :internal, i)
                internal_net = CedarSim.get_net!(circuit, internal_name)
                push!(internal_indices, internal_net.index)
            end

            # Create branch current variables for voltage contributions
            # Store branch local indices (not full indices) - full index = num_nodes + local_index
            branch_indices = Int[]
            branch_names = $(Expr(:vect, [QuoteNode(s) for s in branch_syms]...))
            for bname in branch_names
                branch = CedarSim.get_branch!(circuit, Symbol(name, :_, bname))
                push!(branch_indices, branch.index)  # Store local index only
            end

            # Create charge state variables for ddt() calls
            charge_indices = Int[]
            for i in 1:$n_charges
                charge = CedarSim.get_charge!(circuit, Symbol(name, :_Q, i))
                push!(charge_indices, charge.index)
            end

            $symname(_nets=nets, _internal_net_indices=internal_indices,
                     _branch_indices=branch_indices, _charge_indices=charge_indices,
                     _name=name, _n_charges=$n_charges; kwargs...)
        end

        # Stamp function - adds device to MNA system
        function CedarSim.stamp!(device::$symname, circuit::CedarSim.MNACircuit)
            # Add as nonlinear element
            push!(circuit.nonlinear_elements, (x, params, circ) -> _mna_residual(device, x, circ))
        end

        # Residual function - computes resist (I) and react (Q) contributions separately
        # Returns: (residual, jacobian_entries, react_charges, charge_indices)
        # OpenVAF approach: resist + alpha * react where alpha is integration coefficient
        function _mna_residual(_self::$symname, x::Vector{Float64}, circuit::CedarSim.MNACircuit)
            _mna_n_size = length(x)

            # Get pin voltages from solution vector
            _pin_voltages = Float64[]
            for net in _self._nets
                idx = net.index
                push!(_pin_voltages, idx > 0 ? x[idx] : 0.0)
            end

            # Get internal node voltages
            _internal_voltages = Float64[x[i] for i in _self._internal_net_indices]

            # Initialize current accumulators (resistive contributions)
            _pin_currents = zeros($n_pins)
            _internal_currents = zeros($n_internal)
            _branch_residuals = zeros($n_branches)  # For voltage contributions

            # Initialize reactive charge storage
            # _react_charges[i] stores Q from ddt(Q) expressions (current step)
            # _charge_prev[i] stores Q from previous timestep (for companion model)
            _react_charges = zeros($n_charges)
            _charge_prev = zeros($n_charges)

            # Get previous charge values from circuit storage
            for (i, charge_idx) in enumerate(_self._charge_indices)
                if charge_idx <= length(circuit.charge_values)
                    _charge_prev[i] = circuit.charge_values[charge_idx]
                end
            end

            # Integration coefficient alpha (companion model):
            # - DC: alpha = 0 (ddt terms don't contribute)
            # - Transient: alpha = 1/dt (Backward Euler)
            # ddt(Q) returns alpha * (Q - Q_prev)
            _alpha = 0.0
            if circuit.mode == :tran
                _alpha = get(circuit.params, :alpha, 0.0)
            end

            # Temperature (Kelvin)
            _temperature = circuit.temp + 273.15

            # Local function definitions
            $func_defs

            # Assign voltages to node variables
            $(voltage_assigns...)

            # Assign parameters
            $(param_assigns...)

            # Initialize local variables
            $(var_inits...)

            # Initialize branch state/values
            $(branch_init...)

            # Execute analog block (computes branch values)
            # ddt(Q) calls store Q in _react_charges and contribute alpha*Q to current
            $analog_code

            # Convert branch values to current contributions
            $(current_contributions...)

            # Build residual vector (includes both resist and alpha*react terms)
            residual = zeros(_mna_n_size)
            for (i, net) in enumerate(_self._nets)
                idx = net.index
                if idx > 0
                    residual[idx] = _pin_currents[i]
                end
            end
            for (i, idx) in enumerate(_self._internal_net_indices)
                residual[idx] = _internal_currents[i]
            end
            # Add branch residuals (voltage constraints)
            for (i, local_idx) in enumerate(_self._branch_indices)
                residual[circuit.num_nodes + local_idx] = _branch_residuals[i]
            end

            # Return residual and reactive charges (Q values from ddt expressions)
            # The ODE solver uses Q for capacitance computation: C = dQ/dV
            return (residual, Tuple{Int,Int,Float64}[], _react_charges, _self._charge_indices)
        end

        # Convenience function to add device to circuit
        function $(Symbol(lowercase(modname), "!"))(circuit::CedarSim.MNACircuit, $(ps...); kwargs...)
            dev = $symname(circuit, $(ps...); kwargs...)
            CedarSim.stamp!(dev, circuit)
            return dev
        end
    end
end

"""
    @mna_va_str(str)

String macro for creating MNA-compatible devices from Verilog-A code.

# Example
```julia
mna_va\"\"\"
module resistor(p, n);
    parameter real R = 1000.0;
    electrical p, n;
    analog I(p,n) <+ V(p,n)/R;
endmodule
\"\"\"
```
"""
macro mna_va_str(str)
    va = VerilogAParser.parse(IOBuffer(str))
    if va.ps.errored
        error("Verilog-A parse error")
    end

    vamod = va.stmts[end]
    code = make_mna_va_device(vamod)
    esc(code)
end

"""
    MNAVAFile

Wrapper for loading Verilog-A files for MNA simulation.
"""
struct MNAVAFile
    file::String
end

"""
    mna_va_load(mod::Module, file::MNAVAFile)

Load a Verilog-A file and generate MNA-compatible device code.
"""
function mna_va_load(mod::Module, file::MNAVAFile)
    va = VerilogAParser.parsefile(file.file)
    if va.ps.errored
        error("Verilog-A parse error in $(file.file)")
    end

    vamod = va.stmts[end]
    code = make_mna_va_device(vamod)
    Core.eval(mod, code)
end

"""
    mna_va_load(path::String)

Convenience function to load a Verilog-A file from a path string.
Returns the generated device type.
"""
function mna_va_load(path::String)
    mna_va_load(Main, MNAVAFile(path))
end

"""
    mna_va_load(mod::Module, path::String)

Load a Verilog-A file into the specified module.
"""
function mna_va_load(mod::Module, path::String)
    mna_va_load(mod, MNAVAFile(path))
end
