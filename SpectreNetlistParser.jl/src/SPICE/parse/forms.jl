import ...AbstractTerminal, ...isunit, ...istitle, ...iserror
abstract type Terminal <: AbstractTerminal end
EXPRS.allchildren(::Terminal) = ()
struct Keyword <: Terminal
    kw::Kind
end
struct Operator <: Terminal
    op::Kind
end
struct Identifier <: Terminal end
struct SystemIdentifier <: Terminal; end
struct Notation <: Terminal; end
struct BuiltinFunc <: Terminal; end
struct BuiltinConst <: Terminal; end
struct Literal <: Terminal; end
struct StringLiteral <: Terminal; end
struct NumberLiteral <: Terminal; end
struct JuliaEscapeBody <: Terminal; end

@enum(ErrorKind,
    UnexpectedToken,
    StrictModeViolation,
)

abstract type AbstractASTNode end
abstract type AbstractBlockASTNode <: AbstractASTNode end

struct Error <: Terminal
    kind::ErrorKind
    expected
    got
end
iserror(::Error) = true
Base.String(node::RedTree.Node{Error}) = RedTree.tailcontents(node)

struct Incomplete{T} <: AbstractASTNode
    exprs::EXPRList
    error::EXPR{<:Union{Error, Incomplete}}
end

const Maybe{T} = Union{T, Nothing}
const EXPRErr{T} = Union{EXPR{T}, EXPR{Error}}

@generated function EXPRS.allchildren(x::AbstractASTNode)
    expr = Expr(:tuple)
    for i in 1:fieldcount(x)
        nam = fieldname(x, i)
        typ = fieldtype(x,i)
        if typ <: EXPRList
            push!(expr.args, Expr(:(...), :(x.$(nam))))
        else
            push!(expr.args, :(x.$nam))
        end
    end
    return expr
end

struct DevMod <: AbstractASTNode
    kw::EXPR{Keyword}
    slash::Union{Nothing, EXPR{Notation}}
    distr::Union{Nothing, EXPR{Identifier}}
    eq::EXPR{Notation}
    val::EXPR
end

# TODO struct LotMod

struct Parameter <: AbstractASTNode
    name::EXPR{Identifier}
    eq::Maybe{EXPR{Notation}}
    val::Maybe{EXPR}
    mod::Maybe{EXPR{DevMod}}
end

struct Simulator <: AbstractASTNode
    kw::EXPR{Keyword}
    langkw::EXPR{Keyword}
    eq::EXPR{Notation}
    lang::EXPR{Keyword}
    params::EXPRList{Parameter} # Can only be insensitive = true/false ?
    nl::EXPR{Notation}
end

struct ParamStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end

struct TempStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    temp::EXPR
    nl::EXPR{Notation}
end

struct CSParamStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end

# TODO: https://cseweb.ucsd.edu/classes/wi10/cse241a/assign/hspice_sa.pdf page 42
struct NodeName <: AbstractASTNode
    name::Union{EXPR{Identifier}, EXPR{NumberLiteral}}
end

struct UnaryOp <: AbstractASTNode
    op::EXPR{Operator}
    operand::EXPR
end

struct SubNode <: AbstractASTNode
    dot::EXPR{Notation}
    id::EXPR{NodeName}
end

struct HierarchialNode <: AbstractASTNode
    base::EXPR{NodeName}
    subnodes::EXPRList{SubNode}
end

struct GlobalStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    nodes::EXPRList{HierarchialNode}
    nl::EXPR{Notation}
end

struct BinaryExpression <: AbstractASTNode
    lhs::EXPR
    op::EXPR{Operator}
    rhs::EXPR
end

struct EndStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    nl::EXPR{Notation}
end

struct EndlStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    endl_id::Union{Nothing, EXPR{Identifier}}
    nl::EXPR{Notation}
end

struct Brace <: AbstractASTNode
    lbrace::EXPR{Notation}
    inner::EXPRErr
    rbrace::EXPRErr{Notation}
end

struct Parens <: AbstractASTNode
    lparen::EXPR{Notation}
    inner::EXPRErr
    rparen::EXPRErr{Notation}
end

struct Prime <: AbstractASTNode
    lquote::EXPR{Notation}
    inner::EXPRErr
    rquote::EXPRErr{Notation}
end

struct FunctionArgs{T} <: AbstractASTNode
    comma::Maybe{EXPR{Notation}}
    item::T
end

struct FunctionCall <: AbstractASTNode
    id::Union{EXPR{Identifier}, EXPR{BuiltinFunc}}
    lparen::EXPR{Notation}
    args::EXPRList{FunctionArgs{EXPR}}
    rparen::EXPRErr{Notation}
end

struct Square <: AbstractASTNode
    lsquare::EXPR{Notation}
    args::EXPRList
    rsquare::EXPRErr{Notation}
end

struct Title <: AbstractASTNode
    dot::Maybe{EXPR{Notation}}
    kw::Maybe{EXPR{Keyword}}
    line::EXPR{Notation}
    nl::EXPR{Notation}
end
istitle(::Title) = true

struct DCCommand <: AbstractASTNode
    srcname::EXPR{Identifier}
    start::EXPR
    stop::EXPR
    step::EXPR
end

struct DCStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    commands::EXPRList{DCCommand}
    nl::EXPR{Notation}
end

struct ACCommand <: AbstractASTNode
    srcname::EXPR{Identifier}
    n::EXPR
    fstart::EXPR
    fstop::EXPR
end

struct ACStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    command::EXPR{ACCommand} # can there be multiple like for DC?
    nl::EXPR{Notation}
end

struct IncludeStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    path::EXPR{StringLiteral}
    nl::EXPR{Notation}
end

struct HDLStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    path::EXPR{StringLiteral}
    nl::EXPR{Notation}
end

struct LibStatement <: AbstractBlockASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    name::EXPR{Identifier}
    nl::EXPR{Notation}
    stmts::EXPRList
    endd::EXPR{EndlStatement}
end

struct LibInclude <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    path::EXPR{StringLiteral}
    name::EXPR{Identifier}
    nl::EXPR{Notation}
end


struct OptionStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end

struct WidthStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end

abstract type AbstractInstanceNode <: AbstractASTNode end

struct Capacitor <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    pos::EXPR{HierarchialNode}
    neg::EXPR{HierarchialNode}
    val::Union{EXPR, Nothing} # the value can be specified in a parameter
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end

struct Resistor <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    pos::EXPR{HierarchialNode}
    neg::EXPR{HierarchialNode}
    val::Union{EXPR, Nothing} # can be a model, a value, or nothing (value is specified in a parameter)
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end

struct Inductor <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    pos::EXPR{HierarchialNode}
    neg::EXPR{HierarchialNode}
    val::Union{EXPR, Nothing} # the value can be specified in a parameter
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end

# TODO: Distortions http://bwrcs.eecs.berkeley.edu/Classes/IcBook/SPICE/UserGuide/elements.html#366
struct ACSource <: AbstractInstanceNode
    kw::EXPR{Keyword}
    eq::Maybe{EXPR{Notation}}
    acmag::EXPR
    # TODO
    # acphase::Union{Nothing, EXPR}
end

struct DCSource <: AbstractInstanceNode
    kw::Union{Nothing, EXPR{Keyword}}
    eq::Maybe{EXPR{Notation}}
    dcval::EXPR
end

struct TranSource <: AbstractInstanceNode
    kw::EXPR{Keyword}
    values::EXPRList
end

struct Voltage <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    pos::EXPR{HierarchialNode}
    neg::EXPR{HierarchialNode}
    vals::EXPRList{Union{ACSource, DCSource, TranSource}}
    nl::EXPR{Notation}
end

struct Current <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    pos::EXPR{HierarchialNode}
    neg::EXPR{HierarchialNode}
    vals::EXPRList{Union{ACSource, DCSource, TranSource}}
    nl::EXPR{Notation}
end

struct Behavioral <: AbstractASTNode
    name::EXPR{HierarchialNode}
    pos::EXPR{HierarchialNode}
    neg::EXPR{HierarchialNode}
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end

abstract type SourceSpecASTNode <: AbstractASTNode end

struct VoltageControl <: SourceSpecASTNode
    cpos::Union{EXPR{HierarchialNode}, Nothing}
    cneg::Union{EXPR{HierarchialNode}, Nothing}
    val::Union{EXPR, Nothing} # can be nonlinear (value=)
    params::EXPRList{Parameter}
end

struct CurrentControl <: SourceSpecASTNode
    vnam::Union{EXPR{HierarchialNode}, Nothing}
    val::Union{EXPR, Nothing} # can be nonlinear (value=)
    params::EXPRList{Parameter}
end

struct PolyControl <: SourceSpecASTNode
    poly::EXPR{Keyword}
    dimensions::EXPR{NumberLiteral}  # The N in POLY(N)
    args::EXPRList   # All remaining arguments (control nodes + coefficients)
end

struct TableControl <: SourceSpecASTNode
    table::EXPR{Keyword}
    expr::EXPR   # The expression after TABLE
    eq::Maybe{EXPR{Notation}}
    args::EXPRList   # (x,y) pairs
end

struct ControlledSource{in, out} <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    pos::EXPR{HierarchialNode}
    neg::EXPR{HierarchialNode}
    val::EXPR{<:SourceSpecASTNode}
    nl::EXPR{Notation}
end

struct BipolarTransistor <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    c::EXPR{HierarchialNode}
    b::EXPR{HierarchialNode}
    e::EXPR{HierarchialNode}
    s::Union{Nothing, EXPR{HierarchialNode}}
    t::Union{Nothing, EXPR{HierarchialNode}}
    model::EXPR{HierarchialNode}
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end


struct Diode <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    pos::EXPR{HierarchialNode}
    neg::EXPR{HierarchialNode}
    model::EXPR{HierarchialNode}
    params::EXPRList{Parameter}
    nl::EXPR{Notation}
end

struct MOSFET <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    d::EXPR{HierarchialNode}
    g::EXPR{HierarchialNode}
    s::EXPR{HierarchialNode}
    b::EXPR{HierarchialNode}
    model::EXPR{HierarchialNode}
    parameters::EXPRList{Parameter}
    nl::EXPR{Notation}
end

struct SParameterElement <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    nd1::EXPR{HierarchialNode}
    nd2::EXPR{HierarchialNode}
    model::EXPR{HierarchialNode}
    parameters::EXPRList{Parameter}
    nl::EXPR{Notation}
end

struct Switch <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    nd1::EXPR{HierarchialNode}
    nd2::EXPR{HierarchialNode}
    cnd1::EXPR{HierarchialNode}
    cnd2::EXPR{HierarchialNode}
    model::EXPR{HierarchialNode}
    onoff::EXPR{Keyword}
    nl::EXPR{Notation}
end

struct JuliaEscape <: AbstractASTNode
    open::EXPR{Notation}
    body::EXPR{JuliaEscapeBody}
    rparen::EXPR{Notation}
end

struct JuliaDevice <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    nodes::EXPRList{HierarchialNode}
    dev::EXPR{JuliaEscape}
    nl::EXPR{Notation}
end

struct OSDIDevice <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    nodes::EXPRList{HierarchialNode}
    model::Maybe{EXPR{HierarchialNode}}  # Model name before parameters
    parameters::EXPRList{Parameter}
    model_after::Maybe{EXPR{Identifier}}  # Model name after parameters (bare identifier)
    nl::EXPR{Notation}
end

struct SubcktCall <: AbstractInstanceNode
    name::EXPR{HierarchialNode}
    nodes::EXPRList{HierarchialNode}
    model::Maybe{EXPR{HierarchialNode}}  # Model name before parameters
    parameters::EXPRList{Parameter}
    model_after::Maybe{EXPR{Identifier}}  # Model name after parameters (bare identifier)
    nl::EXPR{Notation}
end


struct Subckt <: AbstractBlockASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    name::Union{EXPR{Identifier}, EXPR{NumberLiteral}}
    subckt_nodes::EXPRList{HierarchialNode}
    parameters::EXPRList{Parameter}
    nl1::EXPR{Notation}
    stmts::EXPRList
    dot2::EXPR{Notation}
    ends::EXPR{Keyword}
    name_end::Union{Nothing,EXPR{Identifier},EXPR{NumberLiteral}}
    nl2::EXPR{Notation}
end

struct Model <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    name::EXPR{HierarchialNode}
    typ::EXPR{Identifier}
    parameters::EXPRList{Parameter}
    nl::EXPR{Notation}
end

struct TernaryExpr <: AbstractASTNode
    condition::EXPR
    questionmark::EXPR{Notation}
    ifcase::EXPR
    colon::EXPRErr{Notation}
    elsecase::EXPR
end

struct DataStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    blockname::EXPR{Identifier}
    row_names::EXPRList{Identifier}
    values::EXPRList{NumberLiteral}
    nl::EXPR{Notation}
    dot2::EXPR{Notation}
    endkw::EXPR{Keyword}
    nl2::EXPR{Notation}
end

struct WildCard <: AbstractASTNode
    v::Union{Nothing, EXPR}
    star::EXPR{Notation}
end
struct Coloned <: AbstractASTNode
    l::EXPR{Identifier} # could be more?
    colon::EXPR{Notation}
    r::EXPR{Identifier} # could be more?
end
struct ICEntry <: AbstractASTNode
    name::EXPR{Identifier}
    lparen::EXPR{Notation}
    arg::EXPR
    rparen::EXPR{Notation}
    eq::EXPR{Notation}
    val::EXPR # number, unit,
end

struct ICStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    entries::EXPRList{ICEntry}
    nl::EXPR{Notation}
end

struct PrintStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    stmts::EXPRList
    nl::EXPR{Notation}
end
# .MEASURE

struct When <: AbstractASTNode
    type::EXPR{Keyword}
    body::EXPR
end

struct At <: AbstractASTNode
    type::EXPR{Keyword}
    ex::EXPR{Notation}
    body::EXPR
end

struct RiseFallCross <: AbstractASTNode
    type::EXPR{Keyword}
    eq::EXPR{Notation}
    val::Union{EXPR{Keyword}, EXPR{NumberLiteral}}
end

# TODO: Rename
struct TD_ <: AbstractASTNode
    kw::EXPR{Keyword}
    eq::EXPR{Notation}
    val::EXPR
end

struct FindDerivParam <: AbstractASTNode
    type::EXPR{Keyword}
    eq::Union{Nothing, EXPR{Notation}}
    body::EXPR
end


#=
Point along abcissa
Syntax: .MEAS[SURE] [AC|DC|OP|TRAN|TF|NOISE] <name>
 + [<FIND|DERIV|PARAM> <expr>]
 + [WHEN <expr> | AT=<expr>]]
 + [TD=<val1>] [<RISE|FALL|CROSS>=[<count1>|LAST]]
 =#
struct MeasurePointStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    type::EXPR{Keyword}
    name::Union{EXPR{StringLiteral}, EXPR{Identifier}}
    findderivparam::Union{Nothing, EXPR{FindDerivParam}}
    whenat::Union{Nothing, EXPR{When}, EXPR{At}}
    risefallcross::Union{Nothing, EXPR{RiseFallCross}}
    td::Union{Nothing, EXPR{TD_}}
    nl::EXPR{Notation}
end


struct AvgMaxMinPPRmsInteg <: AbstractASTNode
    kw::EXPR{Keyword}
    body::EXPR
end

struct Val_ <: AbstractASTNode
    kw::EXPR{Keyword}
    eq::EXPR{Notation}
    val::EXPR
end

struct TrigTarg <: AbstractASTNode
    kw::EXPR{Keyword}
    lhs::EXPR
    val::Union{Nothing, EXPR{Val_}}
    td::Union{Nothing, EXPR{TD_}}
    rfc::Union{Nothing, EXPR{RiseFallCross}}
end

 #=
 Range over abcissa
 Syntax: .MEAS [AC|DC|OP|TRAN|TF|NOISE] <name>
 + [<AVG|MAX|MIN|PP|RMS|INTEG> <expr>]
 + [TRIG <lhs1> [[VAL]=]<rhs1>] [TD=<val1>]
 + [<RISE|FALL|CROSS>=<count1>]
 + [TARG <lhs2> [[VAL]=]<rhs2>] [TD=<val2>]
 + [<RISE|FALL|CROSS>=<count2>]
 =#
struct MeasureRangeStatement <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    type::EXPR{Keyword}
    name::Union{EXPR{StringLiteral}, EXPR{Identifier}}
    avgmaxminpprmsinteg::Union{Nothing, EXPR{AvgMaxMinPPRmsInteg}}
    trig::Union{Nothing, EXPR{TrigTarg}}
    targ::Union{Nothing, EXPR{TrigTarg}}
    nl::EXPR{Notation}
end

struct Tran <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    # TODO: Fixup, just temporary for now
    tstep::Union{Nothing, EXPR{NumberLiteral}}
    tstop::EXPR{NumberLiteral}
    tstart::Union{Nothing, EXPR{NumberLiteral}}
    tmax::Union{Nothing, EXPR{NumberLiteral}}
    uic::Union{Nothing, EXPR{Identifier}}
    nl::EXPR{Notation}
end

struct Condition <: AbstractASTNode
    lparen::EXPR{Notation}
    body::EXPR
    rparen::EXPR{Notation}
end

struct IfElseCase <: AbstractASTNode
    dot::EXPR{Notation}
    kw::EXPR{Keyword}
    condition::Union{Nothing, EXPR{Condition}}
    nl::EXPR{Notation}
    stmts::EXPRList
end

struct IfBlock <: AbstractASTNode
    cases::EXPRList{IfElseCase}
    dot::EXPR{Notation}
    endif::EXPR{Keyword}
    nl::EXPR{Notation}
end

struct SPICENetlistSource <: AbstractBlockASTNode
    stmts::EXPRList{Any}
end
EXPRS.allchildren(sns::SPICENetlistSource) = sns.stmts

########################

function EXPR!(val::Terminal, ps)
    prev = ps.pos
    fw = ps.npos - prev
    next(ps)
    offset = UInt32(ps.t.startbyte - prev)
    return EXPR(fw, offset, UInt32(ps.t.endbyte - ps.t.startbyte + 1), val)
end

# Return an expanded version of terminal extending to the end of the line
function extend_to_line_end(val::EXPR{<:Terminal}, ps)
    (; fullwidth, off, width, form) = val
    while !eol(ps)
        fullwidth += ps.npos - ps.pos
        next(ps)
    end
    fullwidth += ps.npos - ps.pos
    next(ps)
    return EXPR(fullwidth, off, width, form)
end
