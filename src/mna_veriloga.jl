# MNA Verilog-A Integration Layer
# Transforms Verilog-A models to MNA-compatible devices

using VerilogAParser
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
using AbstractTrees

export @mna_va_str, VADevice, mna_va_device

const VAP = VerilogAParser
const VACP = VerilogAParser.VerilogACSTParser
const VANode = VACP.Node

# Scale factor mapping (same as vasim.jl)
const va_sf_mapping = Dict(
    'T' => 1e12, 'G' => 1e9, 'M' => 1e6, 'K' => 1e3, 'k' => 1e3,
    'm' => 1e-3, 'u' => 1e-6, 'n' => 1e-9, 'p' => 1e-12, 'f' => 1e-15, 'a' => 1e-18
)

"""
    VADevice

Abstract base type for Verilog-A device models compatible with MNA.
"""
abstract type VADevice <: MNADevice end

"""
    VAContribution

Represents a Verilog-A contribution statement parsed into MNA-compatible form.
"""
struct VAContribution
    kind::Symbol           # :current or :voltage
    pos_node::Symbol       # positive node
    neg_node::Symbol       # negative node (or :gnd for single-node probes)
    expr_ast::Expr         # Julia expression for the contribution value
    is_linear::Bool        # Whether the contribution is linear
    linear_coeff::Float64  # For linear contributions: coefficient (e.g., G for I = G*V)
end

"""
    VAModule

Represents a parsed Verilog-A module ready for MNA simulation.
"""
struct VAModule
    name::Symbol
    pins::Vector{Symbol}
    internal_nodes::Vector{Symbol}
    parameters::Dict{Symbol, Any}  # name => (type, default)
    contributions::Vector{VAContribution}
    var_types::Dict{Symbol, Type}
    analog_code::Expr
end

# Helper to extract identifier string
function va_assemble_id(id)
    if id isa Node{SystemIdentifier}
        return String(id)
    elseif id isa Node{Identifier}
        return join(va_assemble_id(c) for c in AbstractTrees.children(id))
    elseif id isa Node{IdentifierPart}
        s = String(id)
        id.escaped && (s = s[2:end])
        return s
    elseif id isa Node{IdentifierConcatItem}
        return va_assemble_id(id.id)
    else
        return String(id)
    end
end

formof(e::VANode{S}) where {S} = S

# Parse a Verilog-A module for MNA
function parse_va_module(vm::VANode{VerilogModule})
    modname = Symbol(String(vm.id))

    # Get pins
    pins = Symbol[]
    if vm.port_list !== nothing
        for port_decl in vm.port_list.ports
            push!(pins, Symbol(va_assemble_id(port_decl.item)))
        end
    end

    # Parse module contents
    internal_nodes = Symbol[]
    parameters = Dict{Symbol, Any}()
    var_types = Dict{Symbol, Type}()
    analog_stmts = Any[]

    for child in vm.items
        item = child.item
        @case formof(item) begin
            NetDeclaration => begin
                for net in item.net_names
                    id = Symbol(va_assemble_id(net.item))
                    if !(id in pins)
                        push!(internal_nodes, id)
                    end
                end
            end
            ParameterDeclaration => begin
                for param in item.params
                    param = param.item
                    pT = Float64
                    if item.ptype !== nothing
                        pT = item.ptype.kw == REAL ? Float64 : Int
                    end
                    paramsym = Symbol(va_assemble_id(param.id))
                    default_val = parse_va_literal(param.default_expr)
                    parameters[paramsym] = (type=pT, default=default_val)
                    var_types[paramsym] = pT
                end
            end
            IntRealDeclaration => begin
                T = item.kw.kw == REAL ? Float64 : Int
                for ident in item.idents
                    name = Symbol(String(ident.item))
                    var_types[name] = T
                end
            end
            AnalogBlock => begin
                push!(analog_stmts, item.stmt)
            end
            _ => nothing
        end
    end

    # Parse analog block for contributions
    contributions = VAContribution[]
    analog_code = Expr(:block)

    for stmt in analog_stmts
        parsed = parse_va_analog(stmt, pins, internal_nodes, var_types)
        append!(contributions, parsed.contributions)
        push!(analog_code.args, parsed.code)
    end

    return VAModule(modname, pins, internal_nodes, parameters, contributions, var_types, analog_code)
end

# Parse literal values
function parse_va_literal(node)
    if node isa VANode{Literal}
        return Base.parse(Float64, String(node))
    elseif node isa VANode{FloatLiteral}
        txt = String(node)
        sf = nothing
        if !isempty(txt) && is_scale_factor(txt[end])
            sf = txt[end]
            txt = txt[1:end-1]
        end
        ret = Base.parse(Float64, txt)
        if sf !== nothing && haskey(va_sf_mapping, sf)
            ret *= va_sf_mapping[sf]
        end
        return ret
    else
        return 0.0
    end
end

# Parse analog block statements
function parse_va_analog(stmt, pins, internal_nodes, var_types)
    contributions = VAContribution[]
    code = Expr(:block)

    if stmt isa VANode{AnalogSeqBlock}
        for s in stmt.stmts
            result = parse_va_analog(s, pins, internal_nodes, var_types)
            append!(contributions, result.contributions)
            push!(code.args, result.code)
        end
    elseif stmt isa VANode{AnalogStatement}
        return parse_va_analog(stmt.stmt, pins, internal_nodes, var_types)
    elseif stmt isa VANode{ContributionStatement}
        contrib = parse_va_contribution(stmt, pins, internal_nodes, var_types)
        push!(contributions, contrib)
        push!(code.args, contrib.expr_ast)
    elseif stmt isa VANode{AnalogVariableAssignment}
        assignee = Symbol(stmt.lvalue)
        expr = va_expr_to_julia(stmt.rvalue, var_types)
        push!(code.args, :($assignee = $expr))
    elseif stmt isa VANode{AnalogConditionalBlock}
        # Handle if-else
        aif = stmt.aif
        cond = va_expr_to_julia(aif.condition, var_types)
        then_result = parse_va_analog(aif.stmt, pins, internal_nodes, var_types)
        ifexpr = Expr(:if, cond, then_result.code)
        append!(contributions, then_result.contributions)

        for elsecase in stmt.elsecases
            if formof(elsecase.stmt) == AnalogIf
                elif = elsecase.stmt
                econd = va_expr_to_julia(elif.condition, var_types)
                else_result = parse_va_analog(elif.stmt, pins, internal_nodes, var_types)
                push!(ifexpr.args, Expr(:elseif, econd, else_result.code))
                append!(contributions, else_result.contributions)
            else
                else_result = parse_va_analog(elsecase.stmt, pins, internal_nodes, var_types)
                push!(ifexpr.args, else_result.code)
                append!(contributions, else_result.contributions)
            end
        end
        push!(code.args, ifexpr)
    end

    return (contributions=contributions, code=code)
end

# Parse a contribution statement
function parse_va_contribution(stmt::VANode{ContributionStatement}, pins, internal_nodes, var_types)
    bpfc = stmt.lvalue
    kind_str = Symbol(bpfc.id)

    # Determine contribution kind
    kind = if kind_str == :I
        :current
    elseif kind_str == :V
        :voltage
    else
        error("Unknown contribution kind: $kind_str")
    end

    # Get nodes
    refs = [Symbol(va_assemble_id(ref.item)) for ref in bpfc.references]

    pos_node = refs[1]
    neg_node = length(refs) > 1 ? refs[2] : :gnd

    # Parse the expression
    expr = va_expr_to_julia(stmt.assign_expr, var_types)

    # Try to detect linear contributions
    is_linear, linear_coeff = detect_linear_contribution(stmt.assign_expr, pos_node, neg_node, var_types)

    return VAContribution(kind, pos_node, neg_node, expr, is_linear, linear_coeff)
end

# Convert Verilog-A expression to Julia
function va_expr_to_julia(node, var_types)
    if node isa VANode{Literal}
        return parse_va_literal(node)
    elseif node isa VANode{FloatLiteral}
        return parse_va_literal(node)
    elseif node isa VANode{IdentifierPrimary}
        return Symbol(va_assemble_id(node.id))
    elseif node isa VANode{BinaryExpression}
        op = Symbol(node.op)
        lhs = va_expr_to_julia(node.lhs, var_types)
        rhs = va_expr_to_julia(node.rhs, var_types)
        return Expr(:call, op, lhs, rhs)
    elseif node isa VANode{UnaryOp}
        op = Symbol(node.op)
        operand = va_expr_to_julia(node.operand, var_types)
        return Expr(:call, op, operand)
    elseif node isa VANode{Parens}
        return va_expr_to_julia(node.inner, var_types)
    elseif node isa VANode{FunctionCall}
        fname = Symbol(node.id)
        args = [va_expr_to_julia(arg.item, var_types) for arg in node.args]

        # Handle V() and I() probes
        if fname == :V
            if length(args) == 1
                return :(V($(args[1])))
            else
                return :(V($(args[1]), $(args[2])))
            end
        elseif fname == :I
            if length(args) == 1
                return :(I($(args[1])))
            else
                return :(I($(args[1]), $(args[2])))
            end
        elseif fname == :ddt
            return :(ddt($(args...)))
        elseif fname == :abs
            return :(abs($(args...)))
        elseif fname == :exp
            return :(exp($(args...)))
        elseif fname == :ln || fname == Symbol("\$ln")
            return :(log($(args...)))
        elseif fname == :log || fname == Symbol("\$log10")
            return :(log10($(args...)))
        elseif fname == :sqrt || fname == Symbol("\$sqrt")
            return :(sqrt($(args...)))
        elseif fname == :pow
            return :(^($(args...)))
        elseif fname == :min
            return :(min($(args...)))
        elseif fname == :max
            return :(max($(args...)))
        elseif fname == Symbol("\$temperature")
            return :(_temperature)
        else
            return Expr(:call, fname, args...)
        end
    elseif node isa VANode{TernaryExpr}
        cond = va_expr_to_julia(node.condition, var_types)
        ifcase = va_expr_to_julia(node.ifcase, var_types)
        elsecase = va_expr_to_julia(node.elsecase, var_types)
        return Expr(:if, cond, ifcase, elsecase)
    elseif node isa VANode{SystemIdentifier}
        id = Symbol(node)
        if id == Symbol("\$temperature")
            return :(_temperature)
        else
            return id
        end
    else
        # Fallback
        return 0.0
    end
end

# Try to detect if a contribution is linear (I = G*V form)
function detect_linear_contribution(expr, pos_node, neg_node, var_types)
    # Simple pattern matching for I(p,n) <+ V(p,n)/R or I(p,n) <+ V(p,n)*G
    if expr isa VANode{BinaryExpression}
        op = Symbol(expr.op)

        if op == :/
            # Check if it's V(p,n)/R form
            lhs = expr.lhs
            rhs = expr.rhs

            if lhs isa VANode{FunctionCall} && Symbol(lhs.id) == :V
                # Check if the V() probe matches our branch
                refs = [Symbol(va_assemble_id(arg.item)) for arg in lhs.args]
                if length(refs) == 2 && refs[1] == pos_node && refs[2] == neg_node
                    # RHS should be a parameter (resistance)
                    if rhs isa VANode{IdentifierPrimary}
                        # Linear: I = V/R, coefficient = 1/R (to be evaluated at runtime)
                        return true, 1.0  # Placeholder, actual value computed at instantiation
                    end
                end
            end
        elseif op == :*
            # Check for V(p,n)*G form
            # Either lhs or rhs could be the V() probe
            for (v_side, g_side) in [(expr.lhs, expr.rhs), (expr.rhs, expr.lhs)]
                if v_side isa VANode{FunctionCall} && Symbol(v_side.id) == :V
                    refs = [Symbol(va_assemble_id(arg.item)) for arg in v_side.args]
                    if length(refs) == 2 && refs[1] == pos_node && refs[2] == neg_node
                        if g_side isa VANode{IdentifierPrimary}
                            return true, 1.0
                        end
                    end
                end
            end
        end
    end

    return false, 0.0
end

"""
    make_mna_device(vamod::VAModule)

Generate Julia code for an MNA-compatible device from a parsed Verilog-A module.
"""
function make_mna_device(vamod::VAModule)
    modname = vamod.name
    pins = vamod.pins
    n_pins = length(pins)

    # Generate struct fields for parameters
    struct_fields = Expr[]
    for (pname, pinfo) in vamod.parameters
        ptype = pinfo.type
        default = pinfo.default
        push!(struct_fields, :($pname::$ptype = $default))
    end

    # Generate the stamp! function
    # For each contribution, generate appropriate MNA stamps
    stamp_code = Expr(:block)

    for (i, contrib) in enumerate(vamod.contributions)
        pos = contrib.pos_node
        neg = contrib.neg_node

        # Get node indices
        pos_idx = findfirst(==(pos), pins)
        neg_idx = neg == :gnd ? 0 : findfirst(==(neg), pins)

        if contrib.kind == :current
            if contrib.is_linear
                # Linear resistor-like contribution
                # Need to extract the conductance from the expression
                push!(stamp_code.args, quote
                    # Evaluate conductance from expression
                    # For I(p,n) <+ V(p,n)/R, G = 1/R
                    _V_probe = 1.0  # Symbolic voltage for differentiation
                    _G = let $(pins[pos_idx]) = _V_probe, $(pins[neg_idx !== nothing ? neg_idx : 1]) = 0.0
                        $(contrib.expr_ast)
                    end
                    stamp_conductance!(circuit, device.nets[$pos_idx], device.nets[$(neg_idx === nothing ? 0 : neg_idx)], _G)
                end)
            else
                # Nonlinear contribution - add as nonlinear element
                push!(stamp_code.args, quote
                    # Add nonlinear element for contribution $i
                    add_nonlinear!(circuit, device, $i)
                end)
            end
        elseif contrib.kind == :voltage
            # Voltage source contribution
            push!(stamp_code.args, quote
                # Voltage source: V(pos, neg) = value
                _V = let
                    $(contrib.expr_ast)
                end
                stamp_voltage_source!(circuit, device.nets[$pos_idx],
                    device.nets[$(neg_idx === nothing ? 0 : neg_idx)], device.branches[$i], _V)
            end)
        end
    end

    # Generate the full device code
    return quote
        @kwdef struct $modname <: VADevice
            nets::Vector{MNANet}
            branches::Vector{BranchVar}
            $(struct_fields...)
        end

        function $modname(circuit::MNACircuit, $(pins...); kwargs...)
            nets = [get_net!(circuit, n) for n in [$(pins...)]]
            branches = BranchVar[]
            # Create branches for voltage sources
            for (i, contrib) in enumerate($(vamod.contributions))
                if contrib.kind == :voltage
                    push!(branches, get_branch!(circuit, Symbol($(QuoteNode(modname)), :_, i)))
                end
            end
            $modname(nets=nets, branches=branches; kwargs...)
        end

        function stamp!(device::$modname, circuit::MNACircuit)
            $stamp_code
        end
    end
end

"""
    @mna_va_str(str)

String macro for creating MNA-compatible devices from inline Verilog-A code.

# Example
```julia
mna_va\"\"\"
module VAR(p,n);
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
    parsed = parse_va_module(vamod)
    code = make_mna_device(parsed)

    esc(code)
end

"""
    MNAVAResistor

Simple Verilog-A resistor device for MNA simulation.
This serves as a test/example of the VA-MNA integration.
"""
struct MNAVAResistor <: VADevice
    n_pos::MNANet
    n_neg::MNANet
    R::Float64
    name::Symbol
end

function MNAVAResistor(circuit::MNACircuit, n_pos, n_neg; R=1000.0, name=:VAR)
    net_pos = get_net!(circuit, n_pos)
    net_neg = get_net!(circuit, n_neg)
    MNAVAResistor(net_pos, net_neg, Float64(R), name)
end

function stamp!(device::MNAVAResistor, circuit::MNACircuit)
    # I(p,n) <+ V(p,n)/R is equivalent to a conductance G = 1/R
    G = 1.0 / device.R
    stamp_conductance!(circuit, device.n_pos, device.n_neg, G)
end

"""
    varesistor!(circuit, n_pos, n_neg; R=1000.0, name=:VAR)

Add a Verilog-A style resistor to the circuit.
"""
function varesistor!(circuit::MNACircuit, n_pos, n_neg; R=1000.0, name=:VAR)
    dev = MNAVAResistor(circuit, n_pos, n_neg; R=R, name=name)
    stamp!(dev, circuit)
    return dev
end

#=
BSIM-CMG Integration Support

The BSIM-CMG model is a complex nonlinear MOSFET model with:
- 4 external terminals: d (drain), g (gate), s (source), e (substrate)
- Internal nodes for parasitic resistances
- Many parameters (~300+)
- Nonlinear I-V characteristics

For MNA integration, we need to:
1. Stamp the linear part (parasitic resistances, capacitances)
2. Add nonlinear elements for the core I-V relationships
3. Handle the Newton-Raphson iteration for DC operating point
=#

"""
    MNAMosfet

Simplified MOSFET model for MNA simulation.
This provides basic Level-1 style MOSFET behavior.
"""
struct MNAMosfet <: MNADevice
    n_drain::MNANet
    n_gate::MNANet
    n_source::MNANet
    n_bulk::MNANet
    # Parameters
    vth0::Float64      # Threshold voltage
    kp::Float64        # Transconductance parameter (μCox)
    lambda::Float64    # Channel length modulation
    w::Float64         # Width
    l::Float64         # Length
    ntype::Bool        # true for NMOS, false for PMOS
    name::Symbol
end

function MNAMosfet(circuit::MNACircuit, n_d, n_g, n_s, n_b;
                   vth0=0.7, kp=110e-6, lambda=0.04, w=10e-6, l=1e-6,
                   ntype=true, name=:M)
    nd = get_net!(circuit, n_d)
    ng = get_net!(circuit, n_g)
    ns = get_net!(circuit, n_s)
    nb = get_net!(circuit, n_b)
    MNAMosfet(nd, ng, ns, nb, Float64(vth0), Float64(kp), Float64(lambda),
              Float64(w), Float64(l), ntype, name)
end

function stamp!(device::MNAMosfet, circuit::MNACircuit)
    # MOSFETs are nonlinear - add to nonlinear element list
    push!(circuit.nonlinear_elements, (x, params, circ) -> mosfet_stamp(device, x, circ))
end

function mosfet_stamp(device::MNAMosfet, x::Vector{Float64}, circuit::MNACircuit)
    # Get node indices
    i_d = node_index(device.n_drain)
    i_g = node_index(device.n_gate)
    i_s = node_index(device.n_source)
    i_b = node_index(device.n_bulk)

    # Get node voltages (ground = 0)
    Vd = i_d > 0 ? x[i_d] : 0.0
    Vg = i_g > 0 ? x[i_g] : 0.0
    Vs = i_s > 0 ? x[i_s] : 0.0
    Vb = i_b > 0 ? x[i_b] : 0.0

    # Beta = kp * W/L
    beta = device.kp * device.w / device.l
    Vth = device.vth0
    lambda = device.lambda

    # Calculate drain current and conductances
    Id = 0.0
    gm = 0.0   # dId/dVg
    gds = 0.0  # dId/dVd

    if device.ntype
        # NMOS: current flows from drain to source when Vgs > Vth
        Vgs = Vg - Vs
        Vds = Vd - Vs

        if Vgs <= Vth
            # Cutoff: very small leakage
            Id = 1e-12 * Vds  # Small leakage for numerical stability
            gds = 1e-12
        elseif Vds <= 0
            # Reverse mode - swap D and S conceptually
            Id = 1e-12 * Vds
            gds = 1e-12
        elseif Vds < Vgs - Vth
            # Linear/triode region
            Id = beta * ((Vgs - Vth) * Vds - 0.5 * Vds^2) * (1 + lambda * Vds)
            gm = beta * Vds * (1 + lambda * Vds)
            gds = beta * ((Vgs - Vth - Vds) * (1 + lambda * Vds) +
                  lambda * ((Vgs - Vth) * Vds - 0.5 * Vds^2))
        else
            # Saturation region
            Vdsat = Vgs - Vth
            Id = 0.5 * beta * Vdsat^2 * (1 + lambda * Vds)
            gm = beta * Vdsat * (1 + lambda * Vds)
            gds = 0.5 * beta * Vdsat^2 * lambda
        end
    else
        # PMOS: current flows from source to drain when Vsg > |Vth|
        # In MNA convention: positive Id = current INTO drain node
        # For PMOS, current flows S->D, so INTO drain is positive
        Vsg = Vs - Vg
        Vsd = Vs - Vd

        if Vsg <= Vth
            # Cutoff - tiny leakage
            Id = 1e-12 * Vsd  # Small current proportional to Vsd
            gds = 1e-12
        elseif Vsd <= 0
            # Reverse mode (Vd > Vs)
            Id = 1e-12 * Vsd
            gds = 1e-12
        elseif Vsd < Vsg - Vth
            # Linear/triode region
            Idmag = beta * ((Vsg - Vth) * Vsd - 0.5 * Vsd^2) * (1 + lambda * Vsd)
            Id = Idmag  # Positive: current flows INTO drain
            gm = beta * Vsd * (1 + lambda * Vsd)
            gds = beta * ((Vsg - Vth - Vsd) * (1 + lambda * Vsd) +
                  lambda * ((Vsg - Vth) * Vsd - 0.5 * Vsd^2))
        else
            # Saturation region
            Vdsat = Vsg - Vth
            Idmag = 0.5 * beta * Vdsat^2 * (1 + lambda * Vsd)
            Id = Idmag  # Positive: current flows INTO drain
            gm = beta * Vdsat * (1 + lambda * Vsd)
            gds = 0.5 * beta * Vdsat^2 * lambda
        end
    end

    # Residual vector: KCL contributions (current INTO node from device)
    # For NMOS: Id flows D→S, so current INTO drain = -Id, INTO source = +Id
    # For PMOS: current flows S→D (INTO drain), so current INTO drain = +Id, INTO source = -Id
    residual = zeros(length(x))

    if device.ntype
        # NMOS: current flows from drain to source
        if i_d > 0
            residual[i_d] = -Id  # Current leaving drain (into transistor)
        end
        if i_s > 0
            residual[i_s] = Id   # Current entering source (from transistor)
        end
    else
        # PMOS: current flows from source to drain (Id defined as magnitude)
        if i_d > 0
            residual[i_d] = Id   # Current entering drain (from transistor)
        end
        if i_s > 0
            residual[i_s] = -Id  # Current leaving source (into transistor)
        end
    end

    # Jacobian entries: df_nl/dx where f_nl is the residual
    # These are used in J = G - df_nl/dx formulation (but NonlinearSolve uses autodiff)
    jac_entries = Tuple{Int, Int, Float64}[]

    # Note: These jacobian entries are for reference/manual Newton only
    # NonlinearSolve uses AutoFiniteDiff so these aren't strictly needed
    if device.ntype
        # NMOS: residual[d] = -Id, residual[s] = +Id
        # d(residual[d])/dVd = -gds, d(residual[d])/dVg = -gm, d(residual[d])/dVs = gds+gm
        if i_d > 0
            push!(jac_entries, (i_d, i_d, -gds))
            i_g > 0 && push!(jac_entries, (i_d, i_g, -gm))
            i_s > 0 && push!(jac_entries, (i_d, i_s, gds + gm))
        end
        if i_s > 0
            push!(jac_entries, (i_s, i_d, gds))
            i_g > 0 && push!(jac_entries, (i_s, i_g, gm))
            push!(jac_entries, (i_s, i_s, -(gds + gm)))
        end
    else
        # PMOS: residual[d] = +Id, residual[s] = -Id
        # With Id as a function of Vsg, Vsd (both depend on Vs, Vg, Vd)
        if i_d > 0
            push!(jac_entries, (i_d, i_d, gds))
            i_g > 0 && push!(jac_entries, (i_d, i_g, gm))
            i_s > 0 && push!(jac_entries, (i_d, i_s, -(gds + gm)))
        end
        if i_s > 0
            push!(jac_entries, (i_s, i_d, -gds))
            i_g > 0 && push!(jac_entries, (i_s, i_g, -gm))
            push!(jac_entries, (i_s, i_s, gds + gm))
        end
    end

    return residual, jac_entries
end

"""
    nmos!(circuit, n_d, n_g, n_s, n_b; kwargs...)

Add an NMOS transistor to the circuit.
"""
function nmos!(circuit::MNACircuit, n_d, n_g, n_s, n_b; kwargs...)
    dev = MNAMosfet(circuit, n_d, n_g, n_s, n_b; ntype=true, kwargs...)
    stamp!(dev, circuit)
    return dev
end

"""
    pmos!(circuit, n_d, n_g, n_s, n_b; kwargs...)

Add a PMOS transistor to the circuit.
"""
function pmos!(circuit::MNACircuit, n_d, n_g, n_s, n_b; kwargs...)
    dev = MNAMosfet(circuit, n_d, n_g, n_s, n_b; ntype=false, kwargs...)
    stamp!(dev, circuit)
    return dev
end

export VADevice, MNAVAResistor, varesistor!, MNAMosfet, nmos!, pmos!, @mna_va_str
