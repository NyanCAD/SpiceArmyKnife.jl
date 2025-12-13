# MNA Netlist Converter
# Converts parsed SPICE/Spectre netlists to MNA circuit representation

using SpectreNetlistParser
using SpectreNetlistParser: SpectreNetlistCSTParser, SPICENetlistParser

export parse_netlist, build_circuit_from_netlist, spice_circuit, spectre_circuit

const SNode = SpectreNetlistCSTParser.Node
const SC = SpectreNetlistCSTParser
const SP = SPICENetlistParser.SPICENetlistCSTParser

# Helper to get ground-normalized node symbol
function normalize_node(s::AbstractString)
    s_lower = lowercase(strip(s))
    if s_lower == "0" || s_lower == "gnd"
        return :ground
    end
    return Symbol(s_lower)
end

"""
    NetlistContext

Context for netlist to MNA conversion.
"""
mutable struct NetlistContext
    circuit::MNACircuit
    models::Dict{Symbol, Any}  # Device models
    params::Dict{Symbol, Float64}  # Parameters
    subcircuits::Dict{Symbol, Any}  # Subcircuit definitions
    device_counter::Dict{Symbol, Int}  # For auto-naming
end

function NetlistContext(circuit::MNACircuit)
    NetlistContext(circuit, Dict(), Dict(), Dict(), Dict(:R=>0, :C=>0, :L=>0, :V=>0, :I=>0, :D=>0, :X=>0))
end

function next_name!(ctx::NetlistContext, prefix::Symbol)
    ctx.device_counter[prefix] = get(ctx.device_counter, prefix, 0) + 1
    return Symbol(prefix, ctx.device_counter[prefix])
end

"""
    parse_value(s::AbstractString) -> Float64

Parse a SPICE value string with unit suffixes.
"""
function parse_value(s::AbstractString)
    s = strip(lowercase(s))

    # Unit suffix multipliers
    suffixes = Dict(
        "t" => 1e12, "g" => 1e9, "meg" => 1e6, "k" => 1e3,
        "m" => 1e-3, "u" => 1e-6, "n" => 1e-9, "p" => 1e-12,
        "f" => 1e-15, "a" => 1e-18
    )

    # Try to parse with suffix
    for (suffix, mult) in suffixes
        if endswith(s, suffix)
            num_str = s[1:end-length(suffix)]
            return parse(Float64, num_str) * mult
        end
    end

    # Also handle _V, _A suffixes (Spectre style)
    s = replace(s, r"_[vaohmf]$"i => "")

    return parse(Float64, s)
end

"""
    parse_netlist(source::AbstractString; start_lang=:spice) -> SNode

Parse a netlist string into an AST.
"""
function parse_netlist(source::AbstractString; start_lang=:spice)
    return SpectreNetlistParser.parse(IOBuffer(source); start_lang=start_lang)
end

"""
    build_circuit_from_netlist(ast) -> MNACircuit

Build an MNA circuit from a parsed netlist AST.
"""
function build_circuit_from_netlist(ast; temp=27.0, gmin=1e-12)
    circuit = MNACircuit(temp=temp, gmin=gmin)

    # Add ground reference
    ground!(circuit, :ground)

    ctx = NetlistContext(circuit)

    # Process the AST
    process_netlist!(ctx, ast)

    return circuit
end

"""
    process_netlist!(ctx::NetlistContext, ast)

Process a netlist AST node.
"""
function process_netlist!(ctx::NetlistContext, ast)
    if ast isa SNode
        process_node!(ctx, ast)
    elseif ast isa Vector
        for node in ast
            process_netlist!(ctx, node)
        end
    end
end

# Generic fallback
function process_node!(ctx::NetlistContext, node::SNode)
    # Process children
    if hasfield(typeof(node.expr), :stmts)
        for stmt in node.expr.stmts
            process_node!(ctx, SNode(node.ps, stmt, node.startof + stmt.off))
        end
    end
end

# SPICE Resistor: Rname n+ n- value
function process_node!(ctx::NetlistContext, node::SNode{SP.Resistor})
    name = Symbol(String(node.expr.name))
    n_pos = normalize_node(String(node.expr.pos))
    n_neg = normalize_node(String(node.expr.neg))

    value = 1000.0  # default 1k
    if node.expr.val !== nothing
        try
            value = parse_value(String(node.expr.val))
        catch e
            @warn "Could not parse resistor value: $e"
        end
    end

    resistor!(ctx.circuit, n_pos, n_neg, value; name=name)
end

# SPICE Capacitor: Cname n+ n- value
function process_node!(ctx::NetlistContext, node::SNode{SP.Capacitor})
    name = Symbol(String(node.expr.name))
    n_pos = normalize_node(String(node.expr.pos))
    n_neg = normalize_node(String(node.expr.neg))

    value = 1e-12  # default 1pF
    if node.expr.val !== nothing
        try
            value = parse_value(String(node.expr.val))
        catch e
            @warn "Could not parse capacitor value: $e"
        end
    end

    capacitor!(ctx.circuit, n_pos, n_neg, value; name=name)
end

# SPICE Inductor: Lname n+ n- value
function process_node!(ctx::NetlistContext, node::SNode{SP.Inductor})
    name = Symbol(String(node.expr.name))
    n_pos = normalize_node(String(node.expr.pos))
    n_neg = normalize_node(String(node.expr.neg))

    value = 1e-9  # default 1nH
    if node.expr.val !== nothing
        try
            value = parse_value(String(node.expr.val))
        catch e
            @warn "Could not parse inductor value: $e"
        end
    end

    inductor!(ctx.circuit, n_pos, n_neg, value; name=name)
end

# SPICE Voltage Source: Vname n+ n- [DC value] [AC value]
function process_node!(ctx::NetlistContext, node::SNode{SP.Voltage})
    name = Symbol(String(node.expr.name))
    n_pos = normalize_node(String(node.expr.pos))
    n_neg = normalize_node(String(node.expr.neg))

    dc = 0.0
    ac = 0.0 + 0.0im
    tran = nothing

    # Process source values
    for val_item in node.expr.vals
        val_form = val_item.form
        if val_form isa SP.DCSource
            if val_form.dcval !== nothing
                try
                    dc = parse_value(String(val_form.dcval))
                catch
                end
            end
        elseif val_form isa SP.ACSource
            if val_form.acmag !== nothing
                try
                    ac = parse_value(String(val_form.acmag)) + 0.0im
                catch
                end
            end
        elseif val_form isa SP.TranSource
            # Handle transient sources (PULSE, SIN, etc.)
            # For now, just use DC value
        end
    end

    vsource!(ctx.circuit, n_pos, n_neg; dc=dc, ac=ac, tran=tran, name=name)
end

# SPICE Current Source: Iname n+ n- [DC value]
function process_node!(ctx::NetlistContext, node::SNode{SP.Current})
    name = Symbol(String(node.expr.name))
    n_pos = normalize_node(String(node.expr.pos))
    n_neg = normalize_node(String(node.expr.neg))

    dc = 0.0
    ac = 0.0 + 0.0im
    tran = nothing

    for val_item in node.expr.vals
        val_form = val_item.form
        if val_form isa SP.DCSource
            if val_form.dcval !== nothing
                try
                    dc = parse_value(String(val_form.dcval))
                catch
                end
            end
        end
    end

    isource!(ctx.circuit, n_pos, n_neg; dc=dc, ac=ac, tran=tran, name=name)
end

# SPICE Diode: Dname n+ n- modelname
function process_node!(ctx::NetlistContext, node::SNode{SP.Diode})
    name = Symbol(String(node.expr.name))
    n_anode = normalize_node(String(node.expr.pos))
    n_cathode = normalize_node(String(node.expr.neg))

    # Get model parameters if available
    model_params = get(ctx.models, Symbol(String(node.expr.model)), Dict())

    diode!(ctx.circuit, n_anode, n_cathode;
           IS=get(model_params, :IS, 1e-14),
           N=get(model_params, :N, 1.0),
           name=name)
end

# SPICE Model statement: .MODEL name type (params)
function process_node!(ctx::NetlistContext, node::SNode{SP.Model})
    name = Symbol(String(node.expr.name))
    mtype = Symbol(lowercase(String(node.expr.typ)))

    params = Dict{Symbol, Any}()
    for param in node.expr.parameters
        param_form = param.form
        if param_form.val !== nothing
            try
                pname = Symbol(uppercase(String(param_form.name)))
                pval = parse_value(String(param_form.val))
                params[pname] = pval
            catch
            end
        end
    end

    ctx.models[name] = params
end

# Spectre resistor instance
function process_node!(ctx::NetlistContext, node::SNode{SC.Instance})
    inst = node.expr
    name = Symbol(String(inst.name))

    # Get component type from master
    master = lowercase(String(inst.master))

    # Get ports
    ports = Symbol[]
    if inst.ports !== nothing
        for port in inst.ports
            push!(ports, Symbol(lowercase(String(port))))
        end
    end

    # Get parameters
    params = Dict{Symbol, Any}()
    if inst.params !== nothing
        for param in inst.params
            pname = Symbol(lowercase(String(param.name)))
            pval = parse_value(String(param.val))
            params[pname] = pval
        end
    end

    # Create device based on master type
    if master == "resistor" || master == "r"
        r = get(params, :r, get(params, :resistance, 1000.0))
        resistor!(ctx.circuit, ports[1], ports[2], r; name=name)
    elseif master == "capacitor" || master == "c"
        c = get(params, :c, get(params, :capacitance, 1e-12))
        capacitor!(ctx.circuit, ports[1], ports[2], c; name=name)
    elseif master == "inductor" || master == "l"
        l = get(params, :l, get(params, :inductance, 1e-9))
        inductor!(ctx.circuit, ports[1], ports[2], l; name=name)
    elseif master == "vsource"
        dc = get(params, :dc, 0.0)
        ac_mag = get(params, :mag, 0.0)
        vsource!(ctx.circuit, ports[1], ports[2]; dc=dc, ac=ac_mag, name=name)
    elseif master == "isource"
        dc = get(params, :dc, 0.0)
        isource!(ctx.circuit, ports[1], ports[2]; dc=dc, name=name)
    elseif master == "diode"
        diode!(ctx.circuit, ports[1], ports[2]; name=name)
    else
        @warn "Unknown device type: $master"
    end
end

"""
    spice_circuit(netlist::AbstractString; kwargs...) -> MNACircuit

Parse a SPICE netlist and build an MNA circuit.
"""
function spice_circuit(netlist::AbstractString; temp=27.0, gmin=1e-12)
    ast = parse_netlist(netlist; start_lang=:spice)
    return build_circuit_from_netlist(ast; temp=temp, gmin=gmin)
end

"""
    spectre_circuit(netlist::AbstractString; kwargs...) -> MNACircuit

Parse a Spectre netlist and build an MNA circuit.
"""
function spectre_circuit(netlist::AbstractString; temp=27.0, gmin=1e-12)
    ast = parse_netlist(netlist; start_lang=:spectre)
    return build_circuit_from_netlist(ast; temp=temp, gmin=gmin)
end
