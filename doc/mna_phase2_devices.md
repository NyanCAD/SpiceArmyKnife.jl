# Phase 2: Simple SPICE/Spectre Devices

## Overview

This phase updates the codegen (`spectre.jl`) to emit MNA primitives for built-in devices. The goal is to run existing SPICE/Spectre netlists through the new MNA backend with basic passive and source devices.

## Deliverables

1. `src/mna/devices.jl` - Device wrapper structs
2. `src/mna/circuit.jl` - Circuit builder from parsed netlist
3. Updates to `src/spectre.jl` - Emit MNA primitives
4. Updates to `src/spectre_env.jl` - New device constructors
5. `src/mna/transient.jl` - Basic transient analysis
6. Integration with existing test infrastructure

## Device Wrapper Architecture

### Device Base Interface

```julia
# src/mna/devices.jl

"""
Abstract base for all MNA devices.
"""
abstract type MNADevice end

"""
Setup phase: register nodes and branches, allocate storage.
Called once during circuit elaboration.
"""
function setup!(device::MNADevice, ctx::MNAContext) end

"""
Stamp phase: add constant matrix entries.
Called once after setup, for linear device contributions.
"""
function stamp!(device::MNADevice, ctx::MNAContext) end

"""
Evaluate phase: compute residuals and Jacobian for current state.
Called each Newton iteration (nonlinear devices only).
"""
function evaluate!(device::MNADevice, ctx::MNAContext, x::Vector{Float64}) end

"""
Check if device is linear (stamp-only, no evaluate).
"""
is_linear(::MNADevice) = false
```

### Linear Devices

#### Resistor

```julia
"""
Linear resistor between two nodes.
"""
struct MNAResistor <: MNADevice
    name::Symbol
    pos::Symbol   # Positive node name
    neg::Symbol   # Negative node name
    R::Float64    # Resistance in Ohms
end

is_linear(::MNAResistor) = true

function setup!(r::MNAResistor, ctx::MNAContext)
    # Just ensure nodes exist
    get_node!(ctx, r.pos)
    get_node!(ctx, r.neg)
end

function stamp!(r::MNAResistor, ctx::MNAContext)
    p = r.pos == :0 ? GND : ctx.nodes[r.pos]
    n = r.neg == :0 ? GND : ctx.nodes[r.neg]
    stamp_conductance!(ctx, p, n, 1.0 / r.R)
end
```

#### Capacitor

```julia
"""
Linear capacitor between two nodes.
Stamps into C matrix for transient analysis.
"""
struct MNACapacitor <: MNADevice
    name::Symbol
    pos::Symbol
    neg::Symbol
    C::Float64    # Capacitance in Farads
end

is_linear(::MNACapacitor) = true

function setup!(c::MNACapacitor, ctx::MNAContext)
    get_node!(ctx, c.pos)
    get_node!(ctx, c.neg)
end

function stamp!(c::MNACapacitor, ctx::MNAContext)
    p = c.pos == :0 ? GND : ctx.nodes[c.pos]
    n = c.neg == :0 ? GND : ctx.nodes[c.neg]
    stamp_capacitance!(ctx, p, n, c.C)
end
```

#### Inductor

```julia
"""
Inductor between two nodes.
Creates branch variable for current.
"""
struct MNAInductor <: MNADevice
    name::Symbol
    pos::Symbol
    neg::Symbol
    L::Float64    # Inductance in Henries

    # Filled during setup
    branch::Union{Nothing, Branch}
end

MNAInductor(name, pos, neg, L) = MNAInductor(name, pos, neg, L, nothing)

is_linear(::MNAInductor) = true

function setup!(l::MNAInductor, ctx::MNAContext)
    p = l.pos == :0 ? GND : get_node!(ctx, l.pos)
    n = l.neg == :0 ? GND : get_node!(ctx, l.neg)
    l.branch = create_branch!(ctx, l.name, p, n)
end

function stamp!(l::MNAInductor, ctx::MNAContext)
    stamp_inductor!(ctx, l.branch, l.L)
end
```

### Sources

#### DC Voltage Source

```julia
"""
Independent DC voltage source.
"""
struct MNAVoltageSource <: MNADevice
    name::Symbol
    pos::Symbol
    neg::Symbol
    V::Float64    # DC voltage

    branch::Union{Nothing, Branch}
end

MNAVoltageSource(name, pos, neg, V) = MNAVoltageSource(name, pos, neg, V, nothing)

is_linear(::MNAVoltageSource) = true

function setup!(v::MNAVoltageSource, ctx::MNAContext)
    p = v.pos == :0 ? GND : get_node!(ctx, v.pos)
    n = v.neg == :0 ? GND : get_node!(ctx, v.neg)
    v.branch = create_branch!(ctx, v.name, p, n)
end

function stamp!(v::MNAVoltageSource, ctx::MNAContext)
    stamp_voltage_source!(ctx, v.branch, v.V)
end
```

#### DC Current Source

```julia
"""
Independent DC current source.
"""
struct MNACurrentSource <: MNADevice
    name::Symbol
    pos::Symbol
    neg::Symbol
    I::Float64    # DC current (flows from pos to neg internally)
end

is_linear(::MNACurrentSource) = true

function setup!(i::MNACurrentSource, ctx::MNAContext)
    get_node!(ctx, i.pos)
    get_node!(ctx, i.neg)
end

function stamp!(i::MNACurrentSource, ctx::MNAContext)
    p = i.pos == :0 ? GND : ctx.nodes[i.pos]
    n = i.neg == :0 ? GND : ctx.nodes[i.neg]
    stamp_current_source!(ctx, p, n, i.I)
end
```

#### Time-Varying Sources

```julia
"""
Time-varying voltage source.
Reevaluates at each time point.
"""
struct MNAVSourceTransient <: MNADevice
    name::Symbol
    pos::Symbol
    neg::Symbol
    waveform::Function  # t -> V(t)

    branch::Union{Nothing, Branch}
    bi::Int  # Branch index (filled during setup)
end

is_linear(::MNAVSourceTransient) = false  # Needs evaluation each step

function setup!(v::MNAVSourceTransient, ctx::MNAContext)
    p = v.pos == :0 ? GND : get_node!(ctx, v.pos)
    n = v.neg == :0 ? GND : get_node!(ctx, v.neg)
    v.branch = create_branch!(ctx, v.name, p, n)
    v.bi = branch_index(ctx, v.branch)
end

function stamp!(v::MNAVSourceTransient, ctx::MNAContext)
    # Stamp the matrix structure (KCL entries)
    pi = node_index(v.branch.pos)
    ni = node_index(v.branch.neg)
    bi = v.bi

    stamp!(ctx.G_coo, pi, bi, +1.0)
    stamp!(ctx.G_coo, ni, bi, -1.0)
    stamp!(ctx.G_coo, bi, pi, +1.0)
    stamp!(ctx.G_coo, bi, ni, -1.0)
end

function evaluate!(v::MNAVSourceTransient, ctx::MNAContext, t::Float64)
    # Update RHS with current voltage value
    ctx.b[v.bi] = v.waveform(t)
end
```

### Controlled Sources

#### VCCS (G-element)

```julia
"""
Voltage-Controlled Current Source.
I(out+, out-) = gm * V(ctrl+, ctrl-)
"""
struct MNAVCCS <: MNADevice
    name::Symbol
    out_pos::Symbol
    out_neg::Symbol
    ctrl_pos::Symbol
    ctrl_neg::Symbol
    gm::Float64    # Transconductance
end

is_linear(::MNAVCCS) = true

function setup!(g::MNAVCCS, ctx::MNAContext)
    for n in [g.out_pos, g.out_neg, g.ctrl_pos, g.ctrl_neg]
        n != :0 && get_node!(ctx, n)
    end
end

function stamp!(g::MNAVCCS, ctx::MNAContext)
    op = g.out_pos == :0 ? GND : ctx.nodes[g.out_pos]
    on = g.out_neg == :0 ? GND : ctx.nodes[g.out_neg]
    cp = g.ctrl_pos == :0 ? GND : ctx.nodes[g.ctrl_pos]
    cn = g.ctrl_neg == :0 ? GND : ctx.nodes[g.ctrl_neg]
    stamp_vccs!(ctx, op, on, cp, cn, g.gm)
end
```

#### VCVS (E-element)

```julia
"""
Voltage-Controlled Voltage Source.
V(out+, out-) = gain * V(ctrl+, ctrl-)
"""
struct MNAVCVS <: MNADevice
    name::Symbol
    out_pos::Symbol
    out_neg::Symbol
    ctrl_pos::Symbol
    ctrl_neg::Symbol
    gain::Float64

    branch::Union{Nothing, Branch}
end

MNAVCVS(name, op, on, cp, cn, gain) =
    MNAVCVS(name, op, on, cp, cn, gain, nothing)

is_linear(::MNAVCVS) = true

function setup!(e::MNAVCVS, ctx::MNAContext)
    op = e.out_pos == :0 ? GND : get_node!(ctx, e.out_pos)
    on = e.out_neg == :0 ? GND : get_node!(ctx, e.out_neg)
    for n in [e.ctrl_pos, e.ctrl_neg]
        n != :0 && get_node!(ctx, n)
    end
    e.branch = create_branch!(ctx, e.name, op, on)
end

function stamp!(e::MNAVCVS, ctx::MNAContext)
    cp = e.ctrl_pos == :0 ? GND : ctx.nodes[e.ctrl_pos]
    cn = e.ctrl_neg == :0 ? GND : ctx.nodes[e.ctrl_neg]
    stamp_vcvs!(ctx, e.branch, cp, cn, e.gain)
end
```

#### CCCS (F-element)

```julia
"""
Current-Controlled Current Source.
I(out+, out-) = gain * I(ctrl_vsource)
"""
struct MNACCCS <: MNADevice
    name::Symbol
    out_pos::Symbol
    out_neg::Symbol
    ctrl_vsource::Symbol  # Name of controlling voltage source
    gain::Float64
end

is_linear(::MNACCCS) = true

function stamp!(f::MNACCCS, ctx::MNAContext)
    op = f.out_pos == :0 ? GND : ctx.nodes[f.out_pos]
    on = f.out_neg == :0 ? GND : ctx.nodes[f.out_neg]

    # Control current is the branch current of the controlling voltage source
    ctrl_branch = ctx.branches[f.ctrl_vsource]
    ctrl_bi = branch_index(ctx, ctrl_branch)

    opi = node_index(op)
    oni = node_index(on)

    # G[out+, ctrl_branch] += gain
    # G[out-, ctrl_branch] -= gain
    stamp!(ctx.G_coo, opi, ctrl_bi, +f.gain)
    stamp!(ctx.G_coo, oni, ctrl_bi, -f.gain)
end
```

#### CCVS (H-element)

```julia
"""
Current-Controlled Voltage Source.
V(out+, out-) = gain * I(ctrl_vsource)
"""
struct MNACCVS <: MNADevice
    name::Symbol
    out_pos::Symbol
    out_neg::Symbol
    ctrl_vsource::Symbol
    gain::Float64

    branch::Union{Nothing, Branch}
end

MNACCVS(name, op, on, ctrl, gain) =
    MNACCVS(name, op, on, ctrl, gain, nothing)

is_linear(::MNACCVS) = true

function setup!(h::MNACCVS, ctx::MNAContext)
    op = h.out_pos == :0 ? GND : get_node!(ctx, h.out_pos)
    on = h.out_neg == :0 ? GND : get_node!(ctx, h.out_neg)
    h.branch = create_branch!(ctx, h.name, op, on)
end

function stamp!(h::MNACCVS, ctx::MNAContext)
    ctrl_branch = ctx.branches[h.ctrl_vsource]
    ctrl_bi = branch_index(ctx, ctrl_branch)
    bi = branch_index(ctx, h.branch)

    pi = node_index(h.branch.pos)
    ni = node_index(h.branch.neg)

    # KCL
    stamp!(ctx.G_coo, pi, bi, +1.0)
    stamp!(ctx.G_coo, ni, bi, -1.0)

    # Constraint: V+ - V- - gain*I_ctrl = 0
    stamp!(ctx.G_coo, bi, pi, +1.0)
    stamp!(ctx.G_coo, bi, ni, -1.0)
    stamp!(ctx.G_coo, bi, ctrl_bi, -h.gain)
end
```

## Codegen Updates

### spectre.jl Modifications

The key file to modify is `src/spectre.jl`. The `spice_instance` function needs to emit MNA device constructors instead of DAECompiler calls.

```julia
# Updates to src/spectre.jl

# Add MNA mode flag
const USE_MNA_BACKEND = Ref(false)

# Modified device instantiation for MNA mode
function emit_mna_resistor(to_julia::SpcScope, name, pos, neg, R)
    quote
        MNAResistor($(QuoteNode(name)), $(QuoteNode(pos)),
                    $(QuoteNode(neg)), $R)
    end
end

function emit_mna_capacitor(to_julia::SpcScope, name, pos, neg, C)
    quote
        MNACapacitor($(QuoteNode(name)), $(QuoteNode(pos)),
                     $(QuoteNode(neg)), $C)
    end
end

# ... similar for other devices

# In the resistor handler (around line 998):
function (to_julia::SpcScope)(stmt::SNode{SC.Resistor})
    # ... existing parameter extraction ...

    if USE_MNA_BACKEND[]
        return emit_mna_resistor(to_julia, name, pos, neg, R)
    else
        # Existing DAECompiler code
        return existing_resistor_code(...)
    end
end
```

### spectre_env.jl Modifications

Add MNA-compatible device constructors:

```julia
# src/spectre_env.jl additions

module SpectreEnvironmentMNA

using ..MNA

"""
Create resistor for MNA backend.
"""
function resistor(; r=1e3)
    (pos, neg; name=gensym(:R)) -> MNAResistor(name, pos, neg, r)
end

"""
Create capacitor for MNA backend.
"""
function capacitor(; c=1e-12)
    (pos, neg; name=gensym(:C)) -> MNACapacitor(name, pos, neg, c)
end

"""
Create inductor for MNA backend.
"""
function inductor(; l=1e-6)
    (pos, neg; name=gensym(:L)) -> MNAInductor(name, pos, neg, l)
end

"""
Create voltage source for MNA backend.
"""
function vsource(; dc=0.0, type=:dc)
    (pos, neg; name=gensym(:V)) -> begin
        if type == :dc
            MNAVoltageSource(name, pos, neg, dc)
        else
            # Handle AC, pulse, sine, etc.
            MNAVSourceTransient(name, pos, neg, t -> dc)
        end
    end
end

"""
Create current source for MNA backend.
"""
function isource(; dc=0.0)
    (pos, neg; name=gensym(:I)) -> MNACurrentSource(name, pos, neg, dc)
end

end # module
```

## Circuit Builder

### From Device List to MNACircuit

```julia
# src/mna/circuit.jl

"""
Build MNA circuit from list of devices.
"""
function build_circuit(devices::Vector{<:MNADevice})
    ctx = MNAContext()

    # Phase 1: Setup all devices (register nodes/branches)
    for device in devices
        setup!(device, ctx)
    end

    # Phase 2: Stamp all devices
    for device in devices
        stamp!(device, ctx)
    end

    # Finalize to immutable circuit
    circuit = finalize!(ctx)

    # Store device references for nonlinear evaluation
    circuit.devices = [d for d in devices if !is_linear(d)]

    return circuit
end

"""
Integration point for parsed netlist.
"""
function circuit_from_netlist(netlist_expr)
    # Evaluate the generated expression to get device list
    devices = eval(netlist_expr)
    build_circuit(devices)
end
```

## Basic Transient Analysis

### DAEProblem Construction

```julia
# src/mna/transient.jl

using DifferentialEquations
using SparseArrays

"""
Build DAEProblem from MNA circuit.
"""
function build_dae_problem(circuit::MNACircuit, tspan::Tuple{Float64, Float64})
    n = circuit.n_vars

    # Initial conditions: DC operating point
    dc_sol = solve_dc(circuit)
    u0 = dc_sol.x
    du0 = zeros(n)  # Derivatives zero at DC

    # Residual function: f(x) + C*dx/dt = 0
    function residual!(res, du, u, p, t)
        # Linear part
        mul!(res, circuit.G, u)
        res .-= circuit.b

        # Add reactive term: C * du
        mul!(res, circuit.C, du, 1.0, 1.0)  # res += C * du

        # Nonlinear devices (if any)
        for device in circuit.devices
            evaluate!(device, res, u, t)
        end
    end

    # Jacobian: gamma * C + G
    function jacobian!(J, du, u, p, gamma, t)
        # J = gamma * C + G
        J .= gamma .* circuit.C .+ circuit.G
    end

    # Create sparse Jacobian prototype
    jac_prototype = circuit.G + circuit.C

    dae_fn = DAEFunction(residual!;
        jac = jacobian!,
        jac_prototype = jac_prototype)

    DAEProblem(dae_fn, du0, u0, tspan;
        differential_vars = circuit.differential_vars)
end

"""
Solve transient analysis.
"""
function solve_transient(circuit::MNACircuit, tspan;
                         solver = IDA(),
                         reltol = 1e-6,
                         abstol = 1e-9)
    prob = build_dae_problem(circuit, tspan)
    solve(prob, solver; reltol, abstol)
end
```

### Solution Access

```julia
"""
Extract node voltage from solution at given time.
"""
function voltage(sol, circuit::MNACircuit, node::Symbol, t)
    idx = findfirst(==(node), circuit.node_names)
    idx === nothing && error("Node $node not found")
    sol(t)[idx]
end

"""
Extract branch current from solution at given time.
"""
function current(sol, circuit::MNACircuit, branch::Symbol, t)
    idx = findfirst(==(branch), circuit.branch_names)
    idx === nothing && error("Branch $branch not found")
    sol(t)[circuit.n_nodes + idx]
end
```

## Integration with Existing Test Suite

### Adapter Layer

Create an adapter that matches the existing `solve_circuit` interface:

```julia
# src/mna/compat.jl

"""
Compatibility layer for existing tests.
Wraps MNA circuit in interface matching CircuitIRODESystem.
"""
struct MNACircuitSystem
    circuit::MNACircuit
end

"""
Match existing solve_circuit interface.
"""
function solve_circuit_mna(circuit_fn;
                           time_bounds = (0.0, 1.0),
                           reltol = 1e-6,
                           abstol = 1e-9,
                           kwargs...)
    # Build circuit from function
    devices = circuit_fn()
    circuit = build_circuit(devices)

    # Solve
    sys = MNACircuitSystem(circuit)
    sol = solve_transient(circuit, time_bounds; reltol, abstol)

    return sys, sol
end

"""
Match existing solve_spice_code interface.
"""
function solve_spice_code_mna(spice_code::String; kwargs...)
    # Parse
    ast = SpectreNetlistParser.parse(spice_code)

    # Generate with MNA backend
    USE_MNA_BACKEND[] = true
    circuit_expr = make_spectre_circuit(ast)
    USE_MNA_BACKEND[] = false

    # Build and solve
    devices = eval(circuit_expr)
    circuit = build_circuit(devices)

    solve_transient(circuit, get(kwargs, :time_bounds, (0.0, 1.0)); kwargs...)
end
```

## Test Cases

### Device Tests

```julia
# test/mna/test_devices.jl

@testset "MNA Devices" begin

    @testset "RC Circuit Transient" begin
        # V1 = 1V step, R = 1k, C = 1µF
        # τ = RC = 1ms
        # V_C(t) = 1 - exp(-t/τ)

        devices = [
            MNAVoltageSource(:V1, :in, :0, 1.0),
            MNAResistor(:R1, :in, :out, 1e3),
            MNACapacitor(:C1, :out, :0, 1e-6)
        ]

        circuit = build_circuit(devices)
        sol = solve_transient(circuit, (0.0, 5e-3))

        # Check at t = τ = 1ms
        v_out = voltage(sol, circuit, :out, 1e-3)
        @test v_out ≈ 1.0 - exp(-1) atol=1e-3  # ≈ 0.632

        # Check at t = 5τ
        v_out = voltage(sol, circuit, :out, 5e-3)
        @test v_out ≈ 1.0 atol=1e-2  # Nearly 1V
    end

    @testset "RL Circuit Transient" begin
        # V1 = 1V, R = 1k, L = 1H
        # τ = L/R = 1ms
        # I(t) = (V/R)(1 - exp(-t/τ))

        devices = [
            MNAVoltageSource(:V1, :in, :0, 1.0),
            MNAResistor(:R1, :in, :out, 1e3),
            MNAInductor(:L1, :out, :0, 1.0)
        ]

        circuit = build_circuit(devices)
        sol = solve_transient(circuit, (0.0, 5e-3))

        # Check current at t = τ
        i_L = current(sol, circuit, :L1, 1e-3)
        i_final = 1.0 / 1e3  # 1mA
        @test i_L ≈ i_final * (1 - exp(-1)) atol=1e-6
    end

    @testset "RLC Circuit" begin
        # Underdamped RLC: oscillation check

        devices = [
            MNAVoltageSource(:V1, :in, :0, 1.0),
            MNAResistor(:R1, :in, :n1, 100.0),
            MNAInductor(:L1, :n1, :n2, 1e-3),
            MNACapacitor(:C1, :n2, :0, 1e-6)
        ]

        circuit = build_circuit(devices)
        sol = solve_transient(circuit, (0.0, 1e-2))

        # Should oscillate - check that voltage goes above 1V (overshoot)
        t_samples = range(0, 1e-2, length=100)
        v_max = maximum(voltage(sol, circuit, :n2, t) for t in t_samples)
        @test v_max > 1.0  # Overshoot
    end

end
```

### SPICE Netlist Tests

```julia
@testset "SPICE Netlist Integration" begin

    @testset "Voltage Divider SPICE" begin
        spice = """
        V1 in 0 dc=10
        R1 in mid r=1k
        R2 mid 0 r=1k
        """

        sol = solve_spice_code_mna(spice)
        # Check mid voltage is 5V
        @test voltage(sol, :mid, 0.0) ≈ 5.0 atol=1e-6
    end

    @testset "RC Filter SPICE" begin
        spice = """
        V1 in 0 dc=1 type=dc
        R1 in out r=1k
        C1 out 0 c=1u
        """

        sol = solve_spice_code_mna(spice; time_bounds=(0.0, 5e-3))
        # Check RC time constant behavior
        @test voltage(sol, :out, 1e-3) ≈ 1 - exp(-1) atol=1e-2
    end

end
```

## Migration Strategy

### Gradual Backend Switch

1. Add `USE_MNA_BACKEND` flag to control code generation
2. Run tests with both backends, compare results
3. Identify failing tests, fix device implementations
4. Once parity achieved, switch default

### Test Categories

| Category | Expected Outcome |
|----------|-----------------|
| `test/basic.jl` | Should pass with MNA |
| `test/transients.jl` | Should pass with MNA |
| `test/params.jl` | Need parameter sweep support |
| `test/ac.jl` | Needs Phase 3 |
| `test/varegress.jl` | Needs Phase 3 |

## Success Criteria

Phase 2 is complete when:

1. All linear devices (R, C, L, V, I, controlled sources) working
2. DC and transient analysis produce correct results
3. RC/RL/RLC circuits match analytical solutions
4. Basic SPICE netlists parse and simulate correctly
5. At least 50% of existing test suite passes

## Next Steps

After Phase 2, proceed to Phase 3:
- Verilog-A device support (vasim.jl updates)
- Nonlinear devices (diode, MOSFET)
- AC small-signal analysis
- Full test suite compatibility
