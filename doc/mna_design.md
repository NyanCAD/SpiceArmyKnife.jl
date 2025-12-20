# MNA Migration Design Document

## Goal

Replace the current simulation backend with direct MNA (Modified Nodal Analysis).

**Why:**
- Current backend is unmaintainable
- Need plain Julia functions for DifferentialEquations.jl
- Direct MNA gives explicit control over matrix structure

---

## Architecture Overview

```
Netlist (SPICE/Spectre/Verilog-A)
    ↓
Parser (SpectreNetlistParser.jl, VerilogAParser.jl)
    ↓
Codegen → emits stamp!() calls or contribution functions
    ↓
Circuit trace with MNAContext → collects stamps
    ↓
┌─────────────────────────────────────────┐
│ Analysis Dispatch                        │
├──────────┬──────────┬───────────────────┤
│ DC       │ Transient│ AC                │
│ G*x = b  │ ODEProblem│ (G+jωC)*x = b   │
└──────────┴──────────┴───────────────────┘
    ↓
DifferentialEquations.jl / LinearSolve.jl
    ↓
Solution
```

---

## What Replaces the Current Netlist API?

### Current API (to be removed)

The current system uses:
- `net(name)` → creates a Net with DAECompiler variable
- `branch!(net₊, net₋)` → creates current variable, KCL equations
- `kcl!(net, current)` → accumulates current into KCL equation
- `equation!(residual)` → adds residual equation

This is **replaced** by direct stamping during circuit trace.

### New API

```julia
mutable struct MNAContext
    # Node tracking
    node_names::Vector{Symbol}
    node_to_idx::Dict{Symbol, Int}
    n_nodes::Int

    # Current variables (only for V-sources and inductors)
    current_names::Vector{Symbol}
    n_currents::Int

    # Sparse matrix builders (COO format)
    G_I::Vector{Int}
    G_J::Vector{Int}
    G_V::Vector{Float64}

    C_I::Vector{Int}
    C_J::Vector{Int}
    C_V::Vector{Float64}

    b::Vector{Float64}
end

# Node allocation
function get_node!(ctx::MNAContext, name::Symbol)::Int
    get!(ctx.node_to_idx, name) do
        push!(ctx.node_names, name)
        ctx.n_nodes += 1
    end
end

# Current variable allocation (only for V-sources, inductors)
function alloc_current!(ctx::MNAContext, name::Symbol)::Int
    push!(ctx.current_names, name)
    ctx.n_currents += 1
    return ctx.n_nodes + ctx.n_currents
end

# Stamping primitives
function stamp_G!(ctx, i, j, val)
    (i == 0 || j == 0) && return  # Ground node
    push!(ctx.G_I, i); push!(ctx.G_J, j); push!(ctx.G_V, val)
end

function stamp_C!(ctx, i, j, val)
    (i == 0 || j == 0) && return
    push!(ctx.C_I, i); push!(ctx.C_J, j); push!(ctx.C_V, val)
end

function stamp_b!(ctx, i, val)
    i == 0 && return
    ctx.b[i] += val
end
```

### Circuit Trace

Instead of the current `circuit()` function that calls device functions with nets,
we trace the circuit and collect stamps:

```julia
function trace_circuit!(ctx::MNAContext, circuit)
    # Parse netlist, create nodes, call stamp! on each device
    # Details depend on codegen (see below)
end

function build_mna_system(circuit)
    ctx = MNAContext()
    trace_circuit!(ctx, circuit)

    n = ctx.n_nodes + ctx.n_currents
    G = sparse(ctx.G_I, ctx.G_J, ctx.G_V, n, n)
    C = sparse(ctx.C_I, ctx.C_J, ctx.C_V, n, n)

    return MNASystem(G, C, ctx.b, ctx.node_names)
end
```

---

## SPICE Code Generation

### Current Codegen (spc/codegen.jl)

Currently generates:
```julia
Named(spicecall(resistor; r=1000.0), "R1")(vcc, out)
```

Which calls `SimpleResistor(r=1000)(vcc_net, out_net)` using the `branch!` API.

### New Codegen

SPICE codegen will emit direct stamp calls:

```julia
# For: R1 vcc out 1k
stamp!(SimpleResistor(r=1000.0), ctx, get_node!(ctx, :vcc), get_node!(ctx, :out))

# For: C1 out gnd 1u
stamp!(SimpleCapacitor(c=1e-6), ctx, get_node!(ctx, :out), 0)  # 0 = ground

# For: V1 vcc gnd DC 5
stamp!(VoltageSource(dc=5.0), ctx, get_node!(ctx, :vcc), 0)
```

The circuit function becomes:
```julia
function trace_spice_circuit!(ctx::MNAContext, netlist)
    # Ground node is always 0
    # Parse netlist and emit stamp! calls
    for instance in netlist.instances
        device = create_device(instance)
        nodes = [get_node!(ctx, n) for n in instance.ports]
        stamp!(device, ctx, nodes...)
    end
end
```

---

## Verilog-A Code Generation

### Current Codegen (vasim.jl)

Currently generates device structs with a call operator that uses `branch!`:
```julia
function (self::VAResistor)(port_p, port_n; dscope=...)
    I_p_n = DAECompiler.variable(...)
    branch_value_p_n = (port_p.V - port_n.V) / self.R
    kcl!(port_p, -I_p_n)
    kcl!(port_n, I_p_n)
    DAECompiler.equation!(I_p_n - branch_value_p_n, ...)
end
```

### New Codegen

VA codegen will emit a contribution function evaluated with AD:

```julia
@kwdef struct VAResistor
    R::Float64 = 1.0
end

# Generated contribution function
function contribution(self::VAResistor, Vp, Vn)
    Vpn = Vp - Vn
    return Vpn / self.R  # May include ddt() calls
end

# Stamping uses the general AD-based approach
function stamp!(self::VAResistor, ctx::MNAContext, p::Int, n::Int, x::Vector)
    contrib = V -> contribution(self, V[p], V[n])
    stamp_current_contribution!(ctx, p, n, contrib, x)
end
```

For devices with `ddt()`, the s-dual approach automatically separates
resistive and reactive contributions. See `mna_ad_stamping.md`.

---

## Simple Device Implementations

These replace the current `simpledevices.jl` implementations:

### Resistor

```julia
function stamp!(R::SimpleResistor, ctx::MNAContext, p::Int, n::Int)
    G = 1.0 / R.r
    stamp_G!(ctx, p, p,  G)
    stamp_G!(ctx, p, n, -G)
    stamp_G!(ctx, n, p, -G)
    stamp_G!(ctx, n, n,  G)
end
```

### Capacitor

```julia
function stamp!(C::SimpleCapacitor, ctx::MNAContext, p::Int, n::Int)
    cap = C.capacitance
    stamp_C!(ctx, p, p,  cap)
    stamp_C!(ctx, p, n, -cap)
    stamp_C!(ctx, n, p, -cap)
    stamp_C!(ctx, n, n,  cap)
end
```

### Voltage Source (needs current variable)

```julia
function stamp!(V::VoltageSource, ctx::MNAContext, p::Int, n::Int)
    I_idx = alloc_current!(ctx, V.name)

    # KCL: current flows through
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # Voltage equation: V(p) - V(n) = Vdc
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)
    stamp_b!(ctx, I_idx, V.dc)
end
```

### Inductor (needs current variable)

```julia
function stamp!(L::SimpleInductor, ctx::MNAContext, p::Int, n::Int)
    I_idx = alloc_current!(ctx, L.name)

    # KCL
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # V = L*dI/dt
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)
    stamp_C!(ctx, I_idx, I_idx, -L.inductance)
end
```

### Nonlinear Devices

For nonlinear devices, `stamp!` takes the solution vector and computes
linearized stamps at each Newton iteration:

```julia
function stamp!(D::Diode, ctx::MNAContext, p::Int, n::Int, x::Vector)
    V = x[p] - x[n]

    # Compute current and conductance
    I = D.Is * (exp(V / D.Vt) - 1)
    G = D.Is / D.Vt * exp(V / D.Vt)

    # Equivalent source for Newton
    Ieq = I - G * V

    stamp_G!(ctx, p, p,  G)
    stamp_G!(ctx, p, n, -G)
    stamp_G!(ctx, n, p, -G)
    stamp_G!(ctx, n, n,  G)
    stamp_b!(ctx, p, -Ieq)
    stamp_b!(ctx, n,  Ieq)
end
```

---

## AD-Based Stamping for General VA

For arbitrary VA contributions with mixed resistive/reactive terms,
use ForwardDiff with the Laplace variable `s` as a Dual:

```julia
using ForwardDiff: Dual, value, partials

const s = Dual(0.0, 1.0)
ddt(x) = s * x

# Evaluating: I(p,n) <+ V/R + C*ddt(V)
# Result: Dual(V/R, C*V)
# - value() = V/R → resistive, stamps into G
# - partials() = C*V → charge q, stamps into C via ∂q/∂V
```

See `mna_ad_stamping.md` for the complete approach with nested duals
for Jacobian computation.

---

## When Do We Need Current Variables?

| Contribution Type | Current Variable? | Why |
|-------------------|-------------------|-----|
| `I(a,b) <+ f(V)` | NO | Stamps directly into G |
| `I(a,b) <+ ddt(q(V))` | NO | Stamps directly into C |
| `I(a,b) <+ constant` | NO | Stamps directly into b |
| `V(a,b) <+ anything` | YES | Voltage constraint needs I |
| Inductor | YES | Current is state variable |

---

## SciML Integration

### ODEProblem with Mass Matrix

```julia
function make_ode_problem(sys::MNASystem, tspan)
    G, C, b = sys.G, sys.C, sys.b

    function rhs!(du, u, p, t)
        # C*du/dt = -G*u + b
        mul!(du, G, u)
        du .*= -1
        du .+= b
    end

    f = ODEFunction(rhs!; mass_matrix=C, jac=(J,u,p,t) -> (J .= -G))
    return ODEProblem(f, zeros(size(G,1)), tspan)
end
```

### DC Operating Point

```julia
function dc!(sys::MNASystem)
    x = sys.G \ sys.b
    return DCSolution(x, sys.node_names)
end
```

### AC Analysis

```julia
function ac!(sys::MNASystem, freqs)
    results = Vector{ComplexF64}[]
    for f in freqs
        ω = 2π * f
        A = sys.G + im * ω * sys.C
        x = A \ sys.b
        push!(results, x)
    end
    return ACSolution(freqs, results, sys.node_names)
end
```

---

## Files to Modify/Create

| File | Change |
|------|--------|
| New `src/mna/context.jl` | MNAContext and stamping primitives |
| New `src/mna/solve.jl` | DC, AC, transient solvers |
| `src/simpledevices.jl` | Replace `(::Device)(A, B)` with `stamp!(device, ctx, a, b)` |
| `src/vasim.jl` | Emit contribution functions instead of branch!/kcl! code |
| `src/spc/codegen.jl` | Emit `stamp!` calls instead of `Named(...)` |
| `src/simulate_ir.jl` | Remove Net/branch!/kcl! infrastructure |
