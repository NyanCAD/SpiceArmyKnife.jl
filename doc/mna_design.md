# MNA Migration Design Document

## Goal

Replace the current simulation backend with direct MNA (Modified Nodal Analysis) for SpiceArmyKnife.jl circuit simulation.

**Why:**
- Current backend is unmaintainable and overly complex
- Need plain Julia functions for DifferentialEquations.jl
- Direct MNA gives full control over matrix structure

## Core Approach

Based on research of ngspice, VACASK, OpenVAF, and the SciML ecosystem, the migration follows these principles:

### 1. Direct Stamping for Simple Devices

For linear devices like resistors and capacitors, generate code that stamps directly into MNA matrices without creating unnecessary current variables.

**Example:** `I(p,n) <+ V(p,n)/R` in Verilog-A:
```julia
function stamp!(R::Resistor, ctx::MNAContext, p::Int, n::Int)
    G = 1.0 / R.r
    stamp_G!(ctx, p, p,  G)
    stamp_G!(ctx, p, n, -G)
    stamp_G!(ctx, n, p, -G)
    stamp_G!(ctx, n, n,  G)
end
```

No current variable needed. Just stamps.

### 2. Separate G/C/b Representation

Maintain separate matrices for analysis flexibility:
```
G: Conductance matrix (resistive Jacobian ∂f/∂x)
C: Capacitance matrix (reactive Jacobian ∂q/∂x)
b: RHS vector (independent sources)
```

This enables:
- **DC**: Solve `G*x = b`
- **AC**: Solve `(G + jωC)*x = b` with complex arithmetic
- **Transient**: ODEProblem with mass matrix C

### 3. Current Variables Only When Needed

Only allocate current variables for:
- **Voltage sources**: Voltage is constrained, current unknown
- **Inductors**: Current is a state variable (appears in ddt)

All other devices (resistors, capacitors, current sources, VCCSs) stamp directly.

### 4. AD for Nonlinear Devices

For nonlinear devices where we can't extract coefficients at codegen time, use ForwardDiff at runtime to compute the linearized Jacobian at each Newton iteration.

---

## Architecture Overview

```
Netlist (SPICE/Spectre/Verilog-A)
    ↓
Parser (SpectreNetlistParser.jl, VerilogAParser.jl)
    ↓
Codegen (spc/codegen.jl, vasim.jl) → emits stamp!() calls
    ↓
MNAContext ← collects stamps during circuit trace
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

### Key New Types

```julia
# MNA context accumulates device contributions
mutable struct MNAContext
    # Node tracking
    node_names::Vector{Symbol}
    node_to_idx::Dict{Symbol, Int}

    # Current variables (for V-sources, inductors only)
    current_names::Vector{Symbol}
    n_currents::Int

    # Sparse matrix builders (COO format during construction)
    G_I::Vector{Int}    # Row indices
    G_J::Vector{Int}    # Column indices
    G_V::Vector{Float64}  # Values

    C_I::Vector{Int}
    C_J::Vector{Int}
    C_V::Vector{Float64}

    b::Vector{Float64}  # RHS vector
end

# Stamping functions
function stamp_G!(ctx::MNAContext, i::Int, j::Int, val::Float64)
    i == 0 || j == 0 && return  # Ground node
    push!(ctx.G_I, i)
    push!(ctx.G_J, j)
    push!(ctx.G_V, val)
end
```

### Device Interface

Devices implement stamping directly:

```julia
# Linear resistor - stamps G matrix
function stamp!(R::SimpleResistor, ctx::MNAContext, p::Int, n::Int)
    G = 1.0 / R.r
    stamp_G!(ctx, p, p,  G)
    stamp_G!(ctx, p, n, -G)
    stamp_G!(ctx, n, p, -G)
    stamp_G!(ctx, n, n,  G)
end

# Linear capacitor - stamps C matrix
function stamp!(C::SimpleCapacitor, ctx::MNAContext, p::Int, n::Int)
    cap = C.capacitance
    stamp_C!(ctx, p, p,  cap)
    stamp_C!(ctx, p, n, -cap)
    stamp_C!(ctx, n, p, -cap)
    stamp_C!(ctx, n, n,  cap)
end

# Voltage source - needs current variable
function stamp!(V::VoltageSource, ctx::MNAContext, p::Int, n::Int)
    I_idx = alloc_current!(ctx, V.name)

    # KCL rows
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # Voltage equation
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)
    stamp_b!(ctx, I_idx, V.dc)
end

# Nonlinear device - evaluated at runtime
function evaluate!(D::Diode, ctx::MNAContext, p::Int, n::Int, x::Vector)
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

## Codegen Strategy

### Classifying Verilog-A Contributions

When processing `I(p,n) <+ expr` in vasim.jl, classify the contribution:

| Pattern | Classification | Generated Code |
|---------|---------------|----------------|
| `V(p,n)/R` | Linear conductance | Direct G stamp |
| `C * ddt(V(p,n))` | Linear capacitance | Direct C stamp |
| `Idc` | Current source | Direct b stamp |
| `gm * V(c,e)` | VCCS | Off-diagonal G stamps |
| `Is * (exp(V/Vt) - 1)` | Nonlinear | Runtime evaluate! |

For `V(p,n) <+ expr`, always allocate a current variable.

### Example: Resistor Codegen

**Input (resistor.va):**
```verilog
module BasicVAResistor(p, n);
parameter real R=1;
analog begin
    I(p,n) <+ V(p,n)/R;
end
endmodule
```

**Generated Julia:**
```julia
@kwdef struct BasicVAResistor
    R::Float64 = 1.0
end

function stamp!(self::BasicVAResistor, ctx::MNAContext, p::Int, n::Int)
    G = 1.0 / self.R
    stamp_G!(ctx, p, p,  G)
    stamp_G!(ctx, p, n, -G)
    stamp_G!(ctx, n, p, -G)
    stamp_G!(ctx, n, n,  G)
end
```

---

## SciML Integration

### Building the System

```julia
function build_mna_system(circuit)
    ctx = MNAContext()

    # Trace circuit to collect stamps
    trace_circuit!(ctx, circuit)

    # Build sparse matrices
    n = ctx.n_nodes + ctx.n_currents
    G = sparse(ctx.G_I, ctx.G_J, ctx.G_V, n, n)
    C = sparse(ctx.C_I, ctx.C_J, ctx.C_V, n, n)
    b = ctx.b

    return MNASystem(G, C, b, ctx.node_names)
end
```

### ODEProblem with Mass Matrix

```julia
function make_ode_problem(sys::MNASystem, tspan)
    G, C, b = sys.G, sys.C, sys.b

    function rhs!(du, u, p, t)
        # du = -G*u + b (in mass matrix form: C*du/dt = -G*u + b)
        mul!(du, G, u)
        du .*= -1
        du .+= b
    end

    # C is the mass matrix (may be singular for algebraic constraints)
    f = ODEFunction(rhs!; mass_matrix=C, jac=(J,u,p,t) -> (J .= -G))
    prob = ODEProblem(f, zeros(size(G,1)), tspan)
    return prob
end
```

### DC Operating Point

```julia
function dc!(sys::MNASystem)
    # For DC, just solve G*x = b
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
        # Solve (G + jωC)*x = b
        A = sys.G + im * ω * sys.C
        x = A \ sys.b
        push!(results, x)
    end
    return ACSolution(freqs, results, sys.node_names)
end
```

---

## Handling Tricky Cases

### Voltage Sources

Need current variable because voltage is constrained:

```julia
function stamp!(V::VoltageSource, ctx, p, n)
    I_idx = alloc_current!(ctx)

    # Current flows through source
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # V(p) - V(n) = Vdc
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)
    stamp_b!(ctx, I_idx, Vdc)
end
```

### Inductors

Current is a state variable (appears in ddt):

```julia
function stamp!(L::Inductor, ctx, p, n)
    I_idx = alloc_current!(ctx)

    # KCL
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # V = L*dI/dt → in matrix form
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)
    stamp_C!(ctx, I_idx, I_idx, -L.inductance)
end
```

### Nonlinear ddt(q) where q = q(V)

For `I(p,n) <+ ddt(q)` with nonlinear q:

```julia
function evaluate!(dev, ctx, p, n, x)
    V = x[p] - x[n]

    # Compute charge and incremental capacitance via AD
    q = dev.q_func(V)
    C_incr = ForwardDiff.derivative(dev.q_func, V)

    stamp_C!(ctx, p, p,  C_incr)
    stamp_C!(ctx, p, n, -C_incr)
    stamp_C!(ctx, n, p, -C_incr)
    stamp_C!(ctx, n, n,  C_incr)
end
```

### VCCS (Voltage-Controlled Current Source)

```julia
# I(out+,out-) <+ gm * V(in+,in-)
function stamp!(vccs, ctx, out_p, out_n, in_p, in_n)
    gm = vccs.gm
    stamp_G!(ctx, out_p, in_p,  gm)
    stamp_G!(ctx, out_p, in_n, -gm)
    stamp_G!(ctx, out_n, in_p, -gm)
    stamp_G!(ctx, out_n, in_n,  gm)
end
```

---

## Summary: When Do We Need Current Variables?

| Contribution Type | Current Variable? | Why |
|-------------------|-------------------|-----|
| `I(a,b) <+ linear(V)` | NO | Stamps directly into G |
| `I(a,b) <+ ddt(linear(V))` | NO | Stamps directly into C |
| `I(a,b) <+ constant` | NO | Stamps directly into b |
| `I(a,b) <+ nonlinear(V)` | NO | Linearize + stamp G, b |
| `V(a,b) <+ anything` | YES | Voltage constraint needs I |
| `V(a,b) <+ L*ddt(I)` | YES | Inductor, I is state |

**Key insight:** Most devices are current contributions that stamp directly. Only voltage constraints and inductors need explicit current variables.

---

## Migration Phases

### Phase 1: MNA Core
- Define `MNAContext`, stamping primitives
- DC solver for linear circuits
- Unit tests against analytical solutions

### Phase 2: Simple Devices
- Rewrite simpledevices.jl to use stamp! interface
- Resistor, capacitor, inductor
- Voltage/current sources
- Controlled sources (VCVS, VCCS)

### Phase 3: Verilog-A Codegen
- Modify vasim.jl to classify contributions
- Generate direct stamps for linear cases
- Generate evaluate! for nonlinear cases

### Phase 4: Analysis Types
- Transient via ODEProblem with mass matrix
- AC small-signal analysis
- Integration with existing test suite

---

## Files to Modify/Create

| File | Change |
|------|--------|
| New `src/mna/context.jl` | MNAContext and stamping primitives |
| New `src/mna/solve.jl` | DC, AC, transient solvers |
| `src/simpledevices.jl` | Replace branch! with stamp! |
| `src/vasim.jl` | Classify contributions, emit stamps |
| `src/spc/codegen.jl` | Emit stamp calls instead of Named(...) |
| `src/simulate_ir.jl` | Replace Net/branch! with MNA-aware versions |
