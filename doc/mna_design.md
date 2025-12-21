# MNA Migration Design Document

## Goal

Replace the current simulation backend with direct MNA (Modified Nodal Analysis).

**Why:**
- Current backend is unmaintainable
- Need plain Julia functions for DifferentialEquations.jl
- Direct MNA gives explicit control over matrix structure

---

## Key Principles

| Avoid | Prefer |
|-------|--------|
| Complete rewrite all at once | Incremental: core → simple devices → complex VA |
| Companion models for ddt (`I = (Q-Q_prev)/dt`) | Let DiffEq solver handle time integration |
| AST walking to detect ddt() | ForwardDiff s-dual approach |
| Combining G and C into one matrix | Keep separate for analysis flexibility |
| Losing constant folding | Preserve parameter folding for unswept params |

**Sign convention:** MNA uses "current leaving node." SPICE uses "current into positive terminal." Be careful with signs in stamp! implementations.

**Let the solver handle time integration:** Don't discretize ddt() yourself. DifferentialEquations.jl solvers can reject steps, use higher-order methods, and handle variable timesteps. Express `dQ/dt` contributions and let the solver do the rest.

**Preserve constant folding:** The current system achieves good performance by letting Julia constant-fold device parameters at compile time. Device structs are typed, so `self.R` is known when the device is concrete. ParamLens uses a NamedTuple typed on its keys, so non-overridden parameters still fold while overridden ones flow through the `p` argument. The stamp!-based approach must preserve this - stamp! should be specialized per device type to enable inlining.

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

### Formulation Options

With separate G and C matrices, we can choose the formulation at solve time:

| Formulation | Pros | Cons | Best For |
|-------------|------|------|----------|
| Mass matrix ODE | More solver options (Rosenbrock, SDIRK) | Unstable with state-dependent mass matrix | Linear circuits, constant C |
| Implicit DAE | Robust for any circuit | Limited to DAE solvers (IDA, DFBDF) | Nonlinear capacitors, BSIM models |

**The issue:** For voltage-dependent capacitors like MOSFET Cgs/Cgd, `C(V) = ∂q/∂V` changes with state. State-dependent mass matrices can cause solver instabilities.

**Recommendation:** Start with mass matrix for simple circuits, fall back to DAEProblem for circuits with nonlinear capacitors.

### ODEProblem with Mass Matrix (constant C)

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

### DAEProblem (nonlinear capacitors)

```julia
function make_dae_problem(sys::MNASystem, tspan, stamp_devices!)
    function residual!(res, du, u, p, t)
        # Rebuild matrices at current operating point
        ctx = MNAContext()
        stamp_devices!(ctx, u)  # Stamps depend on u for nonlinear devices
        G, C, b = build_matrices(ctx)

        # Residual: G*u + C*du/dt - b = 0
        mul!(res, G, u)
        mul!(res, C, du, 1.0, 1.0)  # res += C*du
        res .-= b
    end

    return DAEProblem(residual!, du0, u0, tspan)
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

## Phased Implementation

Each phase must be verified before proceeding. No phase should exceed ~500 LOC.
Verify against analytical solutions and ngspice where applicable.

### Phase 0: Dependency Cleanup

**Goal:** Clean baseline on Julia 1.12 with minimal dependencies. Simulation won't work yet, but parsing/codegen should.

**Tasks:**
- Aggressively prune unused dependencies from Project.toml
- Update remaining dependencies to latest versions
- Comment out DAECompiler compile-time dependencies (imports, using statements)
- Fix any segfaults or compatibility issues from updates

**Tests that should pass (parsing/codegen only, no simulation):**
- `spectre_expr.jl` - Spectre expression parsing
- `params.jl` - ParamObserver, get_default_parameterization, nest/flatten param lists
- `sweep.jl` - `nest_param_list`, `flatten_param_list` (the non-simulation parts)
- `binning/bins.jl` - SPICE file parsing
- `sky130/parse_unified.jl` - SKY130 model parsing
- Portions of `basic.jl` that just test parsing (e.g., SPICENetlistParser.parse)

**Exit criteria:**
- Julia 1.12 loads the package without segfaults
- Parsing tests pass (may need temporary test subset)
- Minimal dependency set documented

---

### Phase 1: MNA Core (~200 LOC)

**Goal:** Standalone MNA context and stamping primitives, tested in isolation.

**Files:**
- Create `src/mna/context.jl` - MNAContext struct, stamp_G!, stamp_C!, stamp_b!
- Create `src/mna/build.jl` - build sparse matrices from COO

**New tests:** `test/mna/core.jl`
```julia
# Test: voltage divider (5V, 1k/1k) → V(out) = 2.5V
ctx = MNAContext()
vcc = get_node!(ctx, :vcc); out = get_node!(ctx, :out)
stamp!(VoltageSource(5.0), ctx, vcc, 0)
stamp!(Resistor(1000.0), ctx, vcc, out)
stamp!(Resistor(1000.0), ctx, out, 0)
x = solve_dc(ctx)
@test x[out] ≈ 2.5
```

**Exit criteria:**
- Voltage divider: V(out) = 2.5V ✓
- RC circuit stamps produce correct G and C matrix structure

---

### Phase 2: Simple Device Stamps (~200 LOC)

**Goal:** stamp! methods for basic devices.

**Files:**
- Create stamp! methods (can be in `src/mna/devices.jl` or `src/simpledevices.jl`)

**New tests:** `test/mna/devices.jl`
```julia
# Resistor stamps 4 entries into G
# Capacitor stamps 4 entries into C
# VoltageSource adds current variable, stamps G and b
# Inductor adds current variable, stamps G and C
```

**Exit criteria:**
- stamp! works for: Resistor, Capacitor, Inductor, VoltageSource, CurrentSource
- Matrix patterns match textbook MNA

---

### Phase 3: DC & Transient Solvers (~300 LOC)

**Goal:** Working DC and transient analysis.

**Files:**
- Create `src/mna/solve.jl` - dc(), transient()

**New tests:** `test/mna/solve.jl` - analytical solutions:
```julia
# DC: voltage dividers, current sources
# Transient RC: V(t) = V0 * (1 - exp(-t/RC))
# Transient RLC: oscillation at f = 1/(2π√LC)
```

**Existing tests to validate (subset, hand-built circuits):**
- `basic.jl`: "Simple VR Circuit", "Simple IR circuit", "Simple VRC circuit"
- `transients.jl`: "Butterworth Filter" (RLC analytical solution)

**Exit criteria:**
- DC matches hand calculations
- RC step response within 1% of analytical
- RLC oscillation frequency within 1%

---

### Phase 4: SPICE Codegen (~300 LOC)

**Goal:** Update spc/codegen.jl to emit stamp! calls.

**Files:**
- Modify `src/spc/codegen.jl`
- Update `src/spc/interface.jl` as needed

**Existing tests that must pass:**
- `basic.jl`: "Simple Spectre sources", "Simple SPICE sources"
- `basic.jl`: "Simple Spectre subcircuit", "SPICE include .LIB"
- `basic.jl`: "SPICE parameter scope", "multiplicities"
- `basic.jl`: "units and magnitudes", "functions", "ifelse"
- `transients.jl`: "PWL" (SPICE PWL source)
- `compilation.jl`: "Subcircuit net naming conflict"
- `alias.jl`: net aliasing

**Exit criteria:**
- Basic SPICE netlists (R, C, L, V, I) parse and solve correctly
- Results match ngspice for same netlists
- All listed existing tests pass

---

### Phase 5: VA Contribution Functions (~400 LOC)

**Goal:** Update vasim.jl to emit contribution functions with s-dual ddt().

**Files:**
- Modify `src/vasim.jl` codegen
- Add s-dual ddt() in `src/mna/contrib.jl`

**Existing tests that must pass:**
- `basic.jl`: "Verilog include"
- `ddx.jl`: ddx() functionality
- `varegress.jl`: VA resistor with reversed ports

**New tests:** Simple VA models validated against ngspice:
1. varesistor.va - DC I-V curve
2. vacap.va - AC impedance
3. vadiode.va - DC I-V curve

**Exit criteria:**
- Simple VA models match ngspice results
- All listed existing tests pass

---

### Phase 6: Complex VA & DAE (~400 LOC)

**Goal:** Full VA support including nonlinear capacitors.

**Files:**
- DAEProblem formulation in `src/mna/solve.jl`
- Handle voltage-dependent capacitance

**Existing tests that must pass:**
- `inverter.jl`: GF180 BSIM4 inverter transient
- `bsimcmg/inverter.jl`: BSIMCMG inverter
- `bsimcmg/demo_bsimcmg.jl`: BSIMCMG demo
- `ac.jl`: AC analysis (linear + BSIM inverter)

**Exit criteria:**
- BSIM DC within tolerance of ngspice
- Inverter transient waveforms match expected behavior
- AC frequency response correct

---

### Phase 7: Advanced Features (~300 LOC)

**Goal:** ParamSim, sweeps, sensitivity.

**Existing tests that must pass:**
- `params.jl`: ParamSim, ParamLens, ParamObserver
- `sweep.jl`: Sweep, ProductSweep, CircuitSweep, dc! sweeps
- `sensitivity.jl`: Forward sensitivity analysis

**Exit criteria:**
- Parameter sweeping works
- Sensitivity analysis produces correct gradients

---

### Phase 8: Cleanup

**Goal:** Remove dead code.

**Files:**
- Remove branch!/kcl!/equation! from `src/simulate_ir.jl`
- Remove DAECompiler dependency
- Clean up unused code paths

**All tests must pass:**
- `runtests.jl` complete suite

**Exit criteria:**
- No DAECompiler imports
- Full test suite passes

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
