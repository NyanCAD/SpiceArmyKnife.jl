# MNA Migration Design Document

## Goal

Replace DAECompiler (symbolic DAE compiler) with direct MNA (Modified Nodal Analysis) for SpiceArmyKnife.jl circuit simulation.

**Why:**
- DAECompiler is unmaintainable
- MTK symbolic equations grow exponentially for complex circuits
- Need plain Julia functions for DifferentialEquations.jl

## What Went Wrong (Previous Attempt)

A Claude Code session attempted this migration with 115 commits (+7400/-2500 lines). Result: ~10% functionality retained.

**The core mistake: Complete rewrite instead of incremental update.**

- Created new `mna.jl` module disconnected from existing infrastructure
- Parser pipeline (`make_spectre_circuit()`) left pointing to old API
- Old `solve_circuit()`, `solve_spice_code()` functions broken
- 82 "passing" tests were mostly new simple tests, not the original suite

**The goal is still to update existing code, not rewrite it.** The full test suite should pass with minor tweaks. But an adapter layer converting DAECompiler circuits to MNA doesn't make sense—the abstractions are too different. Instead: define new MNA primitives, test them directly, then update the codegen to target them.

## Lesson from the Failed Attempt

The failed attempt validated that MNA works in principle, but took shortcuts on the abstractions. The key insight:

**Design primitives that strike a good balance between:**
1. Being easy to target from the existing codegen (vasim.jl, spectre.jl)
2. Being easy to translate to DifferentialEquations.jl functions (ODEProblem, DAEProblem)

Get the primitives right first. The code will follow.

## The Existing Pipeline

```
Netlist (SPICE/Spectre)
    ↓
Parser (SpectreNetlistParser.jl)
    ↓
make_spectre_circuit() → generates Julia code
    ↓
CircuitIRODESystem [DAECompiler] ← THIS needs replacing
    ↓
DifferentialEquations.jl
    ↓
Solution
```

The parser works fine. Only the backend (`CircuitIRODESystem`) needs replacement.

---

## Technical Design

### Verilog-A to MNA Mapping

| Verilog-A | MNA Action |
|-----------|------------|
| `I(p,n) <+ resistive_expr` | Stamp into G matrix and b vector |
| `I(p,n) <+ ddt(Q_expr)` | Stamp into C matrix (reactive) |
| `V(p,n) <+ expr` | Create branch variable + constraint equation |
| `ddx(expr, V(n))` | ForwardDiff partials |

Devices stamp into G, C, and b separately. Mixed contributions like `I <+ resistive + ddt(Q)` separate naturally via the t-partial Dual approach.

**Voltage contributions** require a branch current variable (like a voltage source in MNA), a constraint equation `V(p) - V(n) = expr`, and the branch current participates in KCL at both nodes.

**Sign conventions:** MNA uses "current leaving node." SPICE uses "current into positive terminal." Several bugs in the failed attempt were sign errors.

### Time Integration: Let the Solver Handle It

Don't use companion models (`I = (Q - Q_prev)/dt`). DifferentialEquations.jl solvers:
- Can reject steps and step backwards
- Use higher-order methods (not simple 1/dt)
- Handle variable timesteps internally

The primitives should express `dQ/dt` contributions, not discretized approximations. The solver handles the rest.

### Detecting and Handling ddt() Contributions

**1. Trace-time detection via Dual numbers**

Rather than walking the AST to find `ddt()` calls, seed a t-partial in the Dual number. When `ddt()` is called, it appears in the partials. This happens once at setup/trace time to discover the equation structure—not every timestep. The equations are fixed after tracing.

This also handles mixed contributions like `I <+ resistive + ddt(Q)` naturally: the resistive part has no t-partial, the reactive part does.

**2. Compute ∂Q/∂V via ForwardDiff**

For `I(p,n) <+ ddt(Q_expr)` where Q depends on voltages, ForwardDiff computes the charge Jacobian needed by the DAE solver:

```julia
dQ_dV = ForwardDiff.gradient(V -> Q_expr(V), V_current)
```

This is used in the DAE residual Jacobian computation.

**3. Constant folding for unswept parameters**

The current code is careful to let Julia's compiler constant-fold parameters that aren't being swept. This is a significant performance feature. The new design should preserve this: if a parameter is known at compile time, it should fold into the generated code, avoiding runtime lookups and enabling further optimizations.

### Voltage-Dependent Capacitance: Formulation Options

For BSIM and other MOSFET models, charges (Qgs, Qgd, Qgb) are **nonlinear functions of terminal voltages**.

**Two formulation options:**

1. **Fully implicit DAE (`DAEProblem`)** - Robust, works for any circuit, limited to DAE solvers (IDA, DFBDF)
2. **Mass matrix (`ODEProblem` with `mass_matrix`)** - Opens up ODE solvers (Rosenbrock, SDIRK, etc.), works well for linear circuits or circuits with constant capacitances

With separate G and C matrices, both formulations use the same stamped data:
- **Mass matrix**: `C * dV/dt = -G*V + b` → `ODEProblem` with `mass_matrix=C`
- **Implicit DAE**: `0 = G*V + C*dV/dt - b` → `DAEProblem`

The choice becomes a solver-time decision rather than structural.

### Matrix Structure: Keep G and C Separate

For flexibility across analysis types, maintain separate matrices rather than combining everything into one:

```
(G + sC) * x = b

G: Conductance matrix (resistive, DC contributions)
C: Capacitance matrix (reactive contributions)
b: RHS vector (sources, nonlinear contributions)
```

This enables:
- **DC**: Use G only (reactive terms zero)
- **AC**: Use G + jωC directly with complex arithmetic
- **Transient**: Integration schemes combine G, C differently (e.g., G + C/dt for backward Euler)
- **Noise**: Separate access to noise source contributions
- **Sensitivity**: Compute ∂/∂parameter for individual matrices
- **Future analyses**: Harmonic balance, shooting methods, etc.

Devices stamp into G, C, and b separately. The analysis routine decides how to combine them.

### Codegen: Targeting the New Primitives

The existing vasim.jl already uses ForwardDiff Duals for `ddx()` (AC small-signal). The codegen needs to emit calls to the new MNA primitives:

**What codegen emits today:**
- `I(p,n) <+ expr` → DAECompiler contribution
- `ddt(Q)` → `DAECompiler.ddt()`
- `ddx(expr, V(n))` → ForwardDiff partials

**What codegen should emit:**
- `I(p,n) <+ expr` → stamp into G, b via primitive
- `I(p,n) <+ ddt(Q)` → stamp into C via primitive  
- `V(p,n) <+ expr` → branch equation via primitive
- `ddx()` → unchanged (already ForwardDiff)

The primitives are the interface contract between codegen and the solver backend.

**Historical note:** CedarSim once had a `DynCircuitDAESystem` that used CassetteOverlay.jl to generate DAEs. CassetteOverlay allows "overdubbing" functions with different behavior—the Dyn system maintained state on the overlay object so calls like `equation!()` would store values in the residual. This could also enable different behavior for different analysis types (DC vs transient vs AC) via overdubbing.

CassetteOverlay is a powerful tool but adds complexity. A simple API of regular functions is preferable if it suffices. But if the primitive interface needs to support radically different behavior per analysis type, overdubbing may be worth revisiting.

### Constraints and Edge Cases

- **Inductors:** `V(p,n) <+ ddt(Phi)` stamps KVL equation, not KCL
- **Ground node:** Index 0 needs special handling in stamping

---

## Migration Path

### Phase 1: MNA Core and Primitives

Define the new MNA primitives and core stamping logic. Write Julia-level unit tests:
- Matrix stamping for R, C, V/I sources
- DC solve for simple circuits (voltage divider, current source)
- Verify against analytical solutions

### Phase 2: Simple SPICE/Spectre Devices

Update codegen to emit MNA primitives for built-in devices:
- Resistor, capacitor, inductor
- Voltage and current sources
- Controlled sources (VCVS, VCCS, etc.)

Test with SPICE/Spectre netlists using these simple devices.

### Phase 3: Verilog-A Models (Incremental Complexity)

Update VA codegen (`vasim.jl`) to emit MNA primitives:
1. varesistor (pure conductance)
2. vacap (reactive, ddt)
3. vavsource (voltage contribution)
4. vadiode (nonlinear)
5. bsimcmg (everything together)

### Phase 4: Transient Analysis

Implement transient analysis:
- Start with linear RC/RLC circuits, verify against analytical solutions
- Add nonlinear devices
- Verify charge conservation
- Both implicit DAE and mass matrix formulations are options to explore

### Phase 5: Other Analysis Types

1. AC small-signal (linearization around DC point)
2. Noise, sensitivity (later)

---

## Summary

| Avoid | Prefer |
|-------|--------|
| Complete rewrite all at once | Incremental: core → simple devices → complex VA |
| Companion models for ddt() | Let solver handle time integration |
| Track Q_prev manually | Let solver handle state |
| FiniteDiff for Jacobians | ForwardDiff |
| AST walking for ddt detection | Trace-time Dual with t-partial |
| Lose constant folding | Preserve parameter folding for unswept params |
| Combine G and C into one matrix | Keep separate for analysis flexibility |

**On formulation:** Both implicit DAE and mass matrix are viable. Mass matrix gives more solver options for circuits where it works.

**The key:** Design primitives that are easy to target from codegen and easy to translate to DifferentialEquations.jl. Build incrementally: core primitives first, then simple devices, then complex VA models.
