# MNA Migration Design Document

## Goal

Replace DAECompiler with direct MNA (Modified Nodal Analysis) for SpiceArmyKnife.jl circuit simulation.

**Why:**
- DAECompiler is unmaintainable and complex compiler code
- MTK symbolic equations grow exponentially for complex circuits
- Need plain Julia functions for DifferentialEquations.jl

## Core Approach

Based on research of ngspice, VACASK, DAECompiler, OpenVAF, and the SciML ecosystem, the migration follows these principles:

### 1. Separate Evaluation from Loading (VACASK Pattern)

Devices compute contributions in an **eval** phase, then **load** into matrices. This separation enables:
- Instance bypass for unchanged devices (optimization)
- Different loading for different analysis types
- Clean device interface

### 2. Explicit DAE Formulation

Express the circuit as:
```
f(x) + d/dt[q(x)] = 0

where:
  x     = state vector (node voltages + branch currents)
  f(x)  = resistive residual (DC, algebraic)
  q(x)  = reactive residual (charges, fluxes)
```

This maps directly to SciML's `DAEProblem` or `ODEProblem` with mass matrix.

### 3. Let the Solver Handle Time Integration

Unlike ngspice's companion models (`I = C*(V - V_prev)/dt`), we express `dq/dt` directly and let DifferentialEquations.jl's adaptive solvers handle:
- Step size selection
- Higher-order methods (not just trapezoidal/Gear)
- Step rejection and retry
- Stiffness detection

### 4. Separate G/C/b Representation

Maintain separate matrices for analysis flexibility:
```
G: Conductance matrix (resistive Jacobian ∂f/∂x)
C: Capacitance matrix (reactive Jacobian ∂q/∂x)
b: RHS vector (independent sources)
```

This enables:
- **DC**: Solve `G*x = b` (C terms zero, q=0 at steady state)
- **AC**: Solve `(G + jωC)*x = b` with complex arithmetic
- **Transient**: `DAEProblem` with `0 = f(x) + dq/dt`
- **Noise/Sensitivity**: Access individual matrix contributions

### 5. Trace-Time Structure Discovery

Use ForwardDiff Dual numbers at setup time to discover:
- Which contributions are resistive vs reactive (t-partial tagging)
- Jacobian sparsity pattern
- Charge dependencies on voltages

This happens once during circuit compilation, not every timestep.

### 6. Preserve Constant Folding

Parameters not being swept should fold into generated code at compile time. The device interface must support this—unswept parameters become compile-time constants.

---

## Architecture Overview

```
Netlist (SPICE/Spectre/Verilog-A)
    ↓
Parser (SpectreNetlistParser.jl, VerilogAParser.jl)
    ↓
Codegen (spectre.jl, vasim.jl) → emits MNA primitives
    ↓
MNAContext (new) ← collects device contributions
    ↓
┌─────────────────────────────────────────┐
│ Analysis Dispatch                        │
├──────────┬──────────┬───────────────────┤
│ DC       │ Transient│ AC                │
│ G*x = b  │ DAEProblem│ (G+jωC)*x = b   │
└──────────┴──────────┴───────────────────┘
    ↓
DifferentialEquations.jl / LinearSolve.jl
    ↓
Solution
```

### Key New Types

```julia
# MNA context accumulates device contributions
struct MNAContext
    nodes::Vector{Node}           # Circuit nodes
    branches::Vector{Branch}      # Voltage source branches

    # Sparse matrix builders (COO format during construction)
    G_entries::Vector{MatrixEntry}  # Resistive Jacobian
    C_entries::Vector{MatrixEntry}  # Reactive Jacobian
    b_entries::Vector{RHSEntry}     # RHS contributions
end

# Device contribution interface
struct DeviceContribution
    resist_residual::Float64      # f(x) contribution
    react_residual::Float64       # q(x) contribution
    resist_jacobian::SparseEntries  # ∂f/∂x
    react_jacobian::SparseEntries   # ∂q/∂x
end
```

### Device Interface

Devices implement a two-phase protocol:

```julia
# Phase 1: Setup (once per elaboration)
function setup!(device, ctx::MNAContext)
    # Register nodes, create branch variables
    # Return indices for stamping
end

# Phase 2: Evaluate (each Newton iteration)
function evaluate!(device, ctx::MNAContext, x::Vector)
    # Compute residuals and Jacobian entries
    # Stamp into ctx.G, ctx.C, ctx.b
end
```

---

## SciML Integration

### DAEProblem Formulation

For a circuit with `n` node voltages and `m` branch currents:

```julia
function circuit_residual!(res, du, u, p, t)
    # u = [V_nodes..., I_branches...]
    # du = d/dt[u]

    # Zero residuals
    res .= 0.0

    # Evaluate all devices, accumulate into res
    for device in circuit.devices
        evaluate!(device, res, du, u, p, t)
    end
end

# Jacobian: γ * ∂G/∂(du) + ∂G/∂u = γ*C + G
function circuit_jacobian!(J, du, u, p, gamma, t)
    J .= gamma .* C .+ G
end

prob = DAEProblem(circuit_residual!, du0, u0, tspan;
    differential_vars = diff_vars,  # Which vars have du terms
    jac = circuit_jacobian!)

sol = solve(prob, IDA())
```

### ODEProblem with Mass Matrix (Alternative)

For circuits with constant capacitances:

```julia
function circuit_rhs!(du, u, p, t)
    # du here is the RHS, not derivative
    du .= -G * u .+ b
end

# Mass matrix: C * (actual du/dt) = rhs
# Singular where algebraic constraints exist
mass_matrix = build_C_matrix(circuit)

f = ODEFunction(circuit_rhs!; mass_matrix, jac = -G)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Rodas5())
```

### Jacobian Specification

Use sparse Jacobian with precomputed sparsity:

```julia
jac_prototype = build_sparsity_pattern(circuit)  # Sparse matrix

dae_fn = DAEFunction(circuit_residual!;
    jac = circuit_jacobian!,
    jac_prototype = jac_prototype)
```

---

## Migration Phases

### Phase 1: MNA Core and Primitives
**See: [mna_phase1_core.md](mna_phase1_core.md)**

- Define `MNAContext`, `Node`, `Branch` types
- Implement matrix stamping primitives
- DC solver for linear circuits
- Unit tests against analytical solutions

### Phase 2: Simple SPICE/Spectre Devices
**See: [mna_phase2_devices.md](mna_phase2_devices.md)**

- Update `spectre.jl` codegen for built-in devices
- Resistor, capacitor, inductor
- Voltage/current sources
- Controlled sources (VCVS, VCCS, CCVS, CCCS)
- Integration with existing test suite

### Phase 3: Verilog-A and Analysis Types
**See: [mna_phase3_verilog_a.md](mna_phase3_verilog_a.md)**

- Update `vasim.jl` for Verilog-A devices
- Transient analysis with DAEProblem
- AC small-signal analysis
- Nonlinear devices (diode, MOSFET)
- BSIM model support

---

## Key Lessons from Research

### From VACASK
- Residual-based convergence (not solution-based) is more accurate
- Eval/load separation enables optimization
- Instance bypass for unchanged devices
- Proper MNA sign conventions critical

### From ngspice
- Companion models work but limit solver flexibility
- State array management for multi-step methods
- Voltage limiting for convergence (pn junction limiter)
- GMIN stepping and source stepping for difficult convergence

### From DAECompiler
- Structural analysis identifies differential vs algebraic variables
- Automatic differentiation for Jacobians
- Scope system for hierarchical naming
- Pain point: restricted Julia subset, complex compilation

### From OpenVAF
- Clean separation of Verilog-A semantics from simulator backend
- Explicit `f(x) + ddt(q(x)) = 0` formulation
- Automatic differentiation for `ddx()` small-signal
- Parameter caching for expensive computations

### From SciML
- `DAEProblem` with `differential_vars` for mixed DAE
- IDA solver is gold standard for stiff DAEs
- Sparse Jacobian via `jac_prototype`
- Initialization via `BrownFullBasicInit` or custom

---

## Refined Approach: Minimal Changes Based on Code Tracing

**See: [implementation_trace_notes.md](implementation_trace_notes.md)** for detailed code path analysis.

### Key Discovery: Interception Points

After tracing through the codebase, the cleanest approach intercepts at two levels:

1. **`simulate_ir.jl`**: The `Net` and `branch!` infrastructure
2. **`circuitodesystem.jl`**: The `CircuitIRODESystem` entry point

**Critical insight:** Device code doesn't need to change. The `Net` struct and `branch!` function are the aggregation points where all device contributions flow.

### Implementation Strategy

#### Step 1: Dual-Mode Net and branch!

Add MNA collection alongside existing DAECompiler path:

```julia
# simulate_ir.jl - Modified Net constructor
function Net(name::AbstractScope, multiplier::Float64 = 1.0)
    # Always create DAECompiler variables (for compatibility)
    V = variable(name)
    kcl! = equation(name)
    dVdt = ddt(V)

    # Also record for MNA if collector is active
    if mna_collector[] !== nothing
        register_node!(mna_collector[], name)
    end

    return new{typeof(dVdt)}(V, kcl!, multiplier)
end

# Modified branch! to record branch info
function branch!(scope::AbstractScope, net₊::AbstractNet, net₋::AbstractNet)
    I = variable(scope(:I))
    kcl!(net₊, -I)
    kcl!(net₋,  I)
    V = net₊.V - net₋.V
    observed!(V, scope(:V))

    # Record for MNA if collector is active
    if mna_collector[] !== nothing
        record_branch!(mna_collector[], scope, net₊, net₋)
    end

    (V, I)
end
```

#### Step 2: Stamp Collection During Trace

Device equations use `equation!()` which we intercept:

```julia
# Add MNA-aware equation! wrapper
function mna_equation!(val, scope)
    # Original DAECompiler path
    DAECompiler.equation!(val, scope)

    # If MNA collection active, record the contribution
    if mna_collector[] !== nothing
        record_equation!(mna_collector[], val, scope)
    end
end
```

#### Step 3: Alternative System Constructor

```julia
function CircuitMNASystem(circuit::AbstractSim; kwargs...)
    # Create collector
    collector = MNACollector()

    # Trace the circuit with collector active
    with(mna_collector => collector) do
        circuit()
    end

    # Build MNA system from collected stamps
    return build_mna_system(collector)
end
```

### What Stays the Same

- **Device implementations** (`simpledevices.jl`): No changes needed
- **Verilog-A codegen** (`vasim.jl`): Generated code uses same primitives
- **SPICE codegen** (`spc/codegen.jl`): Uses same Named/net/branch patterns
- **ParamLens mechanism**: Works unchanged through trace

### What Changes

| Component | Change |
|-----------|--------|
| `simulate_ir.jl` | Add MNA collection hooks to Net/branch! |
| `circuitodesystem.jl` | Add `CircuitMNASystem` alternative constructor |
| New `mna/collector.jl` | MNA stamp collection during trace |
| New `mna/system.jl` | Build G/C/b matrices from stamps |
| New `mna/solve.jl` | DC, AC, transient using SciML |

### Validation Strategy

Keep both paths available to compare results:

```julia
# Run both and compare
sys_dae = CircuitIRODESystem(circuit)
sys_mna = CircuitMNASystem(circuit)

# DC comparison
sol_dae = dc!(sys_dae, circuit)
sol_mna = dc!(sys_mna, circuit)
@assert isapprox(sol_dae.u, sol_mna.u; rtol=1e-10)

# Transient comparison
prob_dae = ODEProblem(sys_dae, u0, tspan, circuit)
prob_mna = ODEProblem(sys_mna, u0, tspan)
sol_dae = solve(prob_dae, FBDF())
sol_mna = solve(prob_mna, FBDF())
```

This parallel validation allows incremental migration while maintaining correctness.

---

## Summary

| Avoid | Prefer |
|-------|--------|
| Complete rewrite | Incremental: core → devices → VA |
| Companion models | Let solver handle time integration |
| Combined G+C matrix | Separate for analysis flexibility |
| AST walking for ddt | Trace-time Dual detection |
| Dense Jacobians | Sparse with precomputed pattern |
| Losing constant folding | Preserve for unswept parameters |
| Rewriting device code | Intercept at Net/branch! level |
| Breaking existing tests | Parallel validation with DAECompiler |

**The key insight:** Design primitives that are easy to target from existing codegen (`spectre.jl`, `vasim.jl`) and map cleanly to SciML's `DAEProblem`/`ODEProblem`. The solver handles the hard numerical work.

**The refined insight:** Intercept at the `Net` and `branch!` level in `simulate_ir.jl` to collect MNA stamps during circuit trace. This keeps all device code unchanged while enabling a parallel MNA backend for validation and eventual replacement.
