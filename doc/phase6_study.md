# Phase 6 Study: Complex VA & DAE Integration

## Overview

Phase 6 aims to provide full Verilog-A support including nonlinear capacitors, enabling simulation of complex devices like BSIM4 and BSIMCMG MOSFETs. This document summarizes the research conducted on the existing MNA backend, OpenVAF's compilation approach, and SciML's DAE/ODE solver APIs.

## Current MNA Backend Status (Phases 1-5)

### Core Infrastructure

The MNA module (`src/mna/`) provides:

1. **MNAContext** (`context.jl`): Accumulates stamps into sparse matrix triplets
   - `G`: Conductance matrix (resistive)
   - `C`: Capacitance matrix (reactive)
   - `b`: RHS vector (sources)

2. **MNASystem** (`build.jl`): Assembled sparse matrices
   - Converts triplets to CSC sparse format
   - Tracks node names and current variables

3. **Devices** (`devices.jl`): Basic stamp! methods
   - Linear: Resistor, Capacitor, Inductor
   - Sources: VoltageSource, CurrentSource, PWL, SIN
   - Controlled sources: VCVS, VCCS, CCVS, CCCS
   - Behavioral sources: B-sources

4. **Solvers** (`solve.jl`): Analysis functions
   - `solve_dc`: Newton iteration for DC operating point
   - `solve_ac`: Frequency sweep (G + jωC)x = b
   - `make_ode_problem`: Mass matrix ODE for transient
   - `make_dae_problem`: Implicit DAE formulation

5. **VA Contributions** (`contrib.jl`): s-dual stamping
   - `va_ddt(x)`: Time derivative via Dual numbers
   - `stamp_contribution!`: AD-based Jacobian extraction
   - `evaluate_contribution`: Separates resistive/reactive parts

### Current vasim.jl Implementation

The `vasim.jl` module translates Verilog-A modules to MNA stamp functions:

```julia
# VA module → stamp! method
va"""
module resistor(p, n);
    parameter real R = 1000;
    inout p, n;
    electrical p, n;
    analog I(p,n) <+ V(p,n)/R;
endmodule
"""
```

Generates:
```julia
function stamp!(::VAResistor, ctx, p, n, params, spec; x=Float64[])
    V = voltage_between(x, p, n)  # Get Vp - Vn
    contrib_fn = V -> V / params.R
    stamp_current_contribution!(ctx, p, n, contrib_fn, x)
end
```

Key features:
- Uses s-dual approach: `va_ddt(x)` returns `Dual(0, x)`
- ForwardDiff extracts both value and Jacobian
- `ContributionTag` ensures correct dual nesting
- Handles mixed resistive + reactive contributions

### What Works

- Simple VA devices (resistors, capacitors, diodes)
- Linear circuits with time-dependent sources
- DC operating point with Newton iteration
- Transient via mass matrix ODE

### What's Missing for Phase 6

1. **Multi-branch devices**: MOSFETs have 4+ terminals with multiple current contributions
2. **Internal nodes**: Complex devices have internal junctions
3. **Nonlinear capacitors**: Voltage-dependent charge q(V)
4. **Operating point dependence**: Rebuild Jacobian at each Newton step
5. **Complex VA constructs**: `@(initial_step)`, `$limit`, etc.

## OpenVAF Analysis

OpenVAF compiles Verilog-A to OSDI-compatible dynamic libraries. Key insights:

### Topology Representation

```rust
struct Contribution {
    unknown: Option<Value>,    // Node voltage for this branch
    resist: Value,             // Resistive part I(V)
    react: Value,              // Reactive part Q(V) (charge)
    resist_small_signal: Value,// AC small-signal resistive
    react_small_signal: Value, // AC small-signal reactive
    noise: Vec<Noise>,         // Noise sources
}

struct BranchInfo {
    is_voltage_src: Value,     // Whether this is V-source branch
    voltage_src: Contribution, // V(p,n) <+ expr stamps
    current_src: Contribution, // I(p,n) <+ expr stamps
}
```

### DAE System

```rust
struct DaeSystem {
    unknowns: TiSet<SimUnknown, SimUnknownKind>,
    residual: TiVec<SimUnknown, Residual>,
    jacobian: TiVec<MatrixEntryId, MatrixEntry>,
    noise_sources: Vec<NoiseSource>,
    model_inputs: Vec<(u32, u32)>,  // Node pairs
}

struct Residual {
    resist: Value,        // I(x)
    react: Value,         // Q(x)
    resist_lim_rhs: Value, // Limiting correction
    react_lim_rhs: Value,
}

struct MatrixEntry {
    row: SimUnknown,
    col: SimUnknown,
    resist: Value,  // ∂I/∂x
    react: Value,   // ∂Q/∂x
}
```

### Automatic Differentiation

OpenVAF uses MIR-level AD (similar to source-to-source transformation):
- Identifies which derivatives are needed via `LiveDerivatives`
- Applies chain rule to build derivative expressions
- Optimizes away unnecessary computations

### Key Design Decisions

1. **Separate resist/react**: Clear separation enables different treatment in AC vs transient
2. **Jacobian structure**: Pre-computed sparsity pattern, filled at each eval
3. **Limiting support**: `lim_rhs` terms for voltage limiting in Newton iteration
4. **Small-signal network**: Identifies nodes that are always at ground for AC

## SciML API Requirements

### DAEProblem Interface

```julia
# Residual: F(du, u, p, t) = 0
function dae_residual!(resid, du, u, p, t)
    # resid[i] = equation_i(du, u, p, t)
    # All equations rearranged so RHS = 0
end

prob = DAEProblem(
    dae_residual!,
    du0,           # Initial derivatives
    u0,            # Initial state
    tspan,
    params;
    differential_vars = Bool[...],  # Which vars have du terms
)

sol = solve(prob, IDA())
```

### Mass Matrix ODE Form

```julia
# M * du/dt = f(u, p, t)
# For MNA: C * dx/dt = b - G*x

function ode_rhs!(du, u, p, t)
    # du = b(t) - G*u
end

f = ODEFunction(
    ode_rhs!;
    mass_matrix = C,
    jac = jac!,
    jac_prototype = -G,
)

prob = ODEProblem(f, u0, tspan, params)
sol = solve(prob, Rodas5P())  # Rosenbrock handles singular mass matrix
```

### Initialization

- `CedarDCOp`: Switches to :dcop mode, solves DC, then re-initializes
- `BrownFullBasicInit`: Adjusts algebraic variables to satisfy constraints
- IDA's `IDADefaultInit`: Internal consistent initialization

### Solver Recommendations

| Scenario | Solver | Notes |
|----------|--------|-------|
| Small stiff systems | Rodas5P | Good for mass matrix ODE |
| Large systems | IDA | Fixed-coefficient BDF, handles DAE |
| Pure Julia | DFBDF | Adaptive BDF, finite diff Jacobian |
| High accuracy | RadauIIA5 | Implicit Runge-Kutta |

## Phase 6 Design Considerations

### Architecture Goals

1. **Drop-in replacement for DAECompiler**: Same interface as `CircuitIRODESystem`
2. **Efficient Jacobian computation**: Reuse AD infrastructure from contrib.jl
3. **Support complex VA models**: BSIM4, BSIMCMG with charge models
4. **GPU compatibility**: Out-of-place evaluation pattern

### Proposed Approach

#### 1. Enhanced VA Code Generation

Extend `make_mna_device()` to handle:
- Multi-port devices (D, G, S, B terminals)
- Internal node allocation
- Multiple contribution statements per device
- Voltage-dependent charge: `I(p,n) <+ ddt(C(V) * V)`

```julia
# Generated code pattern for MOSFET-like device
function stamp!(dev::BSIM4, ctx, d, g, s, b, params, spec; x=Float64[])
    Vgs = x[g] - x[s]
    Vds = x[d] - x[s]
    Vbs = x[b] - x[s]

    # Compute currents and charges (operating point dependent)
    ids, gm, gds, gmb = compute_ids(Vgs, Vds, Vbs, params)
    qg, qd, qs, qb = compute_charges(Vgs, Vds, Vbs, params)

    # Stamp drain current
    stamp_G!(ctx, d, g, gm)
    stamp_G!(ctx, d, d, gds)
    stamp_G!(ctx, d, b, gmb)
    # ... etc

    # Stamp charges → capacitances via AD
    # ∂qg/∂Vg, ∂qg/∂Vd, etc.
end
```

#### 2. Operating Point Rebuild

For nonlinear devices, the Jacobian depends on the operating point:

```julia
function make_dae_function_nonlinear(builder, params, spec)
    # Rebuilds G, C, b at each Newton step
    function dae_residual!(resid, du, u, p, t)
        # Build circuit at current operating point
        ctx = builder(params, with_time(spec, t); x=u)
        sys = assemble!(ctx)

        # F = C*du + G*u - b = 0
        resid .= sys.C * du + sys.G * u - sys.b
    end

    return dae_residual!
end
```

#### 3. Charge-Based Formulation

For voltage-dependent capacitors, use charge formulation:

```
I = dQ/dt = d(q(V))/dt = dq/dV * dV/dt = C(V) * dV/dt
```

Where `C(V) = dq/dV` is computed via ForwardDiff:

```julia
function stamp_charge_contribution!(ctx, p, n, q_fn, x)
    Vp = p == 0 ? 0.0 : x[p]
    Vn = n == 0 ? 0.0 : x[n]
    V = Vp - Vn

    # q_fn: V → charge
    # Use ForwardDiff to get dq/dV
    q_dual = q_fn(ForwardDiff.Dual(V, 1.0))
    q = ForwardDiff.value(q_dual)
    C = ForwardDiff.partials(q_dual, 1)  # dq/dV

    # Stamp capacitance
    stamp_capacitance!(ctx, p, n, C)

    # For Newton companion: need charge residual
    # Will be incorporated into DAE formulation
end
```

#### 4. MNACircuit (SciML System → Problem Pattern)

Following SciML's convention (System → Problem → Solution), we define:

```julia
# MNACircuit is the "System" - defines circuit structure
struct MNACircuit
    builder::Function  # (params, spec; x) -> MNAContext
    params::NamedTuple
    spec::MNASpec
    tspan::Tuple{Float64, Float64}
end

# Convert to DAEProblem (the "Problem")
function SciMLBase.DAEProblem(circuit::MNACircuit)
    # Build initial system
    ctx0 = circuit.builder(circuit.params, circuit.spec; x=Float64[])
    sys0 = assemble!(ctx0)
    n = system_size(sys0)

    # Create residual function
    function residual!(resid, du, u, p, t)
        ctx = circuit.builder(circuit.params, with_time(circuit.spec, t); x=u)
        sys = assemble!(ctx)
        mul!(resid, sys.C, du)
        mul!(resid, sys.G, u, 1.0, 1.0)
        resid .-= sys.b
    end

    # Determine differential variables
    diff_vars = detect_differential_vars(sys0)

    # DC initialization
    u0 = solve_dc(circuit.builder, circuit.params, circuit.spec).x
    du0 = zeros(n)

    return DAEProblem(residual!, du0, u0, circuit.tspan;
                      differential_vars=diff_vars)
end

# Usage: System → Problem → Solution
circuit = MNACircuit(build_inverter, params, spec, tspan)  # System
prob = DAEProblem(circuit)                                   # Problem
sol = solve(prob, IDA())                                     # Solution
```

### Test Strategy

Phase 6 exit criteria from `mna_design.md`:
- `inverter.jl`: GF180 BSIM4 inverter transient
- `bsimcmg/inverter.jl`: BSIMCMG inverter
- `bsimcmg/demo_bsimcmg.jl`: BSIMCMG demo
- `ac.jl`: AC analysis (linear + BSIM inverter)

Testing approach:
1. Start with simple BSIM-like device (diode with junction capacitance)
2. Add multi-terminal support (4-terminal MOSFET)
3. Validate DC against ngspice
4. Validate transient waveforms
5. Validate AC frequency response

## Next Steps

1. **Extend vasim.jl**: Handle multi-port modules and internal nodes
2. **Implement charge stamping**: Voltage-dependent capacitance via AD
3. **Create MNACircuit**: SciML-compatible circuit system wrapper
4. **Add BSIM4 model**: Either VA→MNA translation or native implementation
5. **Write validation tests**: Compare against ngspice reference

## References

- [SciML DAE Tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dae_example/)
- [SciML Mass Matrix Solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/)
- [OpenVAF Repository](https://github.com/arpadbuermen/OpenVAF)
- [OpenVAF Internals](https://github.com/arpadbuermen/OpenVAF/blob/master/internals.md)
