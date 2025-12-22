# MNA Architecture Design Document

This document captures the design decisions for the MNA (Modified Nodal Analysis)
engine, based on analysis of CedarSim's existing patterns, OpenVAF/VACASK interfaces,
and GPU computing requirements.

## Design Principles

### 1. Out-of-Place Evaluation

Circuit evaluation functions return new matrices rather than mutating:

```julia
function eval_circuit(lens, spec, t, u)
    G = spzeros(n, n)
    C = spzeros(n, n)
    b = zeros(n)
    # ... stamp devices ...
    return (G, C, b)
end
```

**Rationale:**
- **GPU compatibility**: DiffEqGPU.jl requires out-of-place formulation (`ODEProblem{false}`)
- **Simplicity**: No need to track which entries to reset
- **Ensemble solving**: Parameter sweeps use `EnsembleGPUKernel` which needs out-of-place
- **JIT optimization**: Julia's compiler can still optimize constant assignments

### 2. Explicit Parameter Passing (No ScopedValue)

Parameters flow explicitly through function arguments, not via `ScopedValue`:

```julia
# NOT this (ScopedValue):
temper()  # reads from global spec[]

# THIS (explicit):
spec.temp  # passed as argument
```

**Rationale:**
- DAECompiler treats `ScopedValue` as constant via compiler magic - not available in plain Julia
- Explicit passing enables full JIT optimization
- Avoids Julia's closure boxing bug (captured variables become `Core.Box`)

### 3. Separation of Concerns

```julia
struct MNASpec
    temp::Float64      # Temperature (Celsius)
    mode::Symbol       # :dcop, :tran, :tranop, :ac
end

# lens: ParamLens for circuit parameters (from ParamSim)
# spec: Simulation specification
# t: Current time
# u: Current solution (for non-linear devices)
```

**Parameter access via ParamLens:**
```julia
function build_circuit(lens, spec, t, u)
    # lens.subcircuit returns scoped lens for subcircuit
    # lens(; R=1000.0) returns actual value with lens overrides
    p = lens(; R=1000.0, C=1e-6)

    stamp_resistor!(G, n1, n2, p.R)
end
```

### 4. JIT-Friendly Stamping

Since Julia JIT-compiles circuit functions, linear device stamps optimize to constant
assignments. No need to manually separate "setup" from "eval" phases:

```julia
function eval_circuit(lens, spec, t, u)
    p = lens(; R=1000.0)

    # Linear: compiler sees p.R as constant
    stamp_resistor!(G, n1, n2, p.R)  # → G[i,j] = const

    # Temperature-dependent: constant within simulation
    R_t = p.R * (1 + p.tc * (spec.temp - 27))
    stamp_resistor!(G, n3, n4, R_t)

    # Time-dependent: changes each timestep
    v = spec.mode == :dcop ? p.Vdc : pwl(p.times, p.vals, t)
    stamp_vsource!(G, b, n5, n6, v)

    # Non-linear: changes each Newton iteration
    V_d = u[n7] - u[n8]
    I_d, G_d = diode_model(V_d, p.Is, p.Vt)
    stamp_conductance!(G, n7, n8, G_d)
    stamp_current!(b, n7, n8, I_d - G_d * V_d)

    return (G, C, b)
end
```

## Device Categories

### Linear Devices (Constant Stamps)
- Resistor: stamps G matrix
- Capacitor: stamps C matrix
- Inductor: stamps G and C matrices (with current variable)
- Ideal voltage/current sources: stamps G and b

These have constant contributions that the JIT can optimize.

### Temperature-Dependent Devices
- Resistors with tempco: `R(T) = R0 * (1 + tc1*(T-Tnom) + tc2*(T-Tnom)²)`
- Semiconductor models with temperature scaling

Constant within a simulation run, recomputed for temperature sweeps.

### Time-Dependent Sources
- PWL (Piecewise Linear)
- PULSE, SIN, EXP waveforms
- Controlled sources with time-varying control

Evaluated at each timestep. In `:dcop` mode, return DC value.

### Non-Linear Devices
- Diodes: `I = Is*(exp(V/Vt) - 1)`
- MOSFETs, BJTs, etc.

Require Newton-Raphson iteration. Each iteration:
1. Evaluate device equations at current `u`
2. Compute companion model (linearized around operating point)
3. Stamp equivalent conductance and current source

## Integration with SPICE Codegen

SPICE codegen will emit circuit functions following this pattern:

```julia
# .TEMP 50 generates:
spec = @set spec.temp = 50.0

# .PARAM R1=2k generates:
params = @set params.R1 = 2000.0

# Device instantiation:
function circuit(lens, spec, t, u)
    p = lens(; R1=1000.0)  # default, lens overrides
    stamp_resistor!(G, n1, n2, p.R1)
    # ...
end
```

The `@set` macro (Accessors.jl) creates new bindings in SSA style,
enabling constant folding when values are known at compile time.

## GPU Support

### Ensemble GPU (Parameter Sweeps)
For many small circuits with different parameters:
- Use `ODEProblem{false}` (out-of-place)
- StaticArrays for state vectors
- `EnsembleGPUKernel(CUDA.CUDABackend())`

### Large Circuit GPU
For single large circuits:
- `CuSparseMatrixCSR` for sparse matrices
- Iterative solvers via Krylov.jl
- Direct solvers via CUDSS.jl

## Parameterization and Constant Folding

This section documents the careful design of the parameterization system to enable
Julia's JIT compiler to optimize parameters as constants.

### The Optimization Goal

Circuit simulation performance depends critically on the compiler recognizing that
certain values (like resistor values) are constant during a simulation run. When
the compiler knows a value is constant, it can:

1. **Inline the value** directly into machine code
2. **Eliminate dead branches** (e.g., `mode == :dcop` becomes `true` or `false`)
3. **Fold arithmetic** at compile time
4. **Unroll loops** with known bounds

### The Key Insight: In-Function Literals vs Arguments

Julia's JIT specializes functions on **argument types**, not values. This means:

- **Function arguments are NOT constants** - even immutable struct fields
- **Literals defined inside the function body ARE constants** - the compiler sees them

```julia
# BAD: R comes from argument, NOT constant-folded
function build_circuit(params)
    R = params.R  # This is NOT a constant - it came from outside
    stamp_resistor!(G, n1, n2, R)
end

# GOOD: R is a literal in the function body, IS constant-folded
function build_circuit()
    R = 1000.0  # This IS a constant - defined right here
    stamp_resistor!(G, n1, n2, R)
end
```

### The ParamLens Pattern

The brilliance of ParamLens is that it lets you define defaults as literals inside
the function, while still allowing external overrides:

```julia
function build_circuit(lens, spec)
    # Defaults are LITERALS defined right here in the function body
    p = lens(; R=1000.0, C=1e-6, Vcc=5.0)

    # If lens doesn't override R, the compiler sees R=1000.0 as a constant!
    # Only parameters actually in the lens are runtime values
    stamp!(Resistor(p.R), ctx, n1, n2)
end
```

When called with `IdentityLens()` or an empty `ParamLens()`:
- Julia specializes on the lens **type**
- Since `IdentityLens` always returns defaults, the compiler sees through it
- All values use the in-function literals → **all constants**

When called with `ParamLens((params=(R=2000.0,),))`:
- `R` is loaded from the lens struct at runtime → **not constant**
- `C` and `Vcc` use literals → **still constants**

The type specialization is key: `IdentityLens` as a type guarantees defaults are used,
so even when passed as an argument, the compiler can constant-fold.

### The Closure Boxing Problem

Julia implements closures as structs, but captured variables that might be reassigned
are "boxed" in `Core.Box`, preventing optimization:

```julia
# BAD: Closure captures boxed variable
function make_circuit_fn(R)
    return (G, b) -> begin
        stamp!(G, R)  # R is in a Core.Box - NOT optimized
    end
end

# GOOD: Define the literal inside the closure
function make_circuit_fn()
    return (G, b) -> begin
        R = 1000.0  # Defined here - CAN be optimized
        stamp!(G, R)
    end
end
```

### SSA-Style Parameter Updates

Using Accessors.jl `@set` creates new bindings rather than mutating:

```julia
spec1 = MNASpec(temp=27.0, mode=:tran)
spec2 = @set spec1.temp = 50.0  # New binding, spec1 unchanged
```

This maintains SSA (Static Single Assignment) form. Combined with inlining,
the compiler can propagate these values through the code.

### ParamLens Integration

For hierarchical circuits, ParamLens provides type-specialized parameter lookup:

```julia
struct ParamLens{NT<:NamedTuple}
    nt::NT
end

# Accessing a subcircuit returns a new, typed lens
Base.getproperty(lens::ParamLens{T}, sym::Symbol) where T = ...

# Calling the lens merges defaults with overrides
(lens::ParamLens)(; defaults...) = merge(defaults, lens.nt.params)
```

The key insight: each `getproperty` call returns a **new lens type**, specialized
on the remaining parameter structure. This enables the compiler to see through
the lens abstraction.

Example:
```julia
function build_subcircuit(lens, spec)
    # lens(; R=1000.0) returns params with R defaulting to 1000.0
    # but overridden if lens contains R
    p = lens(; R=1000.0, C=1e-6)

    # If lens doesn't override R, compiler sees p.R as constant 1000.0
    stamp!(Resistor(p.R), ctx, n1, n2)
end
```

### Verifying Optimization

To verify that parameters are being constant-folded, use `@code_llvm` or `@code_native`:

```julia
# This version has the literal in the function - WILL be constant-folded
function stamp_test_constant()
    G = zeros(2, 2)
    R = 1000.0  # Literal in function body
    g = 1.0 / R
    G[1,1] = g
    G[1,2] = -g
    G[2,1] = -g
    G[2,2] = g
    return G
end

@code_llvm stamp_test_constant()
# Should show: 0.001 as a constant, no division at runtime

# This version takes R as argument - will NOT be constant-folded
function stamp_test_arg(R)
    G = zeros(2, 2)
    g = 1.0 / R  # R is an argument, not a constant
    G[1,1] = g
    # ...
    return G
end

@code_llvm stamp_test_arg(1000.0)
# Will show: fdiv instruction (division happens at runtime)
```

### What Gets Optimized vs. What Doesn't

| Category | Optimized? | Reason |
|----------|------------|--------|
| Literal `R = 1000.0` in function | ✅ Yes | Compiler sees the literal |
| `IdentityLens()(; R=1000.0)` | ✅ Yes | Type guarantees defaults used |
| `ParamLens(...)(; R=1000.0)` when lens overrides R | ❌ No | Value loaded from lens at runtime |
| `ParamLens(...)(; R=1000.0)` when lens doesn't override R | ✅ Yes | Falls through to literal |
| Scalar argument (e.g. `f(R::Float64)`) | ❌ No | Julia specializes on types, not values |
| Time `t` | ❌ No | Changes each ODE step |
| Solution `u` | ❌ No | Changes each Newton iteration |

### MNASim vs ParamSim

| Feature | MNASim | ParamSim |
|---------|--------|----------|
| Default params | Literals in builder function | Literals in builder function |
| Override params | Via NamedTuple argument | Via `ParamLens` |
| Spec access | `spec.temp` argument | `spec[].temp` (ScopedValue) |
| Mode access | `spec.mode` argument | `sim_mode[]` (ScopedValue) |
| Constant folding | In-function literals only | DAECompiler treats ScopedValue as constant |
| Hierarchical params | Manual lens passing | Automatic via ParamLens |

Both approaches rely on defaults being literals in the function body.
ParamSim additionally uses DAECompiler's ability to treat ScopedValue
reads as constants during compilation, which MNASim cannot do.

## Comparison with OpenVAF/OSDI

OpenVAF's OSDI interface separates:
- `setup_instance(temp, ...)` - one-time setup
- `eval(sim_info)` - per-iteration evaluation
- `load_jacobian_resist/react` - separate resistive/reactive
- `JACOBIAN_ENTRY_RESIST_CONST` flags for constant entries

This separation is for C ABI efficiency without JIT. In Julia, we can
let the JIT handle optimization and use a simpler unified `eval` function.

Key insight from OSDI:
- Temperature is passed explicitly to setup
- Time (`abstime`) flows through `sim_info` each iteration
- Analysis mode is a flag (ANALYSIS_DC, ANALYSIS_TRAN)

## References

- CedarSim ParamLens: `src/spectre.jl:140-180`
- CedarSim SimSpec: `src/simulate_ir.jl:20-32`
- OpenVAF OSDI: `refs/OpenVAF/melange/core/src/veriloga/osdi_0_4.rs`
- VACASK device eval: `refs/VACASK/lib/osdiinstance.cpp`
- DiffEqGPU: https://docs.sciml.ai/DiffEqGPU/stable/getting_started/
