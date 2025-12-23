# Phase 5: VA Contribution Functions - Implementation Plan

## Overview

Phase 5 updates `vasim.jl` to emit MNA-compatible contribution functions using the s-dual approach for automatic resistive/reactive separation, replacing the DAECompiler `branch!/kcl!/equation!` primitives.

**Goal:** Simple VA models (resistor, capacitor, diode, VCCS) work with the MNA backend.

**Target LOC:** ~400

---

## Research Summary

### Current vasim.jl Architecture

The existing `vasim.jl` (940 lines) generates Julia code from Verilog-A AST:

1. **Device struct generation:**
   ```julia
   @kwdef struct VAR <: VAModel
       R::DefaultOr{Float64} = 1000.0
   end
   ```

2. **Call operator generation:**
   ```julia
   function (self::VAR)(port_p, port_n; dscope=...)
       # Internal nodes and currents
       I_p_n = variable(...)
       # Contribution evaluation
       branch_value_p_n = V(p,n)/self.R
       # KCL equations
       kcl!(port_p, -I_p_n)
       kcl!(port_n, I_p_n)
       # Branch equation
       equation!(I_p_n - branch_value_p_n, ...)
   end
   ```

3. **SimTag duals for ddx():**
   ```julia
   struct SimTag end
   # V() returns Dual{SimTag} to track ∂/∂V for ddx()
   ```

### MNA Backend Architecture

The MNA module provides:

1. **MNAContext:** Accumulates G, C, b stamps in COO format
2. **MNASpec:** Simulation specification (temp, mode, time)
3. **stamp!:** Device stamping primitives
4. **Solvers:** DC, AC, transient via SciML

### OpenVAF/OSDI Interface Insights

The OSDI interface (`osdi_0_4.rs`) shows key patterns:

1. **Separate resist/react handling:**
   - `load_residual_resist()`, `load_residual_react()`
   - `load_jacobian_resist()`, `load_jacobian_react(alpha)`
   - `JACOBIAN_ENTRY_RESIST_CONST`, `JACOBIAN_ENTRY_REACT_CONST` flags

2. **Simulation info per-evaluation:**
   - `OsdiSimInfo.abstime` - current time
   - `OsdiSimInfo.flags` - ANALYSIS_DC, ANALYSIS_TRAN, etc.
   - `OsdiSimInfo.prev_solve` - previous solution vector

3. **Temperature at setup:**
   - `setup_instance(temp, ...)` - temperature passed explicitly

### SciML API Target

The target API patterns from [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/):

1. **Mass matrix ODE:**
   ```julia
   M = [1.0 0 0; 0 1.0 0; 0 0 0]  # Singular for DAE
   f = ODEFunction(rhs!; mass_matrix=M, jac=jac!, jac_prototype=sparse_pattern)
   prob = ODEProblem(f, u0, tspan)
   ```

2. **Implicit DAE:**
   ```julia
   prob = DAEProblem(residual!, du0, u0, tspan; differential_vars=[true, true, false])
   ```

---

## Design Decisions

### 1. S-Dual Approach for Resist/React Separation

From `mna_ad_stamping.md`, use ForwardDiff to automatically separate resistive and reactive contributions:

```julia
using ForwardDiff: Dual, value, partials

# Laplace variable s as a Dual
const va_s = Dual(0.0, 1.0)

# ddt in Laplace domain
va_ddt(x) = va_s * x

# Evaluation result:
# value(result) → resistive part f(V) → stamps into G
# partials(result, 1) → charge q(V) → stamps into C via ∂q/∂V
```

**Example:**
```julia
# I(p,n) <+ V(p,n)/R + C*ddt(V(p,n))
function contribution(V_pn)
    V_pn / R + C * va_ddt(V_pn)
end
# Result: Dual(V/R, C*V)
# - value = V/R → resistive
# - partials = C*V → reactive (charge)
```

### 2. Nested Duals for Jacobian

Use a second layer of duals for ∂/∂V:

```julia
# Outer: s-dual for resist/react
# Inner: V-dual for Jacobian

function evaluate_contribution(contrib_fn, x::Vector{Float64}, p::Int, n::Int)
    # Create V dual for differentiation
    Vp_dual = Dual(x[p], 1.0)  # ∂/∂Vp = 1
    Vn_dual = Dual(x[n], -1.0) # ∂/∂Vp via chain rule

    result = contrib_fn(Vp_dual - Vn_dual)

    # Extract components
    resist_val = value(value(result))    # f(V)
    resist_jac = partials(value(result)) # ∂f/∂V
    react_val = value(partials(result))  # q(V)
    react_jac = partials(partials(result)) # ∂q/∂V = C(V)

    return resist_val, resist_jac, react_val, react_jac
end
```

### 3. Code Generation Strategy

Transform the existing `make_spice_device(vm)` function to generate:

**Before (DAECompiler):**
```julia
function (self::VAR)(port_p, port_n; dscope=...)
    I_p_n = variable(...)
    branch_value = V(p,n)/self.R
    kcl!(port_p, -I_p_n)
    kcl!(port_n, I_p_n)
    equation!(I_p_n - branch_value, ...)
end
```

**After (MNA):**
```julia
function stamp!(self::VAR, ctx::MNAContext, p::Int, n::Int, spec::MNASpec, x::Vector{Float64})
    # Contribution function
    function contrib(Vpn)
        Vpn / self.R
    end

    # Evaluate and stamp
    stamp_current_contribution!(ctx, p, n, contrib, x)
end
```

### 4. Contribution Stamping Primitive

Add to `src/mna/contrib.jl`:

```julia
"""
    stamp_current_contribution!(ctx, p, n, contrib_fn, x)

Stamp a general current contribution I(p,n) <+ expr into MNA matrices.
Uses ForwardDiff with s-dual for resist/react separation.
"""
function stamp_current_contribution!(
    ctx::MNAContext,
    p::Int,
    n::Int,
    contrib_fn,
    x::Vector{Float64}
)
    # Get voltage at operating point
    Vp = p == 0 ? 0.0 : x[p]
    Vn = n == 0 ? 0.0 : x[n]
    Vpn = Vp - Vn

    # Create dual for differentiation
    Vpn_dual = Dual(Vpn, 1.0)

    # Evaluate contribution
    result = contrib_fn(Vpn_dual)

    if result isa Dual
        # Has resist/react components
        resist_dual = value(result)
        react_dual = partials(result, 1)

        # Resistive part
        if resist_dual isa Dual
            I_resist = value(resist_dual)
            G = partials(resist_dual, 1)  # ∂I/∂V
        else
            I_resist = resist_dual
            G = 0.0
        end

        # Reactive part
        if react_dual isa Dual
            q = value(react_dual)
            C = partials(react_dual, 1)  # ∂q/∂V
        else
            q = react_dual
            C = 0.0
        end
    else
        # Pure resistive (no ddt)
        if result isa Dual
            I_resist = value(result)
            G = partials(result, 1)
        else
            I_resist = result
            G = 0.0
        end
        q = 0.0
        C = 0.0
    end

    # Stamp Jacobian into G matrix
    stamp_conductance!(ctx, p, n, G)

    # Stamp charge Jacobian into C matrix
    stamp_capacitance!(ctx, p, n, C)

    # Stamp residual into RHS (for Newton iteration)
    Ieq = I_resist - G * Vpn  # Companion model
    stamp_b!(ctx, p, -Ieq)
    stamp_b!(ctx, n,  Ieq)
end
```

---

## Implementation Steps

### Step 1: Create `src/mna/contrib.jl` (~100 LOC)

New file for contribution function evaluation and stamping:

```julia
# src/mna/contrib.jl

using ForwardDiff: Dual, value, partials

# Laplace variable for ddt()
const va_s = Dual(0.0, 1.0)
va_ddt(x) = va_s * x
va_ddt(x::Dual) = va_s * x

export va_s, va_ddt, stamp_current_contribution!

# ... stamp_current_contribution! as shown above
```

### Step 2: Create MNA Code Generator in `vasim.jl` (~200 LOC)

Add new function `make_mna_device(vm)` parallel to existing `make_spice_device(vm)`:

```julia
function make_mna_device(vm::VANode{VerilogModule})
    ps = pins(vm)
    modname = String(vm.id)
    symname = Symbol(modname)

    # 1. Generate struct definition (similar to existing)
    struct_fields = generate_parameter_fields(vm)

    # 2. Generate contribution function
    contrib_fn = generate_contribution_function(vm)

    # 3. Generate stamp! method
    stamp_method = generate_stamp_method(vm, contrib_fn)

    Expr(:toplevel,
        :(VerilogAEnvironment.CedarSim.@kwdef struct $symname <: VerilogAEnvironment.VAModel
            $(struct_fields...)
        end),
        stamp_method,
    )
end
```

Key changes from existing codegen:

| Existing | MNA |
|----------|-----|
| `I_p_n = variable(...)` | No current variable for I contributions |
| `V(p,n)` returns `port_p.V - port_n.V` | `Vpn` parameter to contrib function |
| `kcl!(...)` | `stamp_b!(...)` |
| `equation!(...)` | Implicit via stamp pattern |
| `ddt(x)` via DAECompiler | `va_ddt(x)` via s-dual |
| `ddx(expr, V(a,b))` via SimTag | ForwardDiff partial extraction |

### Step 3: Update `src/mna/MNA.jl` (~10 LOC)

Include the new contribution module:

```julia
# src/mna/MNA.jl
include("contrib.jl")
```

### Step 4: Integration with VA Parsing (~50 LOC)

Update `va_str` macro and `include(mod, VAFile)` to use MNA codegen when `USE_DAECOMPILER == false`:

```julia
macro va_str(str)
    va = VerilogAParser.parse(IOBuffer(str))
    if va.ps.errored
        cedarthrow(LoadError("va_str", 0, VAParseError(va)))
    else
        if CedarSim.USE_DAECOMPILER
            esc(make_module(va))  # Existing DAECompiler path
        else
            esc(make_mna_module(va))  # New MNA path
        end
    end
end
```

### Step 5: Tests (~100 LOC)

New test file `test/mna/va.jl`:

```julia
@testset "VA MNA Integration" begin
    @testset "Simple VA Resistor" begin
        # Test I(p,n) <+ V(p,n)/R
    end

    @testset "VA Capacitor" begin
        # Test I(p,n) <+ C*ddt(V(p,n))
    end

    @testset "VA RC Parallel" begin
        # Test I(p,n) <+ V(p,n)/R + C*ddt(V(p,n))
    end

    @testset "VA Diode" begin
        # Test I(p,n) <+ Is*(exp(V(p,n)/Vt) - 1)
    end

    @testset "ddx() functionality" begin
        # Test ddx(expr, V(a,b))
    end

    @testset "Reversed ports" begin
        # Test I(n,p) <+ V(n,p)/R
    end
end
```

---

## Handling Special Cases

### Case 1: Voltage Contributions

`V(p,n) <+ expr` requires a current variable (constraint equation):

```julia
function stamp_voltage_contribution!(ctx, p, n, v_expr, x)
    I_idx = alloc_current!(ctx, :I_V)

    stamp_G!(ctx, p, I_idx, 1.0)
    stamp_G!(ctx, n, I_idx, -1.0)
    stamp_G!(ctx, I_idx, p, 1.0)
    stamp_G!(ctx, I_idx, n, -1.0)

    v = v_expr(x[p] - x[n])
    stamp_b!(ctx, I_idx, v)
end
```

### Case 2: Internal Nodes

Internal nodes declared with `electrical` need allocation:

```julia
# For: electrical internal;
# Generate:
internal = get_node!(ctx, Symbol("$(instance_name)_internal"))
```

### Case 3: ddx() Function

The `ddx(expr, V(a,b))` function computes ∂expr/∂V(a,b):

```julia
# Existing vasim.jl approach using SimTag still works
# VA code: ddx(cdrain, V(g,s))
# Generated:
let x = cdrain
    isa(x, Dual) ? partials(SimTag, x, idx) : 0.0
end
```

### Case 4: Multiple Contributions to Same Branch

Verilog-A allows `I(p,n) <+` multiple times (accumulation):

```julia
# I(p,n) <+ V(p,n)/R
# I(p,n) <+ Ic

# Generate: accumulate contributions then stamp once
branch_value_p_n = 0.0
branch_value_p_n += V(p,n)/R
branch_value_p_n += Ic
# Then stamp the accumulated value
```

---

## Exit Criteria

| Criterion | Test |
|-----------|------|
| Simple VA resistor works | `I(p,n) <+ V(p,n)/R` stamps correctly |
| VA capacitor works | `I(p,n) <+ C*ddt(V(p,n))` stamps into C matrix |
| VA RC parallel works | Mixed resist/react contribution |
| VA diode works | Nonlinear `Is*(exp(V/Vt)-1)` |
| ddx() works | `ddx(expr, V(a,b))` computes partials |
| Reversed ports work | `I(n,p) <+ V(n,p)/R` sign correct |
| DC analysis matches | Compare with ngspice |
| Transient analysis works | RC charging with VA cap |

---

## Files to Create/Modify

| File | Change | LOC |
|------|--------|-----|
| New `src/mna/contrib.jl` | Contribution stamping primitives | ~100 |
| `src/mna/MNA.jl` | Include contrib.jl | ~5 |
| `src/vasim.jl` | Add `make_mna_device()` function | ~200 |
| `src/CedarSim.jl` | Conditional VA codegen path | ~10 |
| New `test/mna/va.jl` | VA MNA integration tests | ~100 |

**Total: ~415 LOC**

---

## Future Considerations (Phase 6+)

1. **Nonlinear capacitors:** C(V) requires state-dependent mass matrix → DAEProblem
2. **Noise sources:** `white_noise()`, `flicker_noise()` need noise analysis
3. **Operating point info:** `$vt`, `$temperature` access
4. **Limiting:** `limexp()`, `$limit()` for convergence
5. **Branch probes:** `I(vs)` for current-controlled sources

---

## References

- `doc/mna_design.md` - Core MNA design document
- `doc/mna_architecture.md` - Parameterization and GPU patterns
- `doc/mna_ad_stamping.md` - S-dual approach for VA contributions
- [OpenVAF OSDI](https://github.com/arpadbuermen/OpenVAF) - Reference OSDI interface
- [SciML DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) - Target API patterns

Sources consulted:
- [DAE Example Tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dae_example/)
- [Mass Matrix and DAE Solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/)
- [ODE Problem Types](https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/)
