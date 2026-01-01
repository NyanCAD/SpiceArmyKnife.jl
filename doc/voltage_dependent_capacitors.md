# Voltage-Dependent Capacitors: Charge Formulation Implementation

## Problem Statement

When `vasim.jl` encounters `ddt(Q(V))` where `Q(V)` is a nonlinear charge function (voltage-dependent capacitor), it currently stamps `∂Q/∂V = C(V)` into the C matrix. This creates a **state-dependent mass matrix** which SciML solvers don't handle well.

### Current Behavior

For a contribution like:
```verilog
I(p,n) <+ ddt(Q(V(p,n)));  // Q(V) is nonlinear in V
```

The s-dual approach evaluates:
- `va_ddt(Q(V))` returns `Dual{ContributionTag}(0, Q(V))`
- `partials(result, 1)` = `Q(V)` (the charge)
- `∂Q/∂V = C(V)` is stamped into C matrix

This yields the ODE system: `C(V) * dV/dt + G*V = b`

When `C(V)` varies with state `V`, SciML's mass matrix ODE solvers struggle:
- Rosenbrock methods (Rodas4, Rodas5P) only support constant mass matrices
- `RadauIIA5` technically supports non-constant mass via `DiffEqArrayOperator`, but it's undocumented and unstable
- The recommended approach is reformulating to constant mass matrix

### Examples of Voltage-Dependent Capacitors

1. **Junction capacitance** (diodes, MOSFETs):
   ```
   Q(V) = Cj0 * φ * (1 - (1 - V/φ)^(1-m))
   C(V) = dQ/dV = Cj0 * (1 - V/φ)^(-m)
   ```

2. **MOS gate charge** (BSIM models):
   ```
   Qg = f(Vgs, Vds, Vbs)  // Complex nonlinear function
   Cgg = ∂Qg/∂Vgs, Cgd = ∂Qg/∂Vds, etc.
   ```

3. **Varactor**:
   ```
   Q(V) = C0 * V + C1 * V^2 + C2 * V^3
   C(V) = C0 + 2*C1*V + 3*C2*V^2
   ```

## Solution: Charge Formulation

Reformulate using **charge as an explicit state variable**:

### Original (state-dependent mass matrix):
```
C(V) * dV/dt = I_branch
```

### Reformulated (constant mass matrix):
```
dq/dt = I_branch     (differential equation, mass coefficient = 1)
q - Q(V) = 0         (algebraic constraint)
```

This transforms the system from:
```
[C(V)  0] [V̇]   [f(V)]
[0     0] [·] = [·   ]
```

To:
```
[M₀   0  0 ] [V̇]   [KCL equations (include dq/dt from capacitor)]
[0    0  0 ] [·] = [other algebraic                             ]
[0    0  0 ] [q̇]   [constraint row (algebraic)                  ]
              +
[G block     | 0 ] [V]   [b    ]
[            | · ] [·] = [     ]
[−∂Q/∂V      | 1 ] [q]   [Q(V₀)]  <- q = Q(V) constraint
```

Key insight: The charge constraint row `q - Q(V) = 0` is **algebraic** (no C entry).
The dq/dt terms appear in the KCL equations via `C[p, q_idx] = 1` and `C[n, q_idx] = -1`.
This correctly enforces the constraint exactly at each time step, not just at steady state.

The mass matrix is now constant (zeros in the charge row, KCL rows have constant coupling to dq/dt).

## Detection Strategy: Nested Duals (No AST Analysis)

We detect voltage-dependent capacitance **at runtime using AD**, avoiding AST analysis. The key insight: if `∂²Q/∂V² ≠ 0`, then `C(V) = ∂Q/∂V` is voltage-dependent.

### Dual Tag Hierarchy

```
Precedence (outer to inner):
  ContributionTag  - separates resistive/reactive (s-dual for ddt)
  JacobianTag      - ∂/∂V for Jacobian extraction
  CapacitanceDerivTag - ∂²/∂V² to detect voltage dependence
```

### Detection Algorithm

```julia
function is_voltage_dependent_charge(contrib_fn, Vp::Real, Vn::Real)
    # Triple-nested duals:
    # 1. CapacitanceDerivTag (innermost) - for ∂²Q/∂V²
    # 2. JacobianTag (middle) - for ∂Q/∂V
    # 3. ContributionTag (outermost) - separates q from I via va_ddt

    # IMPORTANT: Check at BOTH the operating point AND a perturbed point.
    # Some functions have ∂²Q/∂V² = 0 at specific points (e.g., Q = V³ at V=0)
    # but are still voltage-dependent. The perturbation catches these cases.

    # Create voltage with all three dual layers
    Vp_triple = create_triple_dual(Vp, ...)
    result = contrib_fn(Vp_triple - Vn_triple)

    # Check at operating point
    if has_nonzero_second_derivative(result)
        return true
    end

    # Check at perturbed point (ε = 1e-3) to catch edge cases
    return check_at_point(contrib_fn, Vp + ε, Vn)
end
```

This perturbation approach ensures **type-level detection** (analytically nonlinear functions)
rather than value-level detection (which could miss edge cases at zeros of ∂²Q/∂V²).

### Performance: Type Stability and Inlining

To ensure the detection overhead optimizes away:

1. **Type-stable dual creation**: All dual types are concrete and known at compile time
2. **Inline detection functions**: Use `@inline` for hot paths
3. **Branch on type, not value**: Detection result feeds into type-dispatched stamping
4. **Const-prop friendly**: Simple boolean result enables dead code elimination

```julia
@inline function evaluate_and_detect(contrib_fn, Vp::Real, Vn::Real)
    # Standard evaluation (always needed)
    result = evaluate_contribution(contrib_fn, Vp, Vn)

    # Detection only for reactive contributions
    if result.has_reactive
        is_vdep = is_voltage_dependent_charge(contrib_fn, Vp, Vn)
        return (result..., is_voltage_dependent = is_vdep)
    end
    return (result..., is_voltage_dependent = false)
end
```

For the common case (linear capacitors), the detection should compile away to a constant `false` after inlining since `∂²(C*V)/∂V² = 0` is known at compile time for literal `C`.

## Implementation Plan

### Phase 1: Detection Infrastructure (`src/mna/contrib.jl`)

```julia
# New tag for second derivatives
struct CapacitanceDerivTag end

# Tag ordering (all three must be ordered correctly)
ForwardDiff.:≺(::Type{CapacitanceDerivTag}, ::Type{JacobianTag}) = true
ForwardDiff.:≺(::Type{CapacitanceDerivTag}, ::Type{ContributionTag}) = true
# ... other orderings ...

# Detection function
@inline function is_voltage_dependent_charge(contrib_fn, Vp::Real, Vn::Real)::Bool
    # Implementation using triple-nested duals
end
```

### Phase 2: Charge Variable Support (`src/mna/context.jl`)

```julia
mutable struct MNAContext
    # ... existing fields ...

    # Charge state variables
    charge_names::Vector{Symbol}
    n_charges::Int
    charge_branches::Vector{Tuple{Int, Int}}  # (p, n) for each charge
end

function alloc_charge!(ctx::MNAContext, name::Symbol, p::Int, n::Int)::Int
    push!(ctx.charge_names, name)
    ctx.n_charges += 1
    push!(ctx.charge_branches, (p, n))
    return ctx.n_nodes + ctx.n_currents + ctx.n_charges
end
```

### Phase 3: Charge-Based Stamping (`src/mna/contrib.jl`)

```julia
"""
Stamp voltage-dependent capacitor using charge formulation.
"""
function stamp_charge_state!(
    ctx::MNAContext,
    p::Int, n::Int,
    q_fn,  # V -> Q(V)
    x::AbstractVector,
    charge_name::Symbol
)
    q_idx = alloc_charge!(ctx, charge_name, p, n)

    Vp = p == 0 ? 0.0 : x[p]
    Vn = n == 0 ? 0.0 : x[n]
    q_val = length(x) >= q_idx ? x[q_idx] : 0.0

    result = evaluate_charge_contribution(q_fn, Vp, Vn)

    # 1. dq/dt term: constant mass coefficient of 1
    stamp_C!(ctx, q_idx, q_idx, 1.0)

    # 2. KCL: I = dq/dt flows from p to n
    stamp_C!(ctx, p, q_idx, 1.0)   # current leaves p
    stamp_C!(ctx, n, q_idx, -1.0)  # current enters n

    # 3. Constraint: q - Q(V) = 0
    stamp_G!(ctx, q_idx, q_idx, 1.0)           # ∂/∂q = 1
    stamp_G!(ctx, q_idx, p, -result.dq_dVp)    # ∂/∂Vp = -∂Q/∂Vp
    stamp_G!(ctx, q_idx, n, -result.dq_dVn)    # ∂/∂Vn = -∂Q/∂Vn

    # Newton companion for constraint
    b_val = q_val - result.q - (1.0*q_val - result.dq_dVp*Vp - result.dq_dVn*Vn)
    stamp_b!(ctx, q_idx, b_val)

    return q_idx
end
```

### Phase 4: Update vasim.jl Code Generation

Modify `generate_mna_stamp_method_nterm` to:
1. Call detection at runtime for contributions with `ddt()`
2. Branch to charge formulation when voltage-dependent
3. Use standard C-matrix stamping for constant capacitors

The detection and branching should be type-stable so the JIT can optimize.

### Phase 5: Solver Updates (`src/mna/solve.jl`)

```julia
function make_dae_problem(sys::MNASystem, tspan; kwargs...)
    # Identify differential vs algebraic variables
    n = system_size(sys)
    differential_vars = [sys.C[i,i] != 0 for i in 1:n]

    # Constraint rows (charge variables) have 0 in C diagonal initially
    # but we set them to 1 in stamp_charge_state!, so they're differential

    # Use DAEProblem with IDA or DFBDF
    # ...
end
```

### Phase 6: Testing

New test file `test/mna/charge_formulation.jl`:

```julia
@testset "Detection" begin
    # Linear: I = C * ddt(V) -> not voltage dependent
    @test !is_voltage_dependent_charge(V -> 1e-12 * va_ddt(V), 1.0, 0.0)

    # Quadratic: I = ddt(C * V^2) -> voltage dependent
    @test is_voltage_dependent_charge(V -> va_ddt(1e-12 * V^2), 1.0, 0.0)

    # Junction cap -> voltage dependent
    Cj0, phi, m = 1e-12, 0.7, 0.5
    jcap(V) = va_ddt(Cj0 * phi * (1 - (1 - V/phi)^(1-m)))
    @test is_voltage_dependent_charge(jcap, 0.3, 0.0)
end

@testset "Charge formulation transient" begin
    # Verify transient with nonlinear cap matches expected/reference
end
```

## Key Design Decisions

1. **Runtime detection via AD** (not AST analysis): More robust, handles arbitrary expressions, leverages existing dual infrastructure.

2. **Type-stable detection**: Returns `Bool`, enabling compile-time optimization for common cases.

3. **Charge as explicit state**: Yields constant mass matrix, enabling efficient Rosenbrock solvers.

4. **Backward compatible**: Linear capacitors continue to use existing C-matrix stamping.

## Files to Modify

| File | Changes |
|------|---------|
| `src/mna/contrib.jl` | Add `CapacitanceDerivTag`, `is_voltage_dependent_charge()`, `stamp_charge_state!()` |
| `src/mna/context.jl` | Add charge variable fields and `alloc_charge!()` |
| `src/vasim.jl` | Modify stamp generation to detect and handle voltage-dependent caps |
| `src/mna/solve.jl` | Update DAE problem creation for charge variables |
| `test/mna/charge_formulation.jl` | New test file |

## References

- `doc/Sciml charge formulation.md` - Background on SciML DAE support
- `doc/mna_ad_stamping.md` - Existing s-dual approach for ddt()
- `src/mna/contrib.jl` - Current contribution evaluation
