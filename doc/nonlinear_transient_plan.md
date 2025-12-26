# Nonlinear Transient Analysis Implementation Plan

## Status: Phases 1-2 Complete ✓

**Completed:**
- Phase 1: Validated DAEProblem path with VAMOSCap model
- Phase 2: Added high-level API (`dc!`/`tran!` on `MNACircuit`)
- Unified API around MNACircuit (removed tspan from circuit, moved to tran!)
- Automatic solver dispatch: DAE solvers use DAEProblem, ODE solvers use ODEProblem
- `tran!(circuit, tspan)` now dispatches on solver type
- Tested: MOSFET with capacitances maintains DC equilibrium in transient

**Remaining:**
- Phase 3: Performance optimization (preallocate matrices)
- Phase 4: More MOSFET validation (junction caps, ring oscillator)

## Executive Summary

This document describes the plan for implementing proper nonlinear transient analysis in the MNA backend. The core infrastructure (`MNACircuit` + `DAEProblem`) already exists and is now exposed via user-friendly API.

Key changes:
1. `tran!(circuit, tspan)` dispatches automatically based on solver type:
   - DAE solvers (IDA, DFBDF): Uses DAEProblem (rebuilds matrices each Newton step)
   - ODE solvers (Rodas5P, etc.): Uses ODEProblem with mass matrix
2. MNACircuit no longer stores tspan (system vs problem separation)
3. Default solver is IDA (Sundials) for robustness

## Current Limitations

### The Problem

The current `tran!` function on `MNASim` uses **static matrix assembly**:

```julia
# src/sweeps.jl:458-474 (simplified)
function tran!(sim::MNASim, tspan; ...)
    sys = MNA.assemble!(sim)              # Assembled ONCE at t=0
    ode_data = MNA.make_ode_problem(sys)  # G, C, b captured as constants

    f = ODEFunction(ode_data.f;
        mass_matrix = ode_data.mass_matrix,  # C fixed at DC operating point
        jac = ode_data.jac)                  # -G fixed

    sol = solve(prob, solver)
end
```

**Consequence:** For nonlinear devices (MOSFETs, junction capacitors), the matrices never update during transient, leading to:
- Incorrect transient response for voltage-dependent capacitors
- Equilibrium drift from DC operating point
- Only valid for small-signal perturbations around DC

### What Works

The `MNACircuit` type with `DAEProblem` already implements proper nonlinear transient:

```julia
# src/mna/solve.jl:1235-1255
function make_dae_residual(circuit::MNACircuit)
    function dae_residual!(resid, du, u, p, t)
        # Rebuild circuit at CURRENT operating point and time
        spec_t = MNASpec(temp=base_spec.temp, mode=:tran, time=real_time(t))
        ctx = builder(params, spec_t; x=u)  # x=u enables nonlinear linearization
        sys = assemble!(ctx)

        # F(du, u) = C*du + G*u - b = 0 (with FRESH matrices)
        mul!(resid, sys.C, du)
        mul!(resid, sys.G, u, 1.0, 1.0)
        resid .-= sys.b
    end
end
```

**This is exactly what we need!** The path exists but isn't well-tested or exposed.

## How OpenVAF/SPICE Solvers Handle This

### OpenVAF Architecture

OpenVAF separates device contributions into:
- **Resistive Jacobian**: `∂I/∂V` (stamps into G)
- **Reactive Jacobian**: `∂q/∂V` (stamps into C, where q is charge)

For transient analysis:
```
J = G + γ*C
```
where γ is the integration coefficient (e.g., 1/h for backward Euler, 2/h for trapezoidal).

**Key insight:** Both G and C are recomputed at each Newton iteration within each timestep.

### Traditional SPICE Flow

```
for each timestep:
    predict x_{n+1} from history
    for newton_iter = 1:max_iters:
        stamp_devices(x_{n+1})  # Rebuild G, C, b at current guess
        J = G + γ*C
        solve: J * Δx = -F(x_{n+1})
        x_{n+1} += Δx
        if converged: break
    accept timestep
```

The matrices are rebuilt at **every Newton iteration**, not just every timestep.

## SciML Solver Mapping

### DAEProblem (Recommended)

```julia
# Residual: F(du, u, p, t) = 0
# For MNA: F = C*du + G*u - b

prob = DAEProblem(residual!, du0, u0, tspan; differential_vars=diff_vars)
sol = solve(prob, IDA())  # Sundials IDA for large circuits
```

**IDA behavior:**
- Calls `residual!` at each Newton iteration
- Our `make_dae_residual` rebuilds G, C, b each call → correct nonlinear handling
- Modified Newton: may reuse Jacobian across iterations (configurable)
- Adaptive BDF order 1-5

### ODEProblem with Mass Matrix

```julia
# M * du/dt = f(u, t)
# For MNA: C * du/dt = b - G*u

prob = ODEProblem(f, u0, tspan; mass_matrix=C)
sol = solve(prob, Rodas5P())  # Rosenbrock for singular M
```

**Limitation:** SciML's mass matrix is typically constant. State-dependent M(u) requires:
- `SciMLOperators.jl` with `update_func`
- Can cause order reduction and instability

**Recommendation:** Use DAEProblem for nonlinear circuits.

### Solver Comparison

| Solver | Package | Best For | Notes |
|--------|---------|----------|-------|
| IDA | Sundials.jl | Large nonlinear circuits | Industry standard, Float64 only |
| DFBDF | OrdinaryDiffEq.jl | Any Julia type | Slower, pure Julia |
| Rodas5P | OrdinaryDiffEq.jl | Singular mass matrices | ODE formulation, linear only |

## Implementation Plan

### Phase 1: Validate Existing DAEProblem Path

The `MNACircuit` + `DAEProblem` infrastructure exists. First, validate it works:

**1.1 Create test circuit with voltage-dependent capacitance:**
```julia
va"""
module VAVCap(p, n);
    parameter real C0 = 1e-12;
    parameter real Vc = 1.0;
    inout p, n;
    electrical p, n;
    analog begin
        // Voltage-dependent capacitance: C(V) = C0 / (1 + V/Vc)
        // Charge: q(V) = C0 * Vc * ln(1 + V/Vc)
        I(p,n) <+ ddt(C0 * Vc * ln(1 + V(p,n)/Vc));
    end
endmodule
"""
```

**1.2 Test DAEProblem transient:**
```julia
function vcap_circuit(params, spec; x=Float64[])
    ctx = MNAContext()
    vdd = get_node!(ctx, :vdd)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(1.0; name=:Vin), ctx, vdd, 0)
    stamp!(Resistor(1000.0; name=:R), ctx, vdd, out)
    stamp!(VAVCap(C0=1e-12, Vc=1.0), ctx, out, 0; x=x, spec=spec)

    return ctx
end

circuit = MNACircuit(vcap_circuit, (;), MNASpec(), (0.0, 1e-6))
prob = DAEProblem(circuit)
sol = solve(prob, IDA())
```

**1.3 Verify correctness:**
- Compare to analytical RC response with variable C
- Verify charge conservation

### Phase 2: High-Level API

Add convenient `tran!` overload that uses DAEProblem:

```julia
# src/mna/solve.jl

"""
    solve_tran(builder, params, spec, tspan; solver=IDA(), kwargs...)

Nonlinear transient analysis using DAE formulation.

Rebuilds circuit matrices at each Newton iteration, supporting:
- Voltage-dependent capacitors (MOSFET Cgs/Cgd)
- Nonlinear resistive elements
- Time-dependent sources
"""
function solve_tran(builder::F, params::P, spec::S, tspan::Tuple{Real,Real};
                    solver=nothing, abstol=1e-10, reltol=1e-8, kwargs...) where {F,P,S}
    circuit = MNACircuit(builder, params, spec, Float64.(tspan))
    prob = DAEProblem(circuit)

    if solver === nothing
        solver = IDA()  # Default to Sundials for robustness
    end

    return solve(prob, solver; abstol=abstol, reltol=reltol, kwargs...)
end

# Also update MNASim to use DAE path when devices are nonlinear
function tran!(sim::MNASim, tspan::Tuple{Real,Real};
               nonlinear::Bool=false, solver=nothing, kwargs...)
    if nonlinear
        # Use DAEProblem path for nonlinear devices
        circuit = MNACircuit(sim.builder, sim.params, sim.spec, Float64.(tspan))
        prob = DAEProblem(circuit)
        solver = solver === nothing ? IDA() : solver
        return solve(prob, solver; kwargs...)
    else
        # Existing linear path (unchanged)
        ...
    end
end
```

### Phase 3: Performance Optimization

Rebuilding the entire circuit at each residual call is expensive. Optimizations:

**3.1 Preallocate matrices:**
```julia
mutable struct MNACircuitCache
    ctx::MNAContext
    sys::MNASystem
    G::SparseMatrixCSC{Float64,Int}
    C::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
end

function make_dae_residual_cached(circuit::MNACircuit)
    # Preallocate with initial structure
    cache = MNACircuitCache(...)

    function dae_residual!(resid, du, u, p, t)
        # Reuse cache, only update values
        restamp_devices!(cache, u, t)  # Update G, C, b in-place

        mul!(resid, cache.C, du)
        mul!(resid, cache.G, u, 1.0, 1.0)
        resid .-= cache.b
    end
end
```

**3.2 Separate linear/nonlinear devices:**
```julia
# Only rebuild nonlinear device contributions
# Linear devices (R, C, L) stamp once and reuse
```

**3.3 Jacobian caching:**
```julia
# Let IDA's modified Newton reuse Jacobian when possible
# Only force refresh on convergence failure
```

### Phase 4: MOSFET Validation

Test realistic MOSFET circuits:

**4.1 Junction capacitor model:**
```julia
va"""
module VAJunctionCap(p, n);
    parameter real Cj0 = 1e-12;  // Zero-bias capacitance
    parameter real Vbi = 0.8;    // Built-in potential
    parameter real m = 0.5;      // Grading coefficient
    inout p, n;
    electrical p, n;
    analog begin
        // C(V) = Cj0 / (1 - V/Vbi)^m
        // q(V) = integral of C(V) dV
        I(p,n) <+ Cj0 * ddt(Vbi * (1 - pow(1 - V(p,n)/Vbi, 1-m)) / (1-m));
    end
endmodule
"""
```

**4.2 CMOS inverter with Miller capacitance:**
- NMOS and PMOS with voltage-dependent Cgd
- Verify transient delay matches expected behavior

**4.3 Ring oscillator:**
- Self-oscillating circuit
- Tests stability of nonlinear transient solver

### Phase 5: Documentation and Examples

1. Update `doc/mna_design.md` with nonlinear transient section
2. Add examples in `examples/` directory
3. Document solver selection guidelines

## Technical Details

### Charge-Based Formulation

For voltage-dependent capacitance, we use charge-based formulation:

```julia
# Instead of: I = C(V) * dV/dt  (problematic: C depends on V)
# We use:     I = dq/dt where q = q(V)  (correct: chain rule handles everything)

# In Verilog-A:
I(p,n) <+ ddt(q(V(p,n)));  # Correct: charge is function of voltage

# NOT:
I(p,n) <+ C(V(p,n)) * ddt(V(p,n));  # Wrong: C(V)*dV/dt != d(q(V))/dt
```

The s-dual approach in `va_ddt()` handles this correctly:
```julia
# va_ddt(q) returns Dual(0, q)
# - value = 0: no DC current (capacitor blocks DC)
# - partials = q: charge, differentiated by ForwardDiff gives C = ∂q/∂V
```

### Jacobian Structure

For DAE: `F(du, u) = C*du + G*u - b = 0`

The Jacobian for IDA:
```
J = ∂F/∂u + γ * ∂F/∂(du)
  = G + γ*C
```

For nonlinear circuits, both G and C depend on u:
```
J(u) = G(u) + γ*C(u)
```

Our `make_dae_jacobian` already handles this by rebuilding at each call.

### Integration with `x=` Parameter

VA devices receive `x=` for operating point:
```julia
stamp!(VADevice(...), ctx, p, n; x=x, spec=spec)
```

Within the device, `x[node_idx]` gives the voltage at that node. For transient:
- `x` is the current Newton iterate guess
- Device evaluates currents/charges at this operating point
- ForwardDiff extracts Jacobian w.r.t. these voltages

## Testing Strategy

### Unit Tests

1. **Voltage-dependent capacitor (VAVCap)**
   - Compare transient to analytical solution
   - Verify C matrix entries vary with voltage

2. **Junction diode capacitor**
   - PN junction depletion capacitance
   - Known physics for validation

3. **MOSFET capacitances**
   - Cgs, Cgd with voltage dependence
   - Compare to SPICE reference

### Integration Tests

1. **CMOS inverter transient**
   - Step input, measure delay
   - Compare to static analysis

2. **RC with nonlinear C**
   - Verify charge conservation
   - Compare time constants

3. **Ring oscillator**
   - Self-oscillating (no external stimulus)
   - Verify sustained oscillation

## Timeline

| Phase | Description | Complexity |
|-------|-------------|------------|
| 1 | Validate DAEProblem path | Low |
| 2 | High-level API | Low |
| 3 | Performance optimization | Medium |
| 4 | MOSFET validation | Medium |
| 5 | Documentation | Low |

## Appendix: Code Locations

| Component | File | Lines |
|-----------|------|-------|
| MNACircuit | src/mna/solve.jl | 1186-1197 |
| make_dae_residual | src/mna/solve.jl | 1235-1255 |
| make_dae_jacobian | src/mna/solve.jl | 1264-1283 |
| DAEProblem(circuit) | src/mna/solve.jl | 1379-1401 |
| ODEProblem(circuit) | src/mna/solve.jl | 1427-1475 |
| va_ddt() | src/mna/contrib.jl | 64-83 |
| tran! (current) | src/sweeps.jl | 458-474 |
