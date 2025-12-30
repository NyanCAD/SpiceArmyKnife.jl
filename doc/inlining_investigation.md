# @inline Investigation for Allocation Reduction

## Summary

This document summarizes the investigation into how careful application of `@inline` can reduce allocations in MNA residual functions, particularly for Verilog-A modules.

## Key Insight

If Julia can see the definition and use of a struct (through inlining), it can constant-fold its properties, eliminating:
1. Runtime type dispatch on Dual numbers
2. Conditional branches based on contribution types (ContributionTag vs pure Dual vs scalar)
3. Unnecessary allocations for intermediate values

## Changes Made

### 1. contrib.jl - Contribution Functions

Added `@inline` to the following functions that were previously NOT inlined:

| Function | Line | Purpose |
|----------|------|---------|
| `stamp_contribution!` | 151 | Low-level stamping with pre-computed values |
| `evaluate_contribution` | 220 | Evaluates contribution and extracts Jacobians via AD |
| `stamp_current_contribution!` | 316 | Main entry point for VA contribution stamping |
| `stamp_voltage_contribution!` | 359 | Stamps voltage contributions with current variables |
| `evaluate_charge_contribution` | 420 | Evaluates charge function and extracts capacitances |
| `stamp_charge_contribution!` | 489 | Stamps voltage-dependent charge contributions |
| `stamp_multiport_charge!` | 563 | Stamps multi-terminal charge contributions |

These functions were already calling inlined primitives (`stamp_G!`, `stamp_C!`, `stamp_b!`, `va_ddt`) but weren't inlined themselves, creating a barrier to type inference.

### 2. vasim.jl - Generated stamp! Method

Added `@inline` to the generated `stamp!` method at line 1733.

This is **critical** because the generated code contains runtime type dispatch:

```julia
if I_branch isa ForwardDiff.Dual{CedarSim.MNA.ContributionTag}
    # Has ddt: ContributionTag wraps voltage duals
    I_resist = ForwardDiff.value(I_branch)
    ...
elseif I_branch isa ForwardDiff.Dual
    # Pure resistive: just voltage dual
    ...
else
    # Scalar result
    ...
end
```

With `@inline`, when the caller can see the type of `I_branch` (which is determined statically by whether `va_ddt()` is called in the contribution expression), these branches can be eliminated at compile time.

### 3. precompile.jl - Update Functions

Added `@inline` to:

| Function | Line | Purpose |
|----------|------|---------|
| `update_sparse_from_coo!` | 378 | Updates sparse matrix values in-place from COO |
| `update_b_vector!` | 400 | Updates b vector from direct and deferred stamps |

These are called during every Newton iteration in `fast_rebuild!`.

## Why Inlining Helps

### Before Inlining

When `evaluate_contribution` is not inlined, Julia sees:

```julia
result = evaluate_contribution(contrib_fn, Vp, Vn)
```

The return type is a NamedTuple with several Float64 fields, but Julia can't specialize on whether `has_reactive` is true or false at compile time.

### After Inlining

With `@inline`, the entire evaluation chain inlines:

```julia
# All of this gets inlined:
Vp_dual = Dual{JacobianTag}(Vp, one(Vp), zero(Vp))
Vn_dual = Dual{JacobianTag}(Vn, zero(Vn), one(Vn))
result = contrib_fn(Vpn_dual)  # Type known at compile time!

if result isa ForwardDiff.Dual{ContributionTag}
    # This branch can be constant-folded if contrib_fn's return type is known
    ...
end
```

Since `contrib_fn` typically contains `va_ddt()` calls (or not), its return type is statically determined, allowing the type checks to be eliminated.

## Relationship to va_zero_allocation_plan.md

This investigation corresponds to **Phase 3** of the zero-allocation plan:

> **Phase 3: Aggressive Inlining for Constant Folding**
>
> The current code uses runtime `isa` checks that can be constant-folded with aggressive inlining.

The changes in this PR enable:
1. Type inference to propagate through the evaluation chain
2. Runtime type checks to become compile-time decisions
3. Dual number operations to be lowered to simple scalar arithmetic

## Test Results

All tests pass with the @inline changes:
- **MNA Core Tests**: 351/351 passed
- **MNA VA Integration Tests**: 49/49 passed

## Remaining Work

While `@inline` helps with constant folding, the main allocation source remains:

```julia
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    ctx = pc.builder(pc.params, spec_t; x=u)  # <-- Allocates ~2KB per call
    ...
end
```

Phase 4 of the zero-allocation plan addresses this by adding a "value-only stamping mode" where devices write directly to pre-discovered COO locations without creating an MNAContext.

## Expected Impact

The `@inline` changes should:
1. **Reduce dynamic dispatch** in the contribution evaluation path
2. **Enable branch elimination** for devices with known ddt/no-ddt patterns
3. **Improve code quality** by letting LLVM see through the abstraction layers

Actual allocation reduction depends on how well Julia can specialize the code paths. The main allocation in `fast_rebuild!` (creating MNAContext) requires Phase 4 changes to eliminate.

## Files Modified

| File | Changes |
|------|---------|
| `src/mna/contrib.jl` | Added `@inline` to 7 functions |
| `src/mna/precompile.jl` | Added `@inline` to 2 functions |
| `src/vasim.jl` | Added `@inline` to generated `stamp!` method |
