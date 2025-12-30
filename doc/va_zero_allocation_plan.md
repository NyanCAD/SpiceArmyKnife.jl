# Zero-Allocation Verilog-A Evaluation

## Problem Statement

The current VA integration (`vasim.jl`) allocates ~2KB per Newton iteration. For a 1-second simulation with ~2M iterations, this means ~4GB of allocations and GC pressure.

**Root cause**: `fast_rebuild!` in `precompile.jl:440-482` creates a new `MNAContext` every Newton iteration:

```julia
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    spec_t = MNASpec(temp=pc.spec.temp, mode=:tran, time=real_time(t))
    ctx = pc.builder(pc.params, spec_t; x=u)  # <-- ALLOCATES ~2KB!
    # ... copies values from ctx to pc
end
```

## Key Insight: Duals Inline to Arithmetic

ForwardDiff Duals with compile-time known sizes **inline to simple scalar operations**:

```julia
# This Dual code:
V = Dual{Tag}(1.0, (1.0, 0.0))
I = V / R

# Compiles to equivalent of:
I_val = V_val / R
I_partial_1 = V_partial_1 / R  # = 1/R
I_partial_2 = V_partial_2 / R  # = 0
```

No heap allocation needed - just scalar arithmetic on the stack. The allocations come from:
1. Creating MNAContext every iteration (main cause)
2. Runtime type dispatch (`if x isa Dual{Tag}`)
3. Dynamic tuple sizes for partials

With proper specialization, all of these can be eliminated.

## Target Architectures

| Use Case | Array Type | Formulation | Allocation Target |
|----------|------------|-------------|-------------------|
| CPU standard | `Vector{T}`, `SparseMatrixCSC` | In-place `f!(resid, du, u, p, t)` | 0 bytes/iter |
| GPU single | `CuArray{T}`, `CuSparseMatrixCSC` | In-place `f!(resid, du, u, p, t)` | 0 bytes/iter |
| GPU ensemble | `SVector{N,T}`, `SMatrix{N,N,T}` | Out-of-place `f(u, p, t) -> SVector` | 0 bytes/iter |

## Implementation Plan

### Phase 1: Separate Structure from Values

The OSDI pattern from OpenVAF/VACASK: discover sparsity structure once, then evaluate values in-place.

**Current flow (allocating):**
```
Newton iteration → builder(params, spec; x=u) → new MNAContext → stamp! → copy to matrices
                   ↑ ALLOCATES
```

**Target flow (zero-allocation):**
```
Setup: builder(params, spec) → discover pattern → store COO indices + value pointers

Newton: evaluate_values!(pointers, params, x, t) → write directly through pointers
        ↑ NO ALLOCATION
```

**Changes to precompile.jl:**

```julia
struct PrecompiledCircuit{T}
    # Existing fields...

    # NEW: Pre-allocated evaluation workspace
    eval_ctx::EvalContext{T}  # Reusable, stores G/C/b value pointers
end

# Called once during setup
function compile_circuit(builder, params, spec)
    ctx = builder(params, spec)  # Structure discovery
    pattern = extract_pattern(ctx)
    eval_ctx = EvalContext(pattern)  # Pre-allocate workspace
    return PrecompiledCircuit(..., eval_ctx)
end

# Called every Newton iteration - ZERO ALLOCATION
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    evaluate_values!(pc.eval_ctx, pc.params, u, t)  # Write through pointers
end
```

### Phase 2: Fix Dual Sizes at Compile Time

The generated `stamp!` method creates Duals with inferrable but not explicit sizes:

```julia
# Current (vasim.jl:1588) - size known but not propagated:
partials_tuple = Expr(:tuple, [k == i ? 1.0 : 0.0 for k in 1:n_all_nodes]...)
$node_sym = Dual{JacobianTag}($(Symbol("V_", i)), $partials_tuple...)

# Better - explicit size for compile-time specialization:
$node_sym = Dual{JacobianTag,Float64,$n_all_nodes}(
    $(Symbol("V_", i)),
    Partials{$n_all_nodes,Float64}($partials_tuple))
```

### Phase 3: Aggressive Inlining for Constant Folding

The current code uses runtime `isa` checks that can be constant-folded:

```julia
# Current - runtime dispatch (vasim.jl:1434-1461):
if I_branch isa ForwardDiff.Dual{ContributionTag}
    # has ddt
elseif I_branch isa ForwardDiff.Dual
    # pure resistive
else
    # scalar
end
```

**With aggressive inlining, these branches constant-fold** because the type of `I_branch` is known at compile time from the contribution expression.

**Fix 1**: Add `@inline` to all VA evaluation functions:
```julia
@inline function va_ddt(x::Real)
    Dual{ContributionTag}(zero(x), x)
end
```

**Fix 2**: Mark generated stamp! for forced inlining:
```julia
@inline function stamp!(dev::$symname, ctx, nodes...; x, t, spec)
    # With @inline, the entire evaluation chain inlines
    # Type inference propagates through, branches constant-fold
end
```

### Phase 4: Value-Only Stamping Mode

Add a mode where `stamp!` writes only values to pre-discovered locations:

```julia
struct VADevice{P}
    params::P
    # COO indices discovered at setup
    G_indices::Vector{Tuple{Int,Int,Int}}  # (row, col, value_slot)
    C_indices::Vector{Tuple{Int,Int,Int}}
    b_indices::Vector{Tuple{Int,Int}}
end

# Structure discovery (called once)
function discover_pattern!(dev::VADevice, ctx, nodes...; x, t, spec)
    # Normal stamp! call - fills ctx COO arrays
    stamp!(dev, ctx, nodes...; x, t, spec)
    # Extract indices from ctx
    dev.G_indices = extract_indices(ctx.G)
    dev.C_indices = extract_indices(ctx.C)
    dev.b_indices = extract_indices(ctx.b)
end

# Value evaluation (called every iteration) - ZERO ALLOCATION
@inline function evaluate_values!(dev::VADevice, G_vals, C_vals, b_vals, x, t, spec)
    # Generate Duals, evaluate, write to pre-indexed slots
    # No MNAContext created!
end
```

### Phase 5: GPU Ensemble Support (StaticArrays)

For GPU ensemble (parameter sweeps, Monte Carlo), return fresh StaticArrays:

```julia
struct StaticVACircuit{N,T,F}
    eval_fn::F  # Returns (G::SMatrix, C::SMatrix, b::SVector)
    params::NamedTuple
end

# Zero allocation because SMatrix/SVector are stack-allocated
@inline function evaluate(vc::StaticVACircuit{N,T,F}, u::SVector{N,T}, t) where {N,T,F}
    G, C, b = vc.eval_fn(u, t, vc.params)
    return G, C, b  # All stack-allocated
end

# Residual for SciML out-of-place ODE
function residual(u::SVector{N,T}, p::StaticVACircuit{N,T}, t) where {N,T}
    G, C, b = evaluate(p, u, t)
    return G * u - b  # Returns SVector (stack-allocated)
end
```

**Key**: `@generated` function to create Jacobian Duals backed by StaticArrays:

```julia
@generated function create_jacobian_duals(u::SVector{N,T}) where {N,T}
    exprs = [:(Dual{JacobianTag,T,N}(u[$i],
               Partials{N,T}(ntuple(j -> j == $i ? one(T) : zero(T), Val($N)))))
             for i in 1:N]
    :(SVector{$N}($(exprs...)))
end
```

## Files to Modify

| File | Changes |
|------|---------|
| `src/mna/precompile.jl` | Add `EvalContext`, modify `fast_rebuild!` for value-only mode |
| `src/vasim.jl` | Add `@inline`, explicit Dual sizes, value-only `evaluate!` |
| `src/mna/contrib.jl` | Add `@inline` to `va_ddt()` and contribution helpers |

## Expected Results

| Metric | Current | After Implementation |
|--------|---------|---------------------|
| Bytes/iteration (simple VA) | ~2000 | 0 |
| Bytes/iteration (BSIM4) | ~5000 | 0 |
| GPU ensemble support | No | Yes |

## Open Questions

1. **Conditional contributions**: VA `if` blocks that add/remove contributions change sparsity pattern. May need worst-case pattern with zeros.

2. **ddx()**: The VA `ddx()` function computes partial derivatives - may need special handling.

3. **Time-dependent sources**: PWL, SIN sources change `b` vector values but not structure - fits the value-only pattern.

4. **Internal node aliasing**: Short-circuit detection currently modifies structure at runtime - needs to be part of pattern discovery.
