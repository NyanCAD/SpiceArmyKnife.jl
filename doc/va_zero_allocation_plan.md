# Zero-Allocation Verilog-A Evaluation

## Current Status (January 2026)

### Completed: ValueOnlyContext - Zero-Allocation COO Stamping

Three optimization phases have been implemented:

1. **Phase 1: MNAContext Reuse** - Store and reuse context (71% reduction)
2. **Phase 2: ValueOnlyContext** - Eliminate push! allocations (100% COO allocation eliminated)
3. **Remaining: Device Struct Creation** - Still allocates in generated code (~472 bytes/iter)

**Benchmark Results (VACASK RC circuit, 1s transient with dt=1µs):**

| Phase | Bytes/iteration | Reduction |
|-------|-----------------|-----------|
| Original | 6,314 | baseline |
| MNAContext reuse | 1,838 | 71% |
| ValueOnlyContext | 472* | 93% |
| Target | 0 | 100% |

*Remaining 472 bytes comes from device struct creation in generated code, NOT from stamping.

### Implementation: ValueOnlyContext

The `ValueOnlyContext` type (in `src/mna/value_only.jl`) provides true zero-allocation
stamping by writing directly to pre-sized arrays:

```julia
mutable struct ValueOnlyContext{T}
    # Reference data from MNAContext
    node_to_idx::Dict{Symbol,Int}
    n_nodes::Int
    n_currents::Int

    # Pre-sized value arrays (no push!)
    G_V::Vector{T}
    C_V::Vector{T}
    b::Vector{T}
    b_V::Vector{T}  # deferred b stamps

    # Write positions (reset each iteration)
    G_pos::Int
    C_pos::Int
    b_deferred_pos::Int
    current_pos::Int
end
```

Specialized `stamp_G!`, `stamp_C!`, `stamp_b!` methods write at tracked positions:
```julia
@inline function stamp_G!(vctx::ValueOnlyContext{T}, i, j, val) where T
    iszero(i) && return nothing
    iszero(j) && return nothing
    pos = vctx.G_pos
    @inbounds vctx.G_V[pos] = extract_value(val)
    vctx.G_pos = pos + 1
    return nothing
end
```

The `AnyMNAContext = Union{MNAContext, ValueOnlyContext}` type alias allows device
stamp! methods to work with either context type without code duplication.

### Changes Made

1. **`src/mna/value_only.jl`** (NEW):
   - `ValueOnlyContext` type for zero-allocation stamping
   - Specialized stamp methods that write to pre-sized arrays
   - `create_value_only_context(ctx)` to create from MNAContext
   - `reset_value_only!(vctx)` to reset for new iteration
   - `AnyMNAContext` type alias for Union{MNAContext, ValueOnlyContext}

2. **`src/mna/devices.jl`**: All stamp! methods updated to accept `AnyMNAContext`

3. **`src/mna/context.jl`**: Added `reset_for_restamping!(ctx)` function

4. **`src/spc/codegen.jl`**: Builder accepts `ctx::Union{MNAContext, ValueOnlyContext, Nothing}`

5. **`src/mna/precompile.jl`**:
   - Added `b_deferred_resolved::Vector{Int}` to `CompiledStructure`
   - Added `vctx::ValueOnlyContext` and `supports_value_only_mode::Bool` to `EvalWorkspace`
   - Three-tier `fast_rebuild!`: value-only → ctx reuse → fallback

## Memory Profiling Results (January 2026)

### Current Allocation Breakdown

Profiling the VACASK RC benchmark shows:

| Component | Bytes/iteration | Status |
|-----------|-----------------|--------|
| COO push! operations | 0 | ✅ ELIMINATED by ValueOnlyContext |
| Dictionary lookups | 0 | ✅ Reuses existing node_to_idx |
| Sparse matrix updates | 0 | ✅ In-place via COO→nz mapping |
| Device struct creation | ~300 | ❌ Still allocating (PULSE arrays) |
| Other builder overhead | ~172 | ❌ let blocks, temporaries |
| **Total fast_rebuild!** | ~472 | 93% reduction from original |

### Remaining Allocation Source: Device Creation

The generated SPICE code creates device structs every iteration:

```julia
# Every call allocates:
let v1 = 0.0, v2 = 1.0, td = 1e-6, tr = 1e-6, tf = 1e-6, pw = 1e-3, per = 2e-3
    times = [0.0, td, td + tr, td + tr + pw, td + tr + pw + tf, per]  # 48 bytes
    values = [v1, v1, v2, v2, v1, v1]                                  # 48 bytes
    stamp!(PWLVoltageSource(times, values; name = :vs), ...)           # copies arrays
end
stamp!(Resistor(1000.0; name = :r1), ...)   # creates struct
stamp!(Capacitor(1e-6; name = :c1), ...)    # creates struct
```

### What ValueOnlyContext Eliminates

The ValueOnlyContext eliminates ALL stamping allocations:

| Operation | MNAContext | ValueOnlyContext |
|-----------|------------|------------------|
| `stamp_G!(ctx, i, j, val)` | push! to G_I, G_J, G_V | write to G_V[pos++] |
| `stamp_C!(ctx, i, j, val)` | push! to C_I, C_J, C_V | write to C_V[pos++] |
| `stamp_b!(ctx, i, val)` | push! or accumulate | write or accumulate |
| `get_node!(ctx, name)` | Dict get!/set! | Dict lookup only |
| `alloc_current!(ctx, name)` | push! to current_names | return next index |

For circuits without dynamic device creation (e.g., hand-written builders or optimized
codegen), ValueOnlyContext achieves true zero allocation.

## Next Step: Phase 4 - Constant Folding via Tuples

The remaining 472 bytes/iter comes from device struct creation. The key insight is that
PWL times/values are **compile-time constants** and should use tuples for stack allocation.

```julia
# Current codegen (heap allocates every call):
let v1 = 0.0, v2 = 1.0, td = 1e-6, tr = 1e-6, tf = 1e-6, pw = 1e-3, per = 2e-3
    times = [0.0, td, td + tr, td + tr + pw, td + tr + pw + tf, per]   # allocates
    values = [v1, v1, v2, v2, v1, v1]                                   # allocates
    stamp!(PWLVoltageSource(times, values; name=:vs), ctx, p, n; t=t)   # copies
end

# Target codegen (stack-allocated, constant-folded):
# Use ntuple for fixed-size arrays that compiler can constant-fold
stamp!(PWLVoltageSource(
    (0.0, 1e-6, 2e-6, 1.002e-3, 1.003e-3, 2e-3),  # NTuple{6,Float64} - stack
    (0.0, 0.0, 1.0, 1.0, 0.0, 0.0);               # NTuple{6,Float64} - stack
    name=:vs), ctx, p, n; t=t)
```

**Implementation approach**:
1. Generate tuples instead of vectors for constant PWL data
2. Modify PWLVoltageSource to accept NTuple or SVector
3. Consider separating constant vs variable parts of stamp!:
   ```julia
   # Constant part (node resolution, structure) - can be inlined/constant-folded
   p_idx = get_node!(ctx, :p)
   n_idx = get_node!(ctx, :n)
   i_idx = alloc_current!(ctx, :vs)

   # Variable part (value evaluation) - minimal work per iteration
   v = pwl_eval(times, values, t)
   stamp_G!(ctx, ..., 1.0)  # ideal source: conductance is 1.0
   stamp_b!(ctx, i_idx, v)
   ```

**Estimated improvement**: Eliminate remaining ~472 bytes/iter → 0 bytes/iter

## Completed Phases

### Phase 1: MNAContext Reuse ✅

Store and reuse MNAContext instead of allocating new one each Newton iteration.
- 71% reduction (6314 → 1838 bytes/iter)

### Phase 2: Dictionary Handling ✅

ValueOnlyContext reuses the node_to_idx dictionary from MNAContext without modification.
- No allocations during lookup (existing keys only)

### Phase 3: ValueOnlyContext ✅

Eliminate push! allocations with direct array writes:
- `stamp_G!`, `stamp_C!`, `stamp_b!` write to pre-sized arrays
- Position counters instead of push!
- 74% additional reduction (1838 → 472 bytes/iter)

## Future Optimizations

### Phase 5: Compile-Time Dual Specialization

The generated `stamp!` method creates Duals with inferrable but not explicit sizes:

```julia
# Current - size known but not propagated:
partials_tuple = Expr(:tuple, [k == i ? 1.0 : 0.0 for k in 1:n_all_nodes]...)
$node_sym = Dual{JacobianTag}($(Symbol("V_", i)), $partials_tuple...)

# Better - explicit size for compile-time specialization:
$node_sym = Dual{JacobianTag,Float64,$n_all_nodes}(
    $(Symbol("V_", i)),
    Partials{$n_all_nodes,Float64}($partials_tuple))
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
```

## Target Architectures

| Use Case | Array Type | Formulation | Current | Target |
|----------|------------|-------------|---------|--------|
| CPU standard | `Vector{T}`, `SparseMatrixCSC` | In-place `f!(resid, du, u, p, t)` | 472 B/iter | 0 B/iter |
| GPU single | `CuArray{T}`, `CuSparseMatrixCSC` | In-place `f!(resid, du, u, p, t)` | N/A | 0 B/iter |
| GPU ensemble | `SVector{N,T}`, `SMatrix{N,N,T}` | Out-of-place `f(u, p, t) -> SVector` | N/A | 0 B/iter |

## Files Modified

| File | Status | Changes |
|------|--------|---------|
| `src/mna/value_only.jl` | ✅ NEW | ValueOnlyContext type, specialized stamp methods, AnyMNAContext alias |
| `src/mna/context.jl` | ✅ Done | Added `reset_for_restamping!()` |
| `src/mna/devices.jl` | ✅ Done | All stamp! methods accept AnyMNAContext |
| `src/spc/codegen.jl` | ✅ Done | Builder accepts ctx::Union{MNAContext,ValueOnlyContext,Nothing} |
| `src/mna/precompile.jl` | ✅ Done | ValueOnlyContext in EvalWorkspace, three-tier fast_rebuild! |
| `src/mna/MNA.jl` | ✅ Done | Include value_only.jl |

## Files To Modify (Phase 4)

| File | Status | Changes |
|------|--------|---------|
| `src/spc/codegen.jl` | Planned | Generate device structs at module level, not per-call |
| `src/mna/devices.jl` | Planned | Add stamp! methods that accept value-only parameters |

## Open Questions

1. **Conditional contributions**: VA `if` blocks that add/remove contributions change sparsity
   pattern. May need worst-case pattern with zeros.

2. **ddx()**: The VA `ddx()` function computes partial derivatives - may need special handling.

3. **Time-dependent sources**: PWL, SIN sources change `b` vector values but not structure -
   fits the value-only pattern.

4. **Internal node aliasing**: Short-circuit detection currently modifies structure at runtime -
   needs to be part of pattern discovery.
