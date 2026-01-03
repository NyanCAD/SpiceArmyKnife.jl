# Zero-Allocation Verilog-A Evaluation

## Current Status (January 2026)

### ✅ COMPLETE: Near-Zero-Allocation Circuit Evaluation

Four optimization phases have been implemented to achieve **near-zero-allocation** circuit evaluation:

1. **Phase 1: MNAContext Reuse** - Store and reuse context (71% reduction)
2. **Phase 2: ValueOnlyContext** - Eliminate push! allocations (100% COO allocation eliminated)
3. **Phase 3: Dictionary Reuse** - Reuse node_to_idx without modification
4. **Phase 4: SVector PWL Sources** - Use StaticArrays SVector for PWL times/values (91% reduction from Phase 3)

**Benchmark Results (VACASK RC circuit):**

| Phase | Bytes/iteration | Reduction |
|-------|-----------------|-----------|
| Original | 6,314 | baseline |
| MNAContext reuse | 1,838 | 71% |
| ValueOnlyContext | 472 | 93% |
| **SVector PWL (current)** | **40** | **99.4%** |

**Notes:**
- 16 bytes overhead from function barrier dispatch (affects all circuits)
- Additional 24 bytes from PWL type specialization (SVector parameterization)
- Simple RC circuits (no PWL) achieve ~16 bytes/iter
- All tests pass (351 MNA core, 49 VA integration)

**Circuit evaluation functions now allocate minimal bytes:**
- `fast_rebuild!()`: 40 bytes for PWL circuits, 16 bytes for simple circuits
- `fast_residual!()`: 0 bytes
- `fast_jacobian!()`: 0 bytes

The 40 bytes/iteration overhead is acceptable and represents a 99.4% reduction in
GC pressure compared to the original implementation.

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

## Phase 4: SVector PWL Sources ✅ (Completed January 2026)

The remaining 472 bytes/iter came from device struct creation. PWL times/values were
heap-allocated arrays.

### Solution: SVector with pwl_at_time

Use StaticArrays `SVector` for PWL times/values. SVector is stack-allocated and works
with `searchsortedfirst` for O(log n) lookup. The `pwl_at_time` function (from
CedarSim.spectre_env.jl) handles interpolation with proper edge cases.

**Before (heap allocated every call):**
```julia
let v1 = 0.0, v2 = 1.0, td = 1e-6, tr = 1e-6, tf = 1e-6, pw = 1e-3, per = 2e-3
    times = [0.0, td, td + tr, td + tr + pw, td + tr + pw + tf, per]   # allocates
    values = [v1, v1, v2, v2, v1, v1]                                   # allocates
    stamp!(PWLVoltageSource(times, values; name=:vs), ctx, p, n; t=t)   # copies
end
```

**After (SVector stack-allocated):**
```julia
stamp!(PWLVoltageSource(
    SVector{6,Float64}(0.0, td, td+tr, td+tr+pw, td+tr+pw+tf, per),
    SVector{6,Float64}(v1, v1, v2, v2, v1, v1);
    name=:vs), ctx, p, n; t=t, _sim_mode_=spec.mode)
```

### Changes Made

1. **`src/mna/devices.jl`**:
   - `PWLVoltageSource{T,V}` and `PWLCurrentSource{T,V}` - parameterized on storage type
   - SVector constructor preserves type (no conversion to Vector)
   - `pwl_at_time(ts, ys, t)` - uses `searchsortedfirst` for O(log n) lookup
   - Removed specialized `stamp_pwl_voltage!` / `stamp_pwl_current!` (use regular stamp!)

2. **`src/spc/codegen.jl`**: Updated PULSE codegen for constant parameters
   - Pre-computes PWL times/values as SVector literals at codegen time
   - Uses regular `stamp!(PWLVoltageSource(...), ...)` call
   - Falls back to dynamic `PWLVoltageSource` for variable parameters

**Result**: Reduced from 472 bytes/iter → **40 bytes/iter** (91.5% reduction)

The remaining 40 bytes includes:
- 16 bytes from Float64 boxing when passing time through dynamic function call
- 4 bytes from String allocation in keyword argument processing
- ~20 bytes from PWL type specialization overhead

### Remaining Allocation Sources (Profiled)

Using `Profile.Allocs`, the exact allocation sources per iteration are:

| Type | Bytes | Source |
|------|-------|--------|
| Float64 | 16 | Boxing `ws.time` when calling `cs.builder(...)` |
| String | 4 | Keyword argument processing overhead |

These allocations occur in `fast_rebuild!` at line 900:
```julia
cs.builder(cs.params, cs.spec, ws.time; x=u, ctx=vctx)
```

The builder function is stored in `CompiledStructure{F,P,S}` as a type parameter,
but calling it with keyword arguments causes Julia to box the Float64 time value.

## Phase 5: True Zero-Allocation for GPU ❌ (Not Started)

**GPU requires absolutely zero heap allocation.** The current 20+ bytes/iter is unacceptable.

### Root Cause: Dynamic Function Dispatch

The fundamental issue is that `cs.builder` is a function stored at runtime.
Even though it's a type parameter, the call still goes through Julia's
dynamic dispatch machinery, which boxes arguments.

### Required Changes for GPU

1. **Compile-Time Builder Inlining**

   Replace runtime function calls with `@generated` functions that inline
   the builder at compile time:

   ```julia
   @generated function gpu_rebuild!(ws::EvalWorkspace{T,CS}, u, t) where {T,CS}
       # Extract builder from type parameter
       F = CS.parameters[1]  # Builder function type

       # Generate inlined stamp calls - NO function pointer
       quote
           vctx = ws.vctx
           reset_value_only!(vctx)

           # Inlined builder code here (generated from F)
           # All stamp! calls become direct, no dispatch

           # Copy to workspace...
       end
   end
   ```

2. **Eliminate Keyword Arguments**

   Keyword arguments cause allocation. Replace with:
   - Positional arguments only, OR
   - A context struct that holds time/mode:

   ```julia
   struct StampContext
       t::Float64
       mode::Symbol
   end
   # Pass as single positional arg, not kwargs
   ```

3. **Static Time Access**

   Instead of passing time as argument, read from a fixed location:

   ```julia
   # Store time in workspace, read directly in stamp!
   @inline get_time(ws::EvalWorkspace) = ws.time
   ```

4. **Fully Static Circuit Topology**

   For GPU, the circuit structure must be completely static:
   - No Dict lookups (node_to_idx)
   - All node indices known at compile time
   - Use generated functions to create specialized code per circuit

### GPU Architecture Sketch

```julia
# GPU-compatible circuit evaluation
struct GPUCircuit{N,M,G_NNZ,C_NNZ}
    # All sizes known at compile time
    G_I::SVector{G_NNZ,Int32}
    G_J::SVector{G_NNZ,Int32}
    C_I::SVector{C_NNZ,Int32}
    C_J::SVector{C_NNZ,Int32}
    b_indices::SVector{M,Int32}
end

# Generated function that inlines everything
@generated function gpu_evaluate!(
    G_V::MVector{G_NNZ,T},
    C_V::MVector{C_NNZ,T},
    b::MVector{N,T},
    circuit::GPUCircuit{N,M,G_NNZ,C_NNZ},
    u::SVector{N,T},
    t::T
) where {N,M,G_NNZ,C_NNZ,T}
    # Generate completely inlined stamp code
    # No function calls, no allocations
end
```

This would require significant refactoring but is necessary for GPU execution.

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

---

## VA Device Zero-Allocation: Remaining Work (January 2026)

### Problem: VA stamp! Only Accepts MNAContext

While the value-only mode infrastructure is complete, **VA-generated stamp! methods
only accept `MNAContext`**, not `ValueOnlyContext`.

**Location:** `src/vasim.jl:1891`
```julia
function CedarSim.MNA.stamp!(dev::$symname, ctx::CedarSim.MNA.MNAContext, ...)
```

When `supports_value_only_ctx()` tests passing a `ValueOnlyContext`, the call fails
with `MethodError` and the code falls back to `MNAContext` reuse.

### Experimental Fix Attempted

**Change 1:** VA stamp! method signature
```julia
# Before:
function CedarSim.MNA.stamp!(dev::$symname, ctx::CedarSim.MNA.MNAContext, ...)

# After:
function CedarSim.MNA.stamp!(dev::$symname, ctx::CedarSim.MNA.AnyMNAContext, ...)
```

**Change 2:** Add `AnyMNAContext` import to generated module.

**Result:** Simple diode test passed, but hit a second issue...

### The Charge Detection Problem

VA devices with `ddt()` terms need `detect_or_cached!()` to determine if capacitance
is voltage-dependent:

```julia
# In generated stamp! code:
_is_vdep = detect_or_cached!(ctx, :Q_branch1, contrib_fn, Vp, Vn)
if _is_vdep
    # Use charge formulation (allocates charge variable)
else
    # Use C matrix directly
end
```

**Problem:** `detect_or_cached!()` only works with `MNAContext` because it needs the
`charge_is_vdep` cache (a `Dict{Symbol,Bool}`). `ValueOnlyContext` doesn't have this cache,
and a Dict doesn't fit the value-only paradigm anyway (dynamic allocation, hash lookups).

### Proper Fix: Index-Based Charge Detection

Keep the mutable Dict in MNAContext for first-build detection, then bake results into
a static tuple for ValueOnlyContext. Use the **same index** for both stamping and
charge detection lookup.

**Key insight:** The branch index used for COO stamping (1, 2, 3, ...) can also index
into the charge detection results. No need for Symbol-based lookup.

**First build (MNAContext):**
```julia
# Detection runs with Dict (flexible, mutable)
_is_vdep = detect_or_cached!(ctx, branch_idx, contrib_fn, Vp, Vn)
# Stores: ctx.charge_is_vdep[branch_idx] = result
```

**Compilation (create ValueOnlyContext):**
```julia
# Convert Dict to static tuple, ordered by branch index
charge_tuple = ntuple(i -> get(ctx.charge_is_vdep, i, false), n_branches)

# ValueOnlyContext stores the baked tuple
vctx = ValueOnlyContext{Float64, typeof(charge_tuple)}(
    ...,
    charge_tuple,  # NTuple{N,Bool} - statically sized
)
```

**Value-only builds:**
```julia
# stamp! method specialized on ValueOnlyContext{T, NTuple{N,Bool}}
# Compiler can inline the tuple lookup
@inline is_charge_vdep(vctx::ValueOnlyContext, i::Int) = vctx.charge_is_vdep[i]

# Generated stamp code uses same index for both:
_is_vdep = is_charge_vdep(ctx, branch_idx)  # O(1) indexed lookup
# ... stamp at position branch_idx ...
```

**Implementation steps:**

1. **Codegen (vasim.jl):** Assign each branch a sequential index (1, 2, 3, ...)
2. **MNAContext:** Keep Dict for detection, but key by index not Symbol
3. **create_value_only_context():** Convert Dict → NTuple, parameterize ValueOnlyContext
4. **stamp! methods:** Use `is_charge_vdep(ctx, idx)` that dispatches on context type
5. **Update VA stamp! signature** to accept `AnyMNAContext`

This approach:
- MNAContext keeps Dict for flexible first-build detection
- ValueOnlyContext gets baked NTuple - zero allocation, O(1) lookup
- Same index for stamping and detection - no Symbol overhead
- Tuple is a type parameter → compiler can specialize and inline

### Internal Node Indexing Investigation (January 2026)

The user raised a concern: "we have to be a bit careful because we defer branch indices
because internal nodes can create new ones and invalidate them, so we need to make sure
we use the index as compiled rather than the reference."

**Investigation findings:**

1. **Current charge name approach uses Symbols, not indices:**
   ```julia
   # vasim.jl:1452-1453
   charge_name_suffix = QuoteNode(Symbol(symname, "_Q_", p_sym, "_", n_sym))
   charge_name_expr = :(_mna_instance_ == Symbol("") ? $charge_name_suffix : ...)
   ```
   The charge name is built from port SYMBOLS (p_sym, n_sym) and module name - these are
   STABLE because they come from VA module definition, not runtime node indices.

2. **Internal node indices are allocated at runtime:**
   ```julia
   # vasim.jl:1677
   $int_param = CedarSim.MNA.alloc_internal_node!(ctx, $alloc_name_expr)
   ```
   The index returned depends on allocation ORDER. Different execution orders would
   yield different indices.

3. **Counter-based approach is safe:**
   The `ValueOnlyContext` already uses counter-based access for:
   - `alloc_current!`: Returns sequential CurrentIndex(pos++)
   - `stamp_G!`: Writes to G_V[G_pos++]
   - `stamp_C!`: Writes to C_V[C_pos++]

   These work because **execution order is deterministic** - the same builder function
   always executes stamps in the same sequence.

**Key insight:** We don't need to map Symbol → index. We use the same counter-based
approach as `alloc_current!`:

```julia
# MNAContext: detection stores results sequentially
mutable struct MNAContext
    charge_is_vdep::Vector{Bool}  # Results in detection order
    charge_detection_pos::Int     # Counter (advanced by each detect_or_cached!)
end

# ValueOnlyContext: lookup by same position counter
mutable struct ValueOnlyContext{T,N}
    charge_is_vdep::NTuple{N,Bool}  # Baked from Vector
    charge_detection_pos::Int        # Counter (reset each rebuild)
end

# Unified interface - works for both:
@inline function detect_or_cached!(ctx::MNAContext, contrib_fn, Vp, Vn)::Bool
    pos = ctx.charge_detection_pos
    ctx.charge_detection_pos = pos + 1
    # Check if already detected
    if pos <= length(ctx.charge_is_vdep)
        return ctx.charge_is_vdep[pos]
    end
    # First time: run detection
    result = is_voltage_dependent_charge(contrib_fn, Vp, Vn)
    push!(ctx.charge_is_vdep, result)
    return result
end

@inline function detect_or_cached!(vctx::ValueOnlyContext{T,N}, contrib_fn, Vp, Vn)::Bool where {T,N}
    pos = vctx.charge_detection_pos
    vctx.charge_detection_pos = pos + 1
    return @inbounds vctx.charge_is_vdep[pos]  # Just lookup, no detection
end
```

**Why this is safe:**
1. Detection calls happen in DETERMINISTIC order (same builder = same order)
2. The counter tracks position, not a Symbol→index mapping
3. Internal node indices don't affect the counter (it counts detection calls, not nodes)
4. The baked NTuple preserves the same order as the Vector

**What changes from current design:**
1. Remove `charge_name_expr` argument from `detect_or_cached!` (not needed)
2. Use `Vector{Bool}` instead of `Dict{Symbol,Bool}` in MNAContext
3. Add position counter field to both context types
4. Generated code calls `detect_or_cached!(ctx, contrib_fn, Vp, Vn)` without name

**Alternative: Keep Symbol names, resolve at stamping time**

If we need to preserve Symbol-based identification for debugging:
```julia
# Keep Symbol for MNAContext (debugging, error messages)
detect_or_cached!(ctx::MNAContext, name::Symbol, contrib_fn, Vp, Vn)

# ValueOnlyContext ignores name, uses counter
detect_or_cached!(vctx::ValueOnlyContext, name::Symbol, contrib_fn, Vp, Vn)
    # name parameter ignored - just for API compatibility
    pos = vctx.charge_detection_pos
    ...
```

This maintains API compatibility while allowing zero-allocation operation
