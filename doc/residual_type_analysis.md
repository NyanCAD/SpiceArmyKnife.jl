# MNA Residual Function Type Stability Analysis

## Executive Summary

**The residual function allocates ~2KB per call**, which explains why the VACASK benchmarks show high allocation counts and are slower than ngspice. The root cause is that `fast_rebuild!` calls the circuit builder function on every Newton iteration, creating a **new MNAContext** each time.

For a 1-second simulation with ~2M iterations, this translates to **~4GB of allocations** that need to be garbage collected!

## Investigation Results

### Allocation Measurements

| Function | Allocations per Call |
|----------|---------------------|
| `fast_residual!` | 2160 bytes |
| `fast_rebuild!` | 2192 bytes |
| `build_rc (builder)` | 2224 bytes |
| `MNAContext()` | 640 bytes |
| `update_sparse_from_coo!` | 48 bytes |
| `MNASpec(...)` | 0 bytes |

### Root Cause

The problem is in `fast_rebuild!` (precompile.jl:440-482):

```julia
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    # Creates a new spec (0 bytes - ok)
    spec_t = MNASpec(temp=pc.spec.temp, mode=:tran, time=real_time(t))

    # PROBLEM: Creates a brand new MNAContext!
    # This allocates ~2KB per call:
    # - 640 bytes for MNAContext itself
    # - Additional bytes for push! into vectors during stamping
    ctx = pc.builder(pc.params, spec_t; x=u)  # <-- ALLOCATES!

    # The rest copies values from ctx to pc (OK, no allocations)
    ...
end
```

The builder function (e.g., `build_rc`) creates:
1. A new `MNAContext` (640 bytes for empty vectors and dict)
2. Calls to `push!` for `G_I`, `G_J`, `G_V`, `C_I`, `C_J`, `C_V`, `b`, etc.
3. Dict operations for `node_to_idx`

### Type Stability Analysis

The `@code_warntype` output shows that `fast_residual!` and `fast_rebuild!` are **type-stable** - all types are fully inferred. The problem is not type instability but rather the **allocating operations** inside the builder call.

Key observations:
- All return types are concrete (`Nothing`, `MNAContext`, etc.)
- No `Any` types in the call chain
- The allocations come from creating new vectors and dicts, not from boxing

## Impact on VACASK Benchmarks

For a typical VACASK benchmark:
- 1M timepoints × 2 iterations/timepoint = 2M iterations
- 2160 bytes × 2M = **~4.3 GB allocations**
- Each allocation triggers GC pressure

This is why VACASK is slower than ngspice, which uses in-place updates.

## Recommended Solutions

### Option 1: Pre-allocated Value-Only Builder (Recommended)

**Concept**: Separate structure building from value updates. The builder runs once to create the structure, then a separate "value updater" function updates values in pre-allocated arrays.

```julia
# New fields in PrecompiledCircuit
struct PrecompiledCircuit{F,P,S,U}
    ...
    value_updater::U  # Function that updates values only
    ctx_cache::MNAContext  # Reused context (structure fixed, values updated)
end

# Builder returns TWO things: structure function + value function
function make_value_updater(builder::F, params::P, spec::S, ctx_template::MNAContext)
    # Create a closure that updates ctx_template in-place
    function update_values!(ctx, u, t)
        # Reset values to zero (keeps structure)
        fill!(ctx.G_V, 0.0)
        fill!(ctx.C_V, 0.0)
        fill!(ctx.b, 0.0)

        # Each stamp! call just updates existing indices
        # (would require stamp functions to check if index exists)
        ...
    end
    return update_values!
end
```

**Pros**: Clean separation of concerns, minimal code changes
**Cons**: Requires modifying all stamp! functions to support "update mode"

### Option 2: Cached Context with Reset

**Concept**: Store a cached MNAContext in PrecompiledCircuit, reset its values each iteration, and have the builder fill it rather than create a new one.

```julia
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    spec_t = MNASpec(temp=pc.spec.temp, mode=:tran, time=real_time(t))

    # Reset existing context instead of creating new one
    reset_values!(pc.ctx_cache)

    # Builder fills existing context (needs modified signature)
    pc.builder(pc.ctx_cache, pc.params, spec_t; x=u)

    # Copy values to sparse matrices
    ...
end
```

**Pros**: Single architectural change
**Cons**: Changes builder signature, all existing builders need updates

### Option 3: Direct Stamp Function Updates

**Concept**: During compilation, record which closure evaluates each stamp. At runtime, re-evaluate closures and directly update COO arrays.

```julia
# During compilation, store stamp closures
struct PrecompiledStamp
    matrix::Symbol  # :G or :C
    row::Int
    col::Int
    value_fn::Function  # (params, spec, x) -> Float64
    coo_idx::Int
end

# At runtime, evaluate each stamp
function fast_rebuild!(pc::PrecompiledCircuit, u, t)
    for stamp in pc.stamps
        val = stamp.value_fn(pc.params, spec, u)
        pc.G_V[stamp.coo_idx] = val  # or C_V
    end
    ...
end
```

**Pros**: Minimal allocations, most efficient
**Cons**: Major architectural change, complex implementation

### Option 4: SPICE Codegen Modification

**Concept**: Modify `parse_spice_to_mna` to generate two functions: one for structure, one for values.

```julia
# Generated code becomes:
function circuit_structure(params)
    # Called once - creates nodes and determines COO structure
end

function circuit_values!(G_V, C_V, b, params, spec, x)
    # Called each iteration - fills pre-allocated arrays
end
```

**Pros**: Works with existing stamp! pattern
**Cons**: Only helps SPICE-generated circuits, not hand-written builders

## Implementation Roadmap

### Phase 1: Quick Win (Option 2 variant)

Add a "stateful builder" mode where the context is passed in:

1. Add optional `ctx` parameter to builder signature:
   ```julia
   function build_rc(params, spec; x=Float64[], ctx=nothing)
       ctx = ctx === nothing ? MNAContext() : (clear!(ctx); ctx)
       ...
   end
   ```

2. Store cached context in PrecompiledCircuit
3. Pass cached context to builder in fast_rebuild!

**Estimated impact**: Reduces allocations by ~70% (from 2160 to ~640 bytes per call)

### Phase 2: Full Solution (Option 1)

1. Create `StampContext` type that tracks positions in pre-allocated arrays
2. Modify stamp! functions to work with StampContext
3. Generate value-only updaters during compilation

**Estimated impact**: Near-zero allocations per residual call

### Phase 3: SPICE/VA Integration

1. Modify SPICE codegen to generate dual-mode builders
2. Modify VA codegen similarly
3. Update all device models

## Benchmarking Plan

Create a micro-benchmark to validate improvements:

```julia
# Before optimization
@time for _ in 1:100_000
    fast_residual!(resid, du, u, pc, t)
end

# After optimization
@time for _ in 1:100_000
    fast_residual!(resid, du, u, pc, t)
end
```

Target: < 100 bytes per residual call (matching ngspice-level performance).

## Appendix: Type Stability Verification

`@code_warntype` output confirms the code is fully type-stable:

```julia
# fast_residual! - all types are concrete
Body::Nothing
1 ─ CedarSim.MNA.fast_rebuild!(pc, u, t)
    %2 = CedarSim.MNA.mul!::Core.Const(LinearAlgebra.mul!)
    %3 = Base.getproperty(pc, :C)::SparseMatrixCSC{Float64, Int64}
    ...
```

The allocations are due to object creation (MNAContext, vectors), not type instability.
