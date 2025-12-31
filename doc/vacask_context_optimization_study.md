# VACASK Context Allocation Optimization Study

## Problem Statement

The current implementation allocates ~2KB per Newton iteration in `fast_rebuild!`:

```julia
# src/mna/precompile.jl:440-443
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    spec_t = MNASpec(temp=pc.spec.temp, mode=:tran, time=real_time(t))  # Allocates!
    ctx = pc.builder(pc.params, spec_t; x=u)  # Allocates ~2KB in vectors!
    # ...
end
```

For a 1-second simulation with ~2M iterations, this means ~4GB of allocations and GC pressure.

### Allocation Sources

1. **MNASpec creation** (~80 bytes): Created just to update the `time` field
2. **MNAContext allocation** (~2KB): Creates new vectors every call:
   - `G_I, G_J, G_V::Vector` (~300 entries × 3 = ~7KB typical)
   - `C_I, C_J, C_V::Vector` (~100 entries × 3 = ~2KB typical)
   - `b::Vector`, `b_I::Vector`, `b_V::Vector`
   - `node_names::Vector{Symbol}`, `node_to_idx::Dict`

## Proposed Solution: Store EvalContext in `p`

### Current Architecture

```
DAE Solver calls:
    f!(resid, du, u, p, t)
        ↓
    make_compiled_dae_residual(pc) creates closure over pc
        ↓
    fast_residual!(resid, du, u, pc, t)
        ↓
    fast_rebuild!(pc, u, t)
        ↓
    ctx = pc.builder(pc.params, spec_t; x=u)  ← ALLOCATES
```

The `p` parameter in the DAE problem is currently unused - the `PrecompiledCircuit` is captured in a closure.

### Proposed Architecture

```
DAE Solver calls:
    f!(resid, du, u, p, t)
        ↓
    p::EvalContext contains:
        - Preallocated COO storage (G_V, C_V, b)
        - Reference to PrecompiledCircuit
        - Mutable time field
        ↓
    fast_residual!(resid, du, u, p, t)  # Uses p.eval_ctx directly
        ↓
    evaluate_values!(p.eval_ctx, u, t)  ← NO ALLOCATION
```

### EvalContext Design

```julia
"""
Pre-allocated evaluation context for zero-allocation Newton iterations.
"""
mutable struct EvalContext{T}
    # Reference to compiled structure (immutable)
    pc::PrecompiledCircuit

    # Preallocated COO value storage (reused each iteration)
    G_V::Vector{T}
    C_V::Vector{T}
    b_direct::Vector{T}
    b_deferred_V::Vector{T}

    # Current evaluation state (mutable)
    time::T

    # Working storage for device evaluation
    # (for VA devices that need intermediate storage)
    work::Vector{T}
end

function EvalContext(pc::PrecompiledCircuit{F,P,S}) where {F,P,S}
    EvalContext{Float64}(
        pc,
        zeros(Float64, pc.G_n_coo),
        zeros(Float64, pc.C_n_coo),
        zeros(Float64, pc.n),
        zeros(Float64, 16),  # Typical deferred b size
        0.0,
        zeros(Float64, 32)   # Working storage
    )
end
```

## Passing Time Explicitly

### Current Pattern (Allocating)

```julia
# Every iteration creates a new MNASpec
spec_t = MNASpec(temp=pc.spec.temp, mode=:tran, time=real_time(t))
ctx = pc.builder(pc.params, spec_t; x=u)
```

### Proposed Pattern A: Explicit Time Parameter

Change builder signature to accept time explicitly:

```julia
# Builder signature
function build_circuit(params, spec, t; x=Float64[])
    # spec.time is base time (e.g., for DC), t is current simulation time
end

# In fast_rebuild!
evaluate_values!(eval_ctx, pc.params, pc.spec, t, u)
```

**Pros**: Clear separation, no spec mutation
**Cons**: API change, breaks existing builders

### Proposed Pattern B: Mutable Time in Spec

Make `time` field mutable (or use Ref):

```julia
mutable struct MNASpecMut{T<:Real}
    const temp::Float64
    const mode::Symbol
    time::T  # Mutable
    # ... other const fields
end
```

**Pros**: Minimal API change
**Cons**: Mutable struct, potential thread-safety issues

### Proposed Pattern C: Time via EvalContext (Recommended)

Store time in EvalContext, builders read from context:

```julia
# EvalContext passed through
function evaluate_circuit!(eval_ctx::EvalContext, params, u)
    t = eval_ctx.time  # Read time from context
    # Devices access eval_ctx.time directly
end
```

**Pros**: Clean separation, zero allocation, thread-safe (each thread has own context)
**Cons**: Requires device API changes

## Zeroing Matrices: Type-Agnostic and Sparse-Friendly

### Current Approach (Already Optimal)

The current `update_sparse_from_coo!` already uses the optimal approach:

```julia
# src/mna/precompile.jl:378-391
function update_sparse_from_coo!(S::SparseMatrixCSC, V::Vector{Float64},
                                  mapping::Vector{Int}, n_entries::Int)
    nz = nonzeros(S)
    fill!(nz, 0.0)  # ← Zero nonzeros vector in-place

    @inbounds for k in 1:n_entries
        idx = mapping[k]
        if idx > 0
            nz[idx] += V[k]
        end
    end
end
```

### Type-Agnostic Zeroing Patterns

| Method | Works For | Complexity | Allocation |
|--------|-----------|------------|------------|
| `fill!(nonzeros(S), zero(T))` | `SparseMatrixCSC{T}` | O(nnz) | 0 |
| `fill!(v, zero(eltype(v)))` | `Vector{T}` | O(n) | 0 |
| `S.nzval .= zero(T)` | `SparseMatrixCSC{T}` | O(nnz) | 0 |
| `v .= zero(T)` | Any array | O(n) | 0 (broadcasting) |

### GPU-Friendly Zeroing

For GPU arrays (CuArray, CuSparseMatrixCSC):

```julia
# CuArray supports fill! natively
fill!(cu_v, zero(T))

# For CuSparseMatrixCSC:
fill!(cu_sparse.nzval, zero(T))
```

### StaticArrays (GPU Ensemble)

For StaticArrays, we return fresh values (no zeroing needed):

```julia
# StaticArrays are stack-allocated, returned fresh
function evaluate_static(u::SVector{N,T}, p, t) where {N,T}
    G = zero(SMatrix{N,N,T})  # Stack allocated
    # ... stamp into G ...
    return G, C, b
end
```

### Recommended Generic Zeroing Function

```julia
"""
Zero out matrix/vector values in-place, type-agnostically.
"""
@inline function zero_values!(x::AbstractVector{T}) where T
    fill!(x, zero(T))
end

@inline function zero_values!(S::SparseMatrixCSC{T}) where T
    fill!(nonzeros(S), zero(T))
end

# For StaticArrays, just return fresh zero
@inline zero_values(::Type{SVector{N,T}}) where {N,T} = zero(SVector{N,T})
@inline zero_values(::Type{SMatrix{M,N,T}}) where {M,N,T} = zero(SMatrix{M,N,T})
```

## Implementation Plan

### Phase 1: Add EvalContext to PrecompiledCircuit

```julia
mutable struct PrecompiledCircuit{F,P,S}
    # ... existing fields ...

    # NEW: Pre-allocated evaluation workspace
    eval_ctx::EvalContext{Float64}
end
```

Modify `compile_circuit` to create `EvalContext`.

### Phase 2: Modify fast_rebuild! to Use EvalContext

```julia
function fast_rebuild!(pc::PrecompiledCircuit, u::Vector{Float64}, t::Real)
    eval_ctx = pc.eval_ctx
    eval_ctx.time = real_time(t)

    # Zero COO values
    zero_values!(eval_ctx.G_V)
    zero_values!(eval_ctx.C_V)
    zero_values!(eval_ctx.b_direct)

    # Evaluate circuit into preallocated storage
    evaluate_values!(eval_ctx, pc.params, u)

    # Update sparse matrices (unchanged)
    update_sparse_from_coo!(pc.G, eval_ctx.G_V, pc.G_coo_to_nz, pc.G_n_coo)
    update_sparse_from_coo!(pc.C, eval_ctx.C_V, pc.C_coo_to_nz, pc.C_n_coo)
end
```

### Phase 3: Value-Only Device Evaluation

Change from context-based stamping to direct-write stamping:

```julia
# OLD: Creates MNAContext, stamps into it
function stamp!(dev::VADevice, ctx::MNAContext, p, n; x, t, spec)
    # ... evaluate ...
    stamp_G!(ctx, p, n, G_val)  # Pushes to ctx.G_I, G_J, G_V
end

# NEW: Writes directly to preallocated slots
@inline function evaluate!(dev::VADevice, eval_ctx::EvalContext,
                           slot_G::Int, slot_b::Int, x, p, n)
    V_pn = x[p] - x[n]
    I = dev.Is * (exp(V_pn / dev.nVt) - 1)
    G = dev.Is / dev.nVt * exp(V_pn / dev.nVt)

    # Write directly to preallocated slots
    eval_ctx.G_V[slot_G] = G      # (p,p)
    eval_ctx.G_V[slot_G+1] = -G   # (p,n)
    eval_ctx.G_V[slot_G+2] = -G   # (n,p)
    eval_ctx.G_V[slot_G+3] = G    # (n,n)

    eval_ctx.b_direct[p] += I - G*V_pn
    eval_ctx.b_direct[n] -= I - G*V_pn
end
```

### Phase 4: Store EvalContext in DAE `p` Parameter

```julia
function SciMLBase.DAEProblem(circuit::MNACircuit, tspan; kwargs...)
    pc = compile_circuit(circuit.builder, circuit.params, circuit.spec)
    eval_ctx = pc.eval_ctx  # Use preallocated context

    function dae_residual!(resid, du, u, p, t)
        # p is eval_ctx
        fast_residual!(resid, du, u, p, real_time(t))
    end

    # Pass eval_ctx as p parameter
    return DAEProblem(f, du0, u0, tspan; p=eval_ctx, kwargs...)
end
```

## Performance Targets

| Metric | Current | After Implementation |
|--------|---------|---------------------|
| Bytes/iteration (simple RC) | ~2000 | 0 |
| Bytes/iteration (VA diode) | ~2500 | 0 |
| Bytes/iteration (BSIM4) | ~5000 | 0 |
| Total allocations (1M iter) | ~2-5 GB | ~0 (setup only) |

## Open Questions

1. **Thread Safety**: For multi-threaded ensemble simulations, each thread needs its own EvalContext. Consider:
   - `EvalContext` pool
   - Thread-local storage
   - Copy-on-spawn semantics

2. **Dynamic Structure**: Some VA devices may have operating-point-dependent structure (e.g., `if Vgs > Vth then ...`). Need worst-case pattern with zeros.

3. **Dual Number Sizes**: For ForwardDiff, Dual sizes must be known at compile time. Consider:
   - Generated functions with explicit sizes
   - Template parameter `N` for number of duals

4. **GPU Integration**: For GPU ensemble, consider:
   - `StaticArrays` for small circuits (stack-allocated)
   - `CuSparseMatrixCSC` for large circuits

## References

- [Julia SparseArrays Documentation](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
- [Optimal in-place sparse modification (Julia Discourse)](https://discourse.julialang.org/t/optimal-way-to-do-in-place-modification-of-sparse-matrix/5170)
- OpenVAF OSDI implementation (direct pointer model)
- `doc/va_zero_allocation_plan.md` - Original plan
