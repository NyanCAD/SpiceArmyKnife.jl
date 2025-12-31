# VACASK Context Allocation Optimization Study

## Current Status (2024-12 Update)

### Completed Optimizations

1. **Explicit Time Passing (Phase 2)**: Builder signature changed from `(params, spec; x)` to `(params, spec, t::Real=0.0; x)`. This eliminates MNASpec allocation (~80 bytes) per call.

2. **SPICE Codegen Updated**: Both top-level and subcircuit builders now use explicit time parameter:
   - Top-level: `function circuit(params, spec, t::Real=0.0; x=...)`
   - Subcircuit: `function subckt_mna_builder(lens, spec, t::Real, ctx, ...)`

3. **Unified Code Path**: `MNASim.tran!` now delegates to `MNACircuit.tran!` internally, ensuring all circuits use the same DAE-based path.

4. **PrecompiledCircuit Updated**: `fast_rebuild!` now passes time explicitly to the builder, eliminating MNASpec allocation during iteration.

### Current Benchmark Results

**VACASK RC Benchmark** (`benchmarks/vacask/rc/cedarsim/runme.jl`):
- **Simulation**: 1 second transient with pulse train, ~100K timepoints
- **Time**: 361-946 ms (median ~660 ms)
- **Memory**: 298 MiB, 5.3M allocations
- **GC overhead**: 36-62% of runtime
- **Comparison**: Previously was ~10+ seconds before explicit time optimization

The high GC overhead (36-62%) indicates that the remaining MNAContext allocations are the primary bottleneck.

### Remaining Allocation Sources

The main remaining allocations come from:

1. **MNAContext creation** (~1.9 KB per builder call):
   - `G_I, G_J, G_V::Vector{Float64}` (COO triplets for G matrix)
   - `C_I, C_J, C_V::Vector{Float64}` (COO triplets for C matrix)
   - `b::Vector{Float64}`, `b_I::Vector{Int}`, `b_V::Vector{Float64}` (RHS)
   - `node_to_idx::Dict{Symbol,Int}`, `node_names::Vector{Symbol}`
   - `current_names::Vector{Symbol}`, `voltage_source_currents::Dict`

2. **DAE solver overhead** (~93 KB per simulation):
   - SciMLBase problem construction
   - Sundials IDA internal allocations
   - Solution storage and interpolation

### Next Steps: EvalWorkspace Implementation

To achieve zero per-iteration allocations, implement the EvalWorkspace pattern from this document:

1. Separate `PrecompiledCircuit` into:
   - `CompiledStructure` (immutable): circuit topology, mappings, sparse matrix structure
   - `EvalWorkspace` (mutable): COO values, time, passed as `p` to DAE solver

2. Add `evaluate_values!` function that fills preallocated COO storage instead of creating new `MNAContext`

3. Store `EvalWorkspace` in DAE problem's `p` parameter

---

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

### EvalContext Design: Separating Mutable and Immutable State

**Key Insight**: Julia optimizes immutable structs better (stack allocation, inlining, constant propagation). We should separate:
- **Immutable**: Circuit structure, mappings, parameters, spec
- **Mutable**: COO values, time, working storage

```julia
"""
Immutable compiled circuit structure.
Contains everything that doesn't change during iteration.
"""
struct CompiledStructure{F,P,S}
    builder::F
    params::P
    spec::S          # Base spec (temp, mode, tolerances - but NOT time)

    # System dimensions (fixed)
    n::Int
    n_nodes::Int
    n_currents::Int

    # Fixed sparsity pattern
    G_coo_to_nz::Vector{Int}
    C_coo_to_nz::Vector{Int}
    G_n_coo::Int
    C_n_coo::Int

    # Sparse matrices (structure fixed, values updated via nzval)
    G::SparseMatrixCSC{Float64,Int}
    C::SparseMatrixCSC{Float64,Int}
end

"""
Mutable evaluation workspace.
Contains only values that change during iteration.
Passed as `p` parameter to DAE solver.
"""
mutable struct EvalWorkspace{T}
    # Reference to immutable structure
    structure::CompiledStructure

    # Preallocated COO value storage (zeroed and refilled each iteration)
    G_V::Vector{T}
    C_V::Vector{T}
    b::Vector{T}

    # Current simulation time (updated each call)
    time::T

    # Working storage for device evaluation
    work::Vector{T}
end

function EvalWorkspace(cs::CompiledStructure)
    EvalWorkspace{Float64}(
        cs,
        zeros(Float64, cs.G_n_coo),
        zeros(Float64, cs.C_n_coo),
        zeros(Float64, cs.n),
        0.0,
        zeros(Float64, 32)
    )
end
```

This design enables:
1. **Compiler optimization**: `CompiledStructure` fields can be constant-propagated
2. **Thread safety**: Each thread gets its own `EvalWorkspace`
3. **Clear semantics**: Immutable parts can be shared, mutable parts are per-evaluation

## Passing Time Explicitly

### Current Pattern (Allocating)

```julia
# Every iteration creates a new MNASpec just to update time
spec_t = MNASpec(temp=pc.spec.temp, mode=:tran, time=real_time(t))
ctx = pc.builder(pc.params, spec_t; x=u)
```

### Recommended Pattern: Explicit Time Parameter

**Since API changes are acceptable**, the cleanest approach is passing time explicitly:

```julia
# OLD builder signature (allocates MNASpec each call):
function build_circuit(params, spec; x=Float64[])
    t = spec.time  # Time buried in spec
end

# NEW builder signature (time is explicit, spec is immutable):
function build_circuit(params, spec, t::Real; x=Float64[])
    # spec contains temp, mode, tolerances (never changes during simulation)
    # t is current simulation time (changes each call, passed directly)
end
```

**Why this is better**:

1. **MNASpec stays immutable** - Better compiler optimization
2. **No allocation** - `t` is passed by value (Float64)
3. **Clear semantics** - Time is obviously the thing that changes each iteration
4. **Thread-safe by design** - No shared mutable state

### Updated MNASpec (Immutable, No Time Field)

```julia
# Time removed from MNASpec - it's passed explicitly
struct MNASpec
    temp::Float64      # Temperature (constant during simulation)
    mode::Symbol       # :dcop, :tran, :ac (constant during simulation)
    gmin::Float64      # Minimum conductance
    tnom::Float64      # Nominal temperature
    abstol::Float64    # Tolerances...
    reltol::Float64
    vntol::Float64
    iabstol::Float64
end
```

### Time Storage in EvalWorkspace

For convenience, time is cached in the mutable workspace:

```julia
mutable struct EvalWorkspace{T}
    const structure::CompiledStructure
    G_V::Vector{T}
    C_V::Vector{T}
    b::Vector{T}
    time::T  # Cached for devices to access
end

function fast_rebuild!(ws::EvalWorkspace, u, t::Real)
    ws.time = t  # Update cached time (single assignment, no allocation)
    # ...
end
```

Devices can access time via `ws.time` or receive it as an explicit parameter.

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

### MaybeInplace.jl for Dual In-Place/Out-of-Place Support

[MaybeInplace.jl](https://github.com/SciML/MaybeInplace.jl) provides the `@bb` macro
that writes code once for both mutable (Vector) and immutable (StaticArrays) types:

```julia
using MaybeInplace

# The @bb macro conditionally mutates or reassigns based on array type
function stamp_conductance!(G, p, n, val)
    @bb @. G[p, p] += val
    @bb @. G[p, n] -= val
    @bb @. G[n, p] -= val
    @bb @. G[n, n] += val
    return G
end
```

The macro expands to:
```julia
if setindex_trait(G) === CanSetindex()
    @. G[p, p] += val   # In-place for Vector/SparseMatrix
else
    G = @. G[p, p] + val  # Reassignment for StaticArrays
end
```

**Benefits for our use case**:
1. Single codebase for CPU (in-place) and GPU ensemble (StaticArrays)
2. No manual dispatch on array type
3. Compiler can optimize both paths

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

# For StaticArrays, just return fresh zero (no mutation)
@inline zero_values(::Type{SVector{N,T}}) where {N,T} = zero(SVector{N,T})
@inline zero_values(::Type{SMatrix{M,N,T}}) where {M,N,T} = zero(SMatrix{M,N,T})
```

### Using MaybeInplace for Zeroing

```julia
using MaybeInplace

# Works for both mutable and immutable arrays
function reset_workspace!(ws)
    @bb ws.G_V .= zero(eltype(ws.G_V))
    @bb ws.C_V .= zero(eltype(ws.C_V))
    @bb ws.b .= zero(eltype(ws.b))
    return ws
end
```

## Implementation Plan

### Phase 1: Refactor PrecompiledCircuit into Immutable/Mutable Split

Replace `mutable struct PrecompiledCircuit` with:

```julia
# Immutable - can be shared, compiler optimizes field access
struct CompiledStructure{F,P,S}
    builder::F
    params::P
    spec::S  # Base spec without time

    # Fixed dimensions
    n::Int
    n_nodes::Int
    n_currents::Int

    # Fixed sparsity pattern and mappings
    G_coo_to_nz::Vector{Int}
    C_coo_to_nz::Vector{Int}
    G_n_coo::Int
    C_n_coo::Int

    # Sparse matrices (structure fixed)
    G::SparseMatrixCSC{Float64,Int}
    C::SparseMatrixCSC{Float64,Int}
end

# Mutable - per-evaluation, passed as p
mutable struct EvalWorkspace{T}
    const structure::CompiledStructure  # const for optimization
    G_V::Vector{T}
    C_V::Vector{T}
    b::Vector{T}
    time::T
end
```

Note: Julia 1.8+ supports `const` fields in mutable structs for optimization.

### Phase 2: New Evaluation API with Explicit Time

```julia
# New builder signature: time passed explicitly, not in spec
function build_circuit(params, spec, t::Real; x=Float64[])
    # t is current simulation time
    # spec contains temp, mode, tolerances (immutable during simulation)
end

# fast_rebuild! becomes:
function fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)
    ws.time = t
    cs = ws.structure

    # Zero COO values
    fill!(ws.G_V, zero(eltype(ws.G_V)))
    fill!(ws.C_V, zero(eltype(ws.C_V)))
    fill!(ws.b, zero(eltype(ws.b)))

    # Evaluate circuit into preallocated storage
    # NEW: Pass time explicitly, not via spec
    evaluate_values!(ws, cs.params, cs.spec, t, u)

    # Update sparse matrices
    update_sparse_from_coo!(cs.G, ws.G_V, cs.G_coo_to_nz, cs.G_n_coo)
    update_sparse_from_coo!(cs.C, ws.C_V, cs.C_coo_to_nz, cs.C_n_coo)
end
```

### Phase 3: Value-Only Device Evaluation with MaybeInplace

```julia
using MaybeInplace

# OLD: Creates MNAContext, stamps into it
function stamp!(dev::VADevice, ctx::MNAContext, p, n; x, t, spec)
    stamp_G!(ctx, p, n, G_val)  # Pushes to ctx.G_I, G_J, G_V
end

# NEW: Writes directly to preallocated slots, supports StaticArrays
@inline function evaluate!(dev::VADevice, G_V, b, slot_G::Int, x, t, p, n)
    V_pn = x[p] - x[n]
    I = dev.Is * (exp(V_pn / dev.nVt) - 1)
    G = dev.Is / dev.nVt * exp(V_pn / dev.nVt)

    # Use @bb for in-place or out-of-place depending on array type
    @bb G_V[slot_G] = G        # (p,p)
    @bb G_V[slot_G+1] = -G     # (p,n)
    @bb G_V[slot_G+2] = -G     # (n,p)
    @bb G_V[slot_G+3] = G      # (n,n)

    @bb b[p] += I - G*V_pn
    @bb b[n] -= I - G*V_pn

    return G_V, b  # Return for StaticArrays case
end
```

### Phase 4: Store EvalWorkspace in DAE `p` Parameter

```julia
function SciMLBase.DAEProblem(circuit::MNACircuit, tspan; kwargs...)
    # Compile to immutable structure
    cs = compile_structure(circuit.builder, circuit.params, circuit.spec)

    # Create mutable workspace (this is what gets passed as p)
    ws = EvalWorkspace(cs)

    function dae_residual!(resid, du, u, p::EvalWorkspace, t)
        fast_residual!(resid, du, u, p, real_time(t))
    end

    # Pass workspace as p parameter
    return DAEProblem(f, du0, u0, tspan; p=ws, kwargs...)
end
```

### Phase 5: Thread-Safe Ensemble Support

```julia
# For parallel parameter sweeps, each thread needs its own workspace
function create_ensemble_workspaces(cs::CompiledStructure, n_threads::Int)
    [EvalWorkspace(cs) for _ in 1:n_threads]
end

# Or use a workspace pool
struct WorkspacePool
    structure::CompiledStructure
    workspaces::Vector{EvalWorkspace}
    available::Channel{EvalWorkspace}
end

function borrow!(pool::WorkspacePool)
    take!(pool.available)
end

function return!(pool::WorkspacePool, ws::EvalWorkspace)
    put!(pool.available, ws)
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

## Design Rationale: Immutable vs Mutable

### Why Separate Immutable and Mutable State?

Julia's compiler can heavily optimize immutable structs:

1. **Stack allocation**: Small immutable structs can live on the stack
2. **Field inlining**: Accessing `cs.n_nodes` compiles to a constant if `cs` is known
3. **Constant propagation**: Immutable fields can be propagated through function calls
4. **No aliasing concerns**: Compiler knows immutable data won't change

Example optimization:
```julia
# With immutable CompiledStructure, this:
for k in 1:cs.G_n_coo
    ws.G_V[k] = ...
end

# Can compile to (if cs.G_n_coo is constant):
for k in 1:1247  # Literal constant
    ws.G_V[k] = ...
end
```

### Why `const` Fields in Mutable Structs?

Julia 1.8+ allows `const` fields in mutable structs:

```julia
mutable struct EvalWorkspace{T}
    const structure::CompiledStructure  # Never reassigned
    G_V::Vector{T}                       # Values change
    time::T                              # Updated each call
end
```

This tells the compiler that `ws.structure` never changes after construction,
enabling the same optimizations as for immutable structs on that field.

### Thread Safety Model

```
CompiledStructure (immutable, shared)
    ↓
    ├─→ EvalWorkspace (thread 1)
    ├─→ EvalWorkspace (thread 2)
    └─→ EvalWorkspace (thread 3)
```

Each thread has its own mutable workspace, but all share the same
immutable structure. No locking needed.

## References

- [Julia SparseArrays Documentation](https://docs.julialang.org/en/v1/stdlib/SparseArrays/)
- [Optimal in-place sparse modification (Julia Discourse)](https://discourse.julialang.org/t/optimal-way-to-do-in-place-modification-of-sparse-matrix/5170)
- [MaybeInplace.jl](https://github.com/SciML/MaybeInplace.jl) - Dual in-place/out-of-place code
- OpenVAF OSDI implementation (direct pointer model)
- `doc/va_zero_allocation_plan.md` - Original plan
