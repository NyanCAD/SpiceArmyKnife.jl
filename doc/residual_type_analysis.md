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

## External Reference Implementations

### VACASK Architecture (C++)

VACASK is an analog circuit simulator that uses OpenVAF's OSDI interface for device models.
Source: https://codeberg.org/arpadbuermen/VACASK

**Key Pattern: Direct Pointer Stamping**

VACASK separates circuit setup into distinct phases:

1. **Structure Discovery** (`populateStructures()`):
   - Determines sparsity pattern
   - Allocates Jacobian entries in sparse matrix
   - Allocates state storage

2. **Pointer Binding** (`bindCore()`):
   - Gets raw pointers to sparse matrix nonzeros
   - Stores pointers in device instance: `jacResistArray[i] = matResist->valuePtr(...)`
   - These pointers remain valid for the entire simulation

3. **Evaluation** (`eval()` + `load()`):
   - `eval()` computes device currents/charges from voltages
   - `load()` writes directly to pre-bound pointers - **zero allocation**

```cpp
// In bindCore() - called once during setup
for(auto i=0; i<numEntries; i++) {
    auto& entry = jacobianEntry(i);
    auto e = nodes_[entry.nodes.node_1]->unknownIndex();
    auto u = nodes_[entry.nodes.node_2]->unknownIndex();

    // Store pointer directly to sparse matrix storage
    jacResistArray[i] = matResist->valuePtr(MatrixEntryPosition(e, u), ...);
}

// In loadCore() - called every Newton iteration
// Writes directly through stored pointers - NO ALLOCATION
descriptor->load_jacobian_resist(instance, model);
```

### OpenVAF OSDI Interface

OpenVAF compiles Verilog-A to native code with the OSDI interface.
Source: https://github.com/pascalkuthe/OpenVAF

**OSDI Descriptor Key Fields:**

```c
typedef struct OsdiDescriptor {
    // Jacobian structure - determined at compile time
    uint32_t num_jacobian_entries;
    OsdiJacobianEntry *jacobian_entries;

    // Offset into instance struct where Jacobian pointers are stored
    uint32_t jacobian_ptr_resist_offset;

    // Core functions - write directly to bound pointers
    void (*load_jacobian_resist)(void *inst, void* model);
    void (*load_jacobian_react)(void *inst, void* model, double alpha);
    void (*load_residual_resist)(void *inst, void* model, double *dst);
} OsdiDescriptor;
```

**Key Insight**: The OSDI interface requires the simulator to:
1. Allocate instance storage (`instance_size` bytes)
2. Bind Jacobian pointers during setup
3. Call `load_*` functions during Newton iteration - these write through the pre-bound pointers

### SciML GPU Requirements

For GPU acceleration with DiffEqGPU.jl, residual functions must meet specific requirements.
Sources:
- https://github.com/SciML/DiffEqGPU.jl
- https://docs.sciml.ai/DiffEqGPU/stable/getting_started/

**GPU Compatibility Requirements:**

1. **Array Types**:
   - Large state vectors: Use `CuArray` (lives on GPU)
   - Small ensemble problems: Use `StaticArrays` (`SVector`, `SMatrix`)
   - Prefer `Float32` for better GPU performance

2. **Non-Allocating Code**:
   - Residual functions must be fully non-allocating
   - Use in-place operations: `mul!(du, A, u)` instead of `du = A * u`
   - For ensemble methods: Use `SVector` for immutable, stack-allocated returns

3. **Broadcast-Compatible**:
   - All operations must work with Julia's broadcasting
   - GPU arrays override broadcast to run on GPU

4. **Function Signature**:
   - Standard: `f!(du, u, p, t)` for in-place
   - Out-of-place: `f(u, p, t) -> SVector{N}` for ensemble GPU

**Example GPU-Compatible Pattern:**
```julia
# GPU-compatible residual (no allocations)
function residual!(du, u, p, t)
    # All operations are in-place or use preallocated buffers
    mul!(du, p.G, u)            # G*u into du
    mul!(du, p.C, p.du_cache, 1.0, 1.0)  # add C*du_cache
    du .-= p.b                   # subtract b
    return nothing
end
```

## Recommended Architecture: OSDI-Inspired Design

Based on VACASK, OpenVAF, and SciML requirements, here's the recommended architecture:

### Phase 1: Compilation (Once)

```julia
struct CompiledCircuit{T}
    # Fixed structure
    n::Int                           # System size
    G::SparseMatrixCSC{T,Int}       # Conductance matrix (structure fixed)
    C::SparseMatrixCSC{T,Int}       # Capacitance matrix (structure fixed)
    b::Vector{T}                     # RHS vector

    # Direct-access stamps (like OSDI Jacobian entries)
    stamps::Vector{CompiledStamp{T}}

    # State storage (like OSDI state arrays)
    states::Vector{T}
    prev_states::Vector{T}
end

struct CompiledStamp{T}
    # Which matrix entry to update
    matrix::Symbol        # :G, :C, or :b
    nz_idx::Int          # Index into nonzeros(matrix)

    # How to compute the value
    value_fn::Function   # (params, spec, x, states) -> T
end
```

### Phase 2: Rebuild (Every Newton Iteration)

```julia
function fast_rebuild!(cc::CompiledCircuit, u, t)
    spec = SimSpec(time=t, ...)

    # Reset matrices (fast, no allocation)
    fill!(nonzeros(cc.G), 0.0)
    fill!(nonzeros(cc.C), 0.0)
    fill!(cc.b, 0.0)

    # Evaluate each stamp directly into matrix storage
    @inbounds for stamp in cc.stamps
        val = stamp.value_fn(params, spec, u, cc.states)
        if stamp.matrix == :G
            nonzeros(cc.G)[stamp.nz_idx] += val
        elseif stamp.matrix == :C
            nonzeros(cc.C)[stamp.nz_idx] += val
        else
            cc.b[stamp.nz_idx] += val
        end
    end
end
```

### Benefits of This Architecture

1. **Zero allocation per iteration**: All storage is preallocated
2. **GPU compatible**: Can use CuArray for matrices, stamps are just indices
3. **OSDI compatible**: Same pattern as VACASK, easy to integrate native VA models
4. **Type stable**: No runtime dispatch, all indices known at compile time

### Migration Path

1. **Phase 1**: Modify `PrecompiledCircuit` to store cached `MNAContext`
   - Pass context to builder instead of creating new one
   - ~70% allocation reduction

2. **Phase 2**: Implement `CompiledStamp` pattern
   - Each stamp becomes (matrix, index, value_fn) tuple
   - ~95% allocation reduction

3. **Phase 3**: GPU support
   - Replace `Vector` with `CuVector`
   - Replace `SparseMatrixCSC` with `CuSparseMatrixCSC`
   - Stamps become GPU kernel operations

## Dual-Mode GPU Architecture

For GPU acceleration, we need to support **two distinct modes** with fundamentally different requirements:

### Mode 1: Within-Method GPU (Large Single Problems)

**Use case**: Single large circuit simulation on GPU (e.g., post-layout extraction with millions of nodes).

**Characteristics**:
- State vector lives on GPU as `CuArray`
- Matrices are `CuSparseMatrixCSC`
- Uses in-place formulation: `f!(du, u, p, t)`
- GPU parallelism comes from matrix operations (SpMV, etc.)

```julia
# Within-method GPU residual
function residual!(du::CuArray{T}, u::CuArray{T}, p::GPUCircuitParams{T}, t) where T
    # In-place sparse matrix-vector multiply on GPU
    mul!(du, p.G, u)                    # du = G*u
    mul!(du, p.C, p.du_prev, 1.0, 1.0)  # du += C*du_prev
    du .-= p.b                           # du -= b
    return nothing
end
```

### Mode 2: Ensemble GPU (Parameter Sweeps / Monte Carlo)

**Use case**: Many small circuit simulations in parallel (e.g., Monte Carlo with 10,000 parameter variations of a 10-node circuit).

**Characteristics**:
- State vector is `SVector{N,T}` (stack-allocated, immutable)
- Matrices are `SMatrix{N,N,T}` (compile-time size)
- Uses out-of-place formulation: `f(u, p, t) -> SVector{N,T}`
- GPU parallelism comes from running many problems simultaneously

```julia
# Ensemble GPU residual (out-of-place, zero allocation)
function residual(u::SVector{N,T}, p::StaticCircuitParams{N,T}, t) where {N,T}
    # Out-of-place but zero allocation (all stack-allocated)
    du = p.G * u + p.C * p.du_prev - p.b
    return du  # Returns SVector{N,T}
end
```

### Unified Architecture Design

To support both modes with a single codebase, we parameterize by array type:

```julia
# Generic circuit type parameterized by array/matrix types
struct CompiledCircuit{
    VecType,      # Vector type: Vector, CuArray, or SVector{N}
    MatType,      # Matrix type: SparseMatrixCSC, CuSparseMatrixCSC, or SMatrix{N,N}
    T             # Element type: Float64 or Float32
}
    n::Int                    # System size (runtime for Vector/CuArray, compile-time for SVector)
    G::MatType                # Conductance matrix
    C::MatType                # Capacitance matrix
    b::VecType                # RHS vector

    # Stamp indices (compile-time known for zero allocation)
    stamp_indices::NTuple{K, StampIndex} where K
end

# StampIndex encodes position without runtime dispatch
struct StampIndex
    target::UInt8    # 0=G, 1=C, 2=b
    idx::Int         # Linear index into nonzeros or vector
end
```

### Compile-Time Specialization

The key insight is that **circuit structure is known at compile time**. We can specialize everything:

```julia
# Type alias for different backends
const CPUCircuit{T} = CompiledCircuit{Vector{T}, SparseMatrixCSC{T,Int}, T}
const GPUCircuit{T} = CompiledCircuit{CuArray{T}, CuSparseMatrixCSC{T,Int}, T}
const StaticCircuit{N,T} = CompiledCircuit{SVector{N,T}, SMatrix{N,N,T}, T}

# Generic rebuild function - specialized at compile time
@inline function rebuild_values!(cc::CompiledCircuit, u, t, params)
    # Reset (works for all array types via broadcast)
    cc.G.nzval .= zero(eltype(cc.G))
    cc.C.nzval .= zero(eltype(cc.C))
    cc.b .= zero(eltype(cc.b))

    # Evaluate stamps - unrolled for StaticCircuit, looped otherwise
    _apply_stamps!(cc, u, t, params)
    return nothing
end

# For StaticCircuit: compile-time unrolled via @generated
@generated function _apply_stamps!(cc::StaticCircuit{N,T}, u, t, params) where {N,T}
    # Generate unrolled stamp applications at compile time
    # Each stamp becomes a direct indexed assignment
    exprs = Expr[]
    for i in 1:num_stamps(cc)
        push!(exprs, quote
            val = evaluate_stamp($i, u, t, params)
            _write_stamp!(cc, $(stamp_index(cc, i)), val)
        end)
    end
    quote
        $(exprs...)
        return nothing
    end
end
```

### Out-of-Place Wrapper for Ensemble GPU

For ensemble GPU, DiffEqGPU.jl requires out-of-place functions. We provide a wrapper:

```julia
# In-place version (CPU, within-method GPU)
function residual!(du, u, p::CompiledCircuit, t)
    rebuild_values!(p, u, t, p.params)
    mul!(du, p.G, u)
    mul!(du, p.C, p.du_prev, one(eltype(du)), one(eltype(du)))
    du .-= p.b
    return nothing
end

# Out-of-place version (ensemble GPU with StaticArrays)
function residual(u::SVector{N,T}, p::StaticCircuit{N,T}, t) where {N,T}
    # Rebuild values (all operations are on static arrays, stack-allocated)
    G, C, b = rebuild_static_values(p, u, t)

    # Compute residual (all stack-allocated)
    du = G * u + C * p.du_prev - b
    return du
end

# Returns static matrices with updated values (no allocation)
@inline function rebuild_static_values(cc::StaticCircuit{N,T}, u, t) where {N,T}
    # For small circuits, we can store G, C as dense SMatrix
    # Values are computed and returned as new static matrices
    G = _build_G_static(cc, u, t)  # Returns SMatrix{N,N,T}
    C = _build_C_static(cc, u, t)  # Returns SMatrix{N,N,T}
    b = _build_b_static(cc, u, t)  # Returns SVector{N,T}
    return G, C, b
end
```

### Parameter Sweep Integration

For Monte Carlo and parameter sweeps:

```julia
# Parameter sweep on GPU ensemble
function ensemble_sweep(
    circuit_fn,          # (params) -> StaticCircuit{N,T}
    param_samples,       # Vector of parameter NamedTuples
    u0::SVector{N,T},
    tspan
) where {N,T}

    # Build problems for each parameter sample
    problems = map(param_samples) do params
        cc = circuit_fn(params)
        ODEProblem(residual, u0, tspan, cc)
    end

    # Create ensemble problem
    ensemble_prob = EnsembleProblem(first(problems),
        prob_func = (prob, i, repeat) -> problems[i])

    # Solve on GPU
    sol = solve(ensemble_prob, GPUTsit5(), EnsembleGPUArray(),
                trajectories=length(param_samples))

    return sol
end
```

### Type Hierarchy Summary

```
CompiledCircuit{VecType, MatType, T}
├── CPUCircuit{T}          # VecType=Vector{T}, MatType=SparseMatrixCSC{T,Int}
│   └── In-place: residual!(du, u, p, t)
│   └── Use: Standard CPU simulation
│
├── GPUCircuit{T}          # VecType=CuArray{T}, MatType=CuSparseMatrixCSC{T,Int}
│   └── In-place: residual!(du, u, p, t)
│   └── Use: Large single-circuit GPU simulation
│
└── StaticCircuit{N,T}     # VecType=SVector{N,T}, MatType=SMatrix{N,N,T}
    └── Out-of-place: residual(u, p, t) -> SVector{N,T}
    └── Use: Ensemble GPU for parameter sweeps, Monte Carlo
```

### Zero-Allocation Verification

To verify zero allocation for both modes:

```julia
# Test in-place mode (CPU/GPU)
cc_cpu = compile_circuit(builder, params, CPUCircuit{Float64})
u = rand(cc_cpu.n)
du = similar(u)
@assert @allocated(residual!(du, u, cc_cpu, 0.0)) == 0

# Test out-of-place mode (ensemble)
cc_static = compile_circuit(builder, params, StaticCircuit{10, Float64})
u_static = @SVector rand(10)
@assert @allocated(residual(u_static, cc_static, 0.0)) == 0
```

### Implementation Phases

**Phase 3a: CPU Zero-Allocation**
- Implement `CompiledStamp` pattern with index-based access
- Achieve zero allocation for `CPUCircuit`
- Target: < 100 bytes per residual call

**Phase 3b: StaticArrays Support**
- Add `StaticCircuit{N,T}` specialization
- Implement `@generated` unrolled stamp application
- Create out-of-place `residual()` wrapper
- Target: 0 bytes per residual call (all stack-allocated)

**Phase 3c: GPU Integration**
- Add CUDA.jl dependency (optional)
- Implement `GPUCircuit{T}` with CuArrays
- Test with DiffEqGPU.jl ensemble problems
- Verify correct GPU kernel generation
