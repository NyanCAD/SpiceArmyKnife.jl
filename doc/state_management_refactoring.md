# State Management Refactoring for Constant Folding and SROA Optimization

## Implementation Status

**COMPLETED** - The following optimizations have been implemented:

1. ✅ **EvalWorkspace is now an immutable struct** - Vectors inside are still mutable, but the struct itself is immutable (no field reassignment). Removed `time` field (passed explicitly) and eliminated `G_V`, `C_V` intermediate arrays.

2. ✅ **Direct stamping from vctx to sparse matrices** - Eliminated the intermediate copy: `vctx.G_V → ws.G_V → sparse` is now `vctx.G_V → sparse`.

3. ✅ **SpecializedWorkspace with SVector-based structure capture** - For small circuits (n_G, n_C ≤ 64), COO-to-CSC mappings are stored as SVectors in a closure, enabling SROA optimization. Achieves **0 bytes per iteration**.

## Executive Summary

This document analyzes the current data structure hierarchy for MNA circuit evaluation and proposes refactoring to achieve the following hierarchy:

1. **CONSTANT** (compiled into code, stack-allocated):
   - Literals in builder function bodies
   - Circuit topology (node count, COO structure sizes)
   - Default parameter values via ParamLens when not overridden
   - SVector PWL times/values

2. **IMMUTABLE** (heap-allocated but never changes after construction):
   - CompiledStructure - fixed after compilation
   - MNASpec - fixed for duration of simulation (except time)
   - Sparse matrix structures (G.colptr, G.rowval - not nzval)
   - COO-to-CSC mappings

3. **MUTABLE** (changes every Newton iteration):
   - COO values (G_V, C_V, b_V)
   - Sparse matrix values (G.nzval, C.nzval)
   - Position counters in ValueOnlyContext
   - Current simulation time

**Key Design Principle**: Mutable data should be **contained inside** immutable data structures, not the reverse. A mutable struct "taints" any immutable data within it.

## Current Architecture Analysis

### Data Structure Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│ CompiledStructure{F,P,S} (immutable struct)                             │
│ Defined in: src/mna/precompile.jl:68                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ CONSTANT (type parameters):                                             │
│   F = typeof(builder)  ──► enables builder function specialization      │
│   P = typeof(params)   ──► NamedTuple, enables ParamLens constant fold  │
│   S = typeof(spec)     ──► MNASpec{Float64}, enables spec const fold    │
│                                                                         │
│ IMMUTABLE (heap but fixed):                                             │
│   builder::F           # Circuit builder function (closure or function) │
│   params::P            # NamedTuple of circuit parameters               │
│   spec::S              # MNASpec (temp, mode, tolerances - NOT time)    │
│   n, n_nodes, n_currents  # System dimensions (Int)                     │
│   node_names::Vector{Symbol}     # For solution interpretation          │
│   current_names::Vector{Symbol}  # For solution interpretation          │
│   G_coo_to_nz::Vector{Int}       # Mapping COO → sparse nzval position  │
│   C_coo_to_nz::Vector{Int}       # Mapping COO → sparse nzval position  │
│   G_n_coo, C_n_coo::Int          # Number of COO entries                │
│   G, C::SparseMatrixCSC          # Sparse matrices (structure fixed)    │
│   b_deferred_resolved::Vector{Int}  # Pre-resolved b stamp indices      │
│   n_b_deferred::Int              # Number of deferred stamps            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ EvalWorkspace{T,CS} (mutable struct)                                    │
│ Defined in: src/mna/precompile.jl:129                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ IMMUTABLE REFERENCE (should use `const` in Julia 1.8+):                 │
│   structure::CS        # Reference to CompiledStructure (never changes) │
│   ctx::MNAContext      # Preallocated fallback context                  │
│   vctx::ValueOnlyContext{T}  # Zero-allocation stamping context         │
│   supports_ctx_reuse::Bool   # Feature flag (fixed at construction)     │
│   supports_value_only_mode::Bool  # Feature flag (fixed)                │
│                                                                         │
│ MUTABLE (changes every iteration):                                      │
│   G_V::Vector{T}       # COO values for G (zeroed and refilled)         │
│   C_V::Vector{T}       # COO values for C                               │
│   b::Vector{T}         # RHS vector                                     │
│   b_deferred_I::Vector{MNAIndex}  # Deferred b stamp indices            │
│   b_deferred_V::Vector{T}         # Deferred b stamp values             │
│   time::T              # Current simulation time                        │
│   resid_tmp::Vector{T} # Working storage                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ValueOnlyContext{T} (mutable struct)                                    │
│ Defined in: src/mna/value_only.jl:49                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ IMMUTABLE AFTER CREATION (but struct is mutable ❌):                    │
│   node_to_idx::Dict{Symbol,Int}  # Reference from original MNAContext   │
│   n_nodes::Int, n_currents::Int  # System dimensions                    │
│   n_G::Int, n_C::Int, n_b_deferred::Int  # Expected sizes               │
│   charge_is_vdep::Vector{Bool}   # Detection cache (fixed after 1st)    │
│                                                                         │
│ MUTABLE (reset every iteration):                                        │
│   G_V::Vector{T}       # Pre-sized G values                             │
│   C_V::Vector{T}       # Pre-sized C values                             │
│   b::Vector{T}         # Pre-sized b vector                             │
│   b_V::Vector{T}       # Pre-sized deferred b values                    │
│   G_pos::Int           # Write position counter                         │
│   C_pos::Int           # Write position counter                         │
│   b_deferred_pos::Int  # Write position counter                         │
│   current_pos::Int     # Counter for alloc_current!                     │
│   charge_pos::Int      # Counter for alloc_charge!                      │
│   charge_detection_pos::Int  # Cache access counter                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ MNASpec{T} (immutable struct)                                           │
│ Defined in: src/mna/solve.jl:54                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ temp::Float64          # Temperature in Celsius                         │
│ mode::Symbol           # :dcop, :tran, :ac                              │
│ time::T                # Simulation time (parameterized for ForwardDiff)│
│ gmin, tnom, abstol, reltol, vntol, iabstol  # Tolerances                │
│                                                                         │
│ NOTE: time is in MNASpec but the fast path passes t explicitly to avoid │
│ allocating a new MNASpec every iteration.                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ ParamLens{NT} (immutable struct)                                        │
│ Defined in: src/spectre.jl:126                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ nt::NT                 # NamedTuple for parameter overrides             │
│                                                                         │
│ KEY OPTIMIZATION: When lens.subckt(; R=default) is called:              │
│   - If override exists in nt: returns override value                    │
│   - If no override: returns default value (CONSTANT FOLDS!)             │
│ This enables parameter values to be compiled into code.                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Current Evaluation Path

```julia
# In fast_rebuild!(ws::EvalWorkspace, u::AbstractVector, t::Real)
if ws.supports_value_only_mode
    vctx = ws.vctx
    reset_value_only!(vctx)
    cs.builder(cs.params, cs.spec, ws.time, u, vctx)  # Positional call
    # Copy values from vctx to ws
    @inbounds @simd for k in 1:n_G
        ws.G_V[k] = vctx.G_V[k]
    end
    # Update sparse matrices
    update_sparse_from_coo!(cs.G, ws.G_V, cs.G_coo_to_nz, n_G)
```

### Current Allocation Profile

From `doc/va_zero_allocation_plan.md`:

| Phase | Bytes/iteration | Reduction |
|-------|-----------------|-----------|
| Original | 6,314 | baseline |
| MNAContext reuse | 1,838 | 71% |
| ValueOnlyContext | 472 | 93% |
| SVector PWL (current) | 40 | 99.4% |

The remaining ~40 bytes comes from:
- Float64 boxing when calling builder through type parameter (~16 bytes)
- SVector type specialization overhead (~24 bytes)

## Problems Identified

### 1. ValueOnlyContext Mixes Read-Only and Mutable State

The fundamental issue is that `ValueOnlyContext` is a **mutable struct** that contains fields that never change after creation:

```julia
mutable struct ValueOnlyContext{T}
    # These NEVER change after create_value_only_context():
    node_to_idx::Dict{Symbol,Int}  # Fixed reference
    n_nodes::Int                   # Fixed dimension
    n_currents::Int                # Fixed dimension
    n_G::Int                       # Fixed count
    n_C::Int                       # Fixed count
    n_b_deferred::Int              # Fixed count
    charge_is_vdep::Vector{Bool}   # Fixed cache

    # These are truly mutable (reset every iteration):
    G_pos::Int
    C_pos::Int
    ...
end
```

**Impact**: Because the struct is mutable, the compiler cannot constant-propagate `n_G`, `n_C`, etc. into loops:

```julia
# In fast_rebuild!:
@inbounds @simd for k in 1:n_G        # n_G = cs.G_n_coo, could be constant
    ws.G_V[k] = vctx.G_V[k]
end
```

### 2. Double Indirection in Value Copying

The current flow copies values twice:

```
Builder stamps → vctx.G_V → ws.G_V → cs.G.nzval
                    ↑           ↑         ↑
                 Copy 1      Copy 2    Copy 3
```

This is in `fast_rebuild!` (precompile.jl:891-939):
```julia
# Step 1: Builder stamps to vctx.G_V
cs.builder(cs.params, cs.spec, ws.time, u, vctx)

# Step 2: Copy from vctx to ws
@inbounds @simd for k in 1:n_G
    ws.G_V[k] = vctx.G_V[k]          # WHY? Both are in workspace!
end

# Step 3: Copy from ws to sparse
update_sparse_from_coo!(cs.G, ws.G_V, cs.G_coo_to_nz, n_G)
```

**Impact**: 3× memory bandwidth when 1× would suffice.

### 3. EvalWorkspace Duplicates ValueOnlyContext Storage

Both `EvalWorkspace` and `ValueOnlyContext` have `G_V`, `C_V`, `b` arrays:

```julia
# EvalWorkspace has:
G_V::Vector{T}      # Size: G_n_coo
C_V::Vector{T}      # Size: C_n_coo
b::Vector{T}        # Size: n

# ValueOnlyContext ALSO has:
G_V::Vector{T}      # Size: n_G (same as G_n_coo)
C_V::Vector{T}      # Size: n_C (same as C_n_coo)
b::Vector{T}        # Size: n_nodes
b_V::Vector{T}      # Size: n_b_deferred
```

**Impact**: Wasted memory and unnecessary copies between identical arrays.

### 4. Sparse Matrix Values Not Written Directly

The COO-to-CSC mapping is precomputed and stored in `CompiledStructure`:
```julia
G_coo_to_nz::Vector{Int}   # Maps COO position k → G.nzval[idx]
```

But we still go through intermediate `ws.G_V` instead of writing directly to `G.nzval`:
```julia
# Current:
vctx.G_V[pos] = val          # ValueOnlyContext stamp
ws.G_V[k] = vctx.G_V[k]      # Copy to workspace
cs.G.nzval[nz_idx] += ws.G_V[k]  # Update sparse

# Could be:
cs.G.nzval[nz_idx] += val    # Direct stamp to sparse!
```

## Proposed Refactoring

### Overview: Target Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONSTANT (Stack/Registers)                      │
├─────────────────────────────────────────────────────────────────────────┤
│ • Literals in generated builder code                                    │
│ • ParamLens defaults when no override                                   │
│ • SVector PWL times/values                                              │
│ • Loop bounds (n_G, n_C, n) when specialized                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         IMMUTABLE (Heap, Fixed)                         │
├─────────────────────────────────────────────────────────────────────────┤
│ CompiledCircuit (new, replaces CompiledStructure + EvalWorkspace):      │
│   const builder::F                                                      │
│   const params::P                                                       │
│   const spec::S                                                         │
│   const n_G::Int, n_C::Int, n::Int                                     │
│   const G_nz_mapping::Vector{Int}   # COO pos → G.nzval position        │
│   const C_nz_mapping::Vector{Int}   # COO pos → C.nzval position        │
│   const b_resolved::Vector{Int}     # Deferred b stamp → b position     │
│   const G::SparseMatrixCSC          # Structure fixed, nzval mutable    │
│   const C::SparseMatrixCSC          # Structure fixed, nzval mutable    │
│   const node_to_idx::Dict           # For stamp lookups                 │
│   state::StampState                 # Mutable counters (Box{T})         │
│   b::Vector{Float64}                # Mutable b vector                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MUTABLE (Per-Iteration)                         │
├─────────────────────────────────────────────────────────────────────────┤
│ StampState (mutable struct, small):                                     │
│   G_pos::Int           # Current position in G stamping                 │
│   C_pos::Int           # Current position in C stamping                 │
│   b_deferred_pos::Int  # Current position in deferred b stamps          │
│   current_pos::Int     # Counter for alloc_current!                     │
│   charge_pos::Int      # Counter for alloc_charge!                      │
│   time::Float64        # Current simulation time                        │
│                                                                         │
│ Sparse matrix values (mutated in-place):                                │
│   G.nzval::Vector{Float64}                                              │
│   C.nzval::Vector{Float64}                                              │
│   b::Vector{Float64}                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Direct Sparse Stamping

**Goal**: Eliminate intermediate COO value arrays by stamping directly to sparse matrix nzval.

**Current flow** (3 copies):
```
stamp_G!(vctx, i, j, val) → vctx.G_V[pos]        # Write 1
Copy: ws.G_V[k] = vctx.G_V[k]                     # Write 2
update_sparse: G.nzval[nz_idx] += ws.G_V[k]      # Write 3
```

**New flow** (1 write, direct to sparse):
```
stamp_G!(ctx, i, j, val) → G.nzval[nz_idx] += val   # Direct!
```

**Implementation:**
```julia
"""
DirectStampContext provides zero-copy stamping to sparse matrices.

Stamps are resolved at compile time to nzval positions, then written directly.
No intermediate COO arrays, no copying between arrays.
"""
struct DirectStampContext
    # Immutable after construction:
    node_to_idx::Dict{Symbol,Int}
    G_nz_mapping::Vector{Int}      # pos → nzval index (from CompiledStructure)
    C_nz_mapping::Vector{Int}
    b_resolved::Vector{Int}        # Deferred b stamp positions

    # References to sparse matrix storage (mutated in-place):
    G_nzval::Vector{Float64}       # = nonzeros(G)
    C_nzval::Vector{Float64}       # = nonzeros(C)
    b::Vector{Float64}

    # Mutable position counters (wrapped to allow mutation in immutable struct):
    state::Base.RefValue{StampState}
end

mutable struct StampState
    G_pos::Int
    C_pos::Int
    b_deferred_pos::Int
    current_pos::Int
    charge_pos::Int
end

@inline function stamp_G!(ctx::DirectStampContext, i, j, val)
    iszero(i) && return nothing
    iszero(j) && return nothing
    s = ctx.state[]
    nz_idx = @inbounds ctx.G_nz_mapping[s.G_pos]
    if nz_idx > 0
        @inbounds ctx.G_nzval[nz_idx] += extract_value(val)
    end
    s.G_pos += 1
    return nothing
end
```

### Phase 2: Merge EvalWorkspace and ValueOnlyContext

**Goal**: Eliminate duplicate storage and simplify the hierarchy.

**Current**: Two separate structs with overlapping arrays:
- `EvalWorkspace.G_V` and `ValueOnlyContext.G_V` (same size, copied between)
- `EvalWorkspace.b` and `ValueOnlyContext.b` (same purpose, copied between)

**New**: Single struct that references sparse matrix storage directly:
```julia
"""
EvalContext replaces both EvalWorkspace and ValueOnlyContext.

All stamping goes directly to sparse matrix nzval arrays.
No intermediate COO value storage.
"""
struct EvalContext{CS<:CompiledCircuit}
    # Immutable reference to compiled structure:
    circuit::CS

    # Mutable state (counters):
    state::Base.RefValue{StampState}
end

function reset!(ctx::EvalContext)
    s = ctx.state[]
    s.G_pos = 1
    s.C_pos = 1
    s.b_deferred_pos = 1
    s.current_pos = 1
    s.charge_pos = 1

    # Zero sparse matrix values
    fill!(nonzeros(ctx.circuit.G), 0.0)
    fill!(nonzeros(ctx.circuit.C), 0.0)
    fill!(ctx.circuit.b, 0.0)
    return nothing
end
```

### Phase 3: Use `const` Fields in EvalWorkspace (Julia 1.8+)

If keeping the current structure, mark fields that never change:

```julia
mutable struct EvalWorkspace{T,CS<:CompiledStructure}
    const structure::CS                    # Never changes
    const vctx::ValueOnlyContext{T}        # Never changes
    const ctx::MNAContext                  # Never changes
    const supports_ctx_reuse::Bool         # Never changes
    const supports_value_only_mode::Bool   # Never changes

    # Truly mutable:
    G_V::Vector{T}
    C_V::Vector{T}
    b::Vector{T}
    time::T
    ...
end
```

**Benefits**: Compiler can propagate constants through `ws.structure.n_G` etc.

### Phase 4: Make StampState Stack-Allocatable

If counters are passed as a mutable struct, consider making it small enough for stack allocation:

```julia
# Current approach uses Ref to box mutable state:
state::Base.RefValue{StampState}

# Alternative: Pass state explicitly through the builder call
function fast_rebuild!(circuit::CompiledCircuit, u::Vector{Float64}, t::Float64)
    state = StampState(1, 1, 1, 1, 1, t)  # Stack-allocated

    fill!(nonzeros(circuit.G), 0.0)
    fill!(nonzeros(circuit.C), 0.0)
    fill!(circuit.b, 0.0)

    # Builder modifies state and writes to G/C/b directly
    circuit.builder(circuit.params, circuit.spec, state, u, circuit)

    return nothing
end
```

### Phase 5: ParamLens Constant Propagation Verification

Ensure ParamLens enables constant folding when defaults are not overridden:

```julia
# In generated builder:
function circuit_builder(params, spec::MNASpec, t::Float64, ...)
    lens = ParamLens(params)

    # When params = (;) (empty), lens.R1(; R=1000.0) returns 1000.0
    # This should constant-fold if ParamLens is properly specialized
    p = lens.subcircuit(; R1=1000.0, R2=1000.0)

    # p.R1 should be compile-time constant 1000.0
    stamp_conductance!(ctx, p1, p2, 1.0 / p.R1)
end
```

Use `@code_typed` and Cthulhu.jl to verify constant propagation:
```julia
using Cthulhu
@descend fast_rebuild!(circuit, u, 0.0)
# Check that p.R1 appears as Float64 constant, not field access
```

## Verification Strategy

### 1. Per-Iteration Allocation Measurement

```julia
using BenchmarkTools

# Setup
circuit = MNACircuit(builder; R=1000.0, C=1e-6)
prob = DAEProblem(circuit, (0.0, 1e-3))
ws = prob.p  # EvalWorkspace is passed as p parameter
u = zeros(ws.structure.n)

# Measure allocations in fast_rebuild!
@allocated fast_rebuild!(ws, u, 0.0)  # Should be 0 after optimization

# Warm up and measure
for _ in 1:100
    fast_rebuild!(ws, u, 0.0)
end
allocs = @allocated begin
    for _ in 1:1000
        fast_rebuild!(ws, u, 0.0)
    end
end
@assert allocs == 0 "Expected 0 allocations, got $allocs bytes"
```

### 2. Cthulhu Code Inspection

```julia
using Cthulhu

# Verify no allocations in hot path
@descend fast_rebuild!(ws, u, 0.0)

# Check for:
# - No Core.Box (closure boxing)
# - No gc_pool_alloc in LLVM
# - Constants appear as ::Float64 literals, not field accesses
# - stamp! calls are inlined

# Verify ParamLens constant folding
@descend circuit.builder(circuit.params, circuit.spec, 0.0, ZERO_VECTOR, ws.vctx)
# Check that lens.subcircuit(; R=1000.0) returns compile-time constant
```

### 3. LLVM IR Analysis

```julia
# Check generated LLVM for allocation calls
@code_llvm optimize=true debuginfo=:none fast_rebuild!(ws, u, 0.0)

# Search for:
# - "gc_pool_alloc" → indicates heap allocation
# - "jl_apply_generic" → indicates dynamic dispatch
# - "jl_box_float64" → indicates boxing
```

### 4. Run VACASK Benchmarks

```bash
# Install Julia if needed
curl -fsSL https://install.julialang.org | sh -s -- -y
. ~/.bashrc
~/.juliaup/bin/juliaup default 1.11

# Run RC circuit benchmark
~/.juliaup/bin/julia --project=. benchmarks/vacask/rc/cedarsim/runme.jl

# Run all benchmarks
~/.juliaup/bin/julia --project=. benchmarks/vacask/run_benchmarks.jl
```

Key metrics to compare before/after:
- **Median time** per iteration
- **Memory allocation** total
- **Allocations count** (should be 0 in hot path)
- **GC time** percentage

### 5. Examine Generated Builder Code

```julia
# Generate and inspect SPICE builder code
circuit_code = parse_spice_to_mna("""
V1 vcc 0 5
R1 vcc out 1k
C1 out 0 1u
"""; circuit_name=:test_circuit)

# Print generated code
println(circuit_code)

# Check that:
# - ParamLens is used for parameter access
# - Positional builder signature is generated
# - stamp! calls use inline annotation
```

## Implementation Roadmap

| Phase | Description | Files to Modify | Effort | Impact |
|-------|-------------|-----------------|--------|--------|
| **1** | Add `const` fields to EvalWorkspace | `src/mna/precompile.jl` | Low | Low - Enables const-prop |
| **2** | Merge EvalWorkspace.G_V with vctx.G_V | `src/mna/precompile.jl`, `value_only.jl` | Medium | Medium - Saves memory |
| **3** | Direct sparse stamping | `src/mna/value_only.jl`, `precompile.jl` | High | High - Eliminates 2 copies |
| **4** | Split ValueOnlyContext config/state | `src/mna/value_only.jl` | Medium | Medium - Cleaner design |
| **5** | Verify ParamLens constant folding | `test/mna/param_lens.jl` | Low | Verification only |

### Phase 1: Quick Win - Add `const` Fields (Julia 1.8+)

```julia
# In src/mna/precompile.jl:129
mutable struct EvalWorkspace{T,CS<:CompiledStructure}
    const structure::CS              # NEW: const
    const ctx::MNAContext            # NEW: const
    const vctx::ValueOnlyContext{T}  # NEW: const
    const supports_ctx_reuse::Bool   # NEW: const
    const supports_value_only_mode::Bool  # NEW: const
    # Mutable fields unchanged:
    G_V::Vector{T}
    ...
end
```

### Phase 2: Eliminate Duplicate Storage

Remove `ws.G_V`, `ws.C_V`, `ws.b` - use `vctx.G_V`, `vctx.C_V`, `vctx.b` directly:

```julia
# In fast_rebuild!, instead of copying:
#   ws.G_V[k] = vctx.G_V[k]
#   update_sparse_from_coo!(cs.G, ws.G_V, ...)
# Do:
#   update_sparse_from_coo!(cs.G, vctx.G_V, ...)
```

### Phase 3: Direct Sparse Stamping

Add new context type `DirectStampContext` that stamps directly to `G.nzval`:

```julia
# New file: src/mna/direct_stamp.jl
struct DirectStampContext
    G_nzval::Vector{Float64}
    C_nzval::Vector{Float64}
    G_nz_mapping::Vector{Int}
    C_nz_mapping::Vector{Int}
    state::Ref{StampState}
    ...
end

# stamp_G! writes directly to sparse matrix
@inline function stamp_G!(ctx::DirectStampContext, i, j, val)
    ...
    @inbounds ctx.G_nzval[nz_idx] += val
end
```

## Expected Outcomes

After Phase 1-2 (quick wins):
- Cleaner code with explicit const/mutable separation
- ~20% memory reduction from eliminating duplicate arrays
- No behavior change, same API

After Phase 3 (direct stamping):
- **0 bytes** per Newton iteration (target)
- ~30% improvement in iteration throughput (fewer memory copies)
- Better cache locality (single write location)

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API breakage | Low | High | Keep external API stable, only refactor internals |
| Compile time increase | Medium | Low | Use `@noinline` for cold paths, benchmark compile times |
| Type instability | Medium | High | Test with `@code_warntype`, add type assertions |
| Performance regression | Low | High | Benchmark each phase, revert if regression |

## Appendix: Current Allocation Profile

From `doc/va_zero_allocation_plan.md`:

| Optimization Stage | Bytes/iteration | Reduction |
|--------------------|-----------------|-----------|
| Original (push! + sparse rebuild) | 6,314 | baseline |
| MNAContext reuse (empty! not push!) | 1,838 | 71% |
| ValueOnlyContext (pre-sized arrays) | 472 | 93% |
| SVector PWL (stack-allocated) | 40 | 99.4% |
| **Target: Direct stamping** | **0** | **100%** |

The remaining 40 bytes comes from Float64 boxing when calling the builder function through a type parameter. This may be unavoidable without generated functions.

## References

- `src/mna/precompile.jl` - CompiledStructure, EvalWorkspace definitions
- `src/mna/value_only.jl` - ValueOnlyContext implementation
- `src/mna/context.jl` - MNAContext and stamping primitives
- `src/mna/solve.jl` - MNASpec definition
- `src/spectre.jl` - ParamLens implementation
- `doc/mna_architecture.md` - Overall MNA design
- `benchmarks/vacask/` - Performance benchmarks
