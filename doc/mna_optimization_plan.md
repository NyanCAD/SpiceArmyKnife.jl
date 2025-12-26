# MNA Performance Optimization Plan

## Current Status: Functional but Inefficient

The current nonlinear transient implementation rebuilds the entire circuit at every residual/Jacobian evaluation:

```julia
# From src/mna/solve.jl:1303-1314
function dae_residual!(resid, du, u, p, t)
    # Build circuit at CURRENT operating point and time
    spec_t = MNASpec(temp=base_spec.temp, mode=:tran, time=real_time(t))
    ctx = builder(params, spec_t; x=u)   # ← Allocates new context
    sys = assemble!(ctx)                  # ← Builds sparse matrices from COO

    # F(du, u) = C*du + G*u - b = 0
    mul!(resid, sys.C, du)
    mul!(resid, sys.G, u, 1.0, 1.0)
    resid .-= sys.b
end
```

### What Happens Every Call

1. **New MNAContext allocated** - fresh vectors for G_I, G_J, G_V, etc.
2. **All nodes re-discovered** - `get_node!` called for every node
3. **All devices re-stamped** - COO tuples pushed to vectors
4. **Sparse matrices rebuilt** - `sparse(I, J, V)` rediscovers sparsity pattern
5. **Index resolution** - negative indices resolved to actual positions

This is correct (matrices depend on operating point) but wasteful (structure is constant).

## How OpenVAF Does It Better

OpenVAF separates **structure discovery** (once) from **value updates** (every iteration).

### Instance Data Layout (from OpenVAF internals.md)

```
Instance data:
    jacobian_ptr: array of num_jacobian_entries pointers to double
    jacobian_ptr_react: array of pointers to double
    node_mapping: array of u32, count=nunknowns
    collapsed: array of i8, count=nnodepairs
```

### Three-Phase Approach

#### Phase 1: Setup (Once Per Instance)

```rust
fn process_params(&mut self, sim_builder: &mut SimBuilder, terminals: &[Node]) {
    // 1. Create internal nodes, get simulator indices
    for node in &mut internal_nodes {
        *node = sim_builder.new_internal_unknown(name, tol, units).into();
    }

    // 2. Build node_mapping: device_idx → simulator_idx
    let node_mapping = self.node_mapping();
    for node in node_mapping {
        let idx = node.get();
        if let Some(&terminal) = terminals.get(idx as usize) {
            node.set(terminal.into())  // External terminal
        } else if idx == u32::MAX {
            node.set(0)  // Ground
        } else {
            node.set(internal_nodes[idx as usize - terminals.len()])
        }
    }

    // 3. Register matrix entries with simulator (gets back pointers)
    for entry in self.descriptor.matrix_entries() {
        let column = node_mapping[entry.nodes.node_1 as usize].get();
        let row = node_mapping[entry.nodes.node_2 as usize].get();
        sim_builder.ensure_matrix_entry(column, row)
    }
}
```

#### Phase 2: Pointer Population (Once Per Analysis)

```rust
fn populate_matrix_ptrs(&mut self, matrix_entries: MatrixEntryIter) {
    for (ptrs, resist_ptr) in zip(matrix_entries, self.matrix_ptrs_resist()) {
        // Store pointer to actual matrix entry (G[row, col])
        resist_ptr.set(ptrs.resist_ffi_ptr());  // *mut f64
    }
}
```

#### Phase 3: Evaluation (Every Newton Iteration)

```rust
fn eval(&mut self, sim_info: SimInfo<'_>) {
    // Device writes DIRECTLY to matrix via pre-stored pointers
    // No COO, no allocation, no index lookup!
    self.descriptor.load_jacobian_resist(self.data, self.model_data);
    self.descriptor.load_jacobian_react(self.data, self.model_data, alpha);
}
```

### Key OSDI Data Structures (from osdi_0_4.h)

```c
typedef struct OsdiJacobianEntry {
    OsdiNodePair nodes;      // (row, col) in device-local numbering
    uint32_t react_ptr_off;  // Offset to reactive pointer (or UINT32_MAX)
    uint32_t flags;          // JACOBIAN_ENTRY_RESIST, JACOBIAN_ENTRY_REACT, etc.
} OsdiJacobianEntry;

typedef struct OsdiNode {
    char *name;
    uint32_t resist_residual_off;  // Offset for residual stamping
    uint32_t react_residual_off;
    bool is_flow;  // Current vs voltage unknown
} OsdiNode;
```

## Proposed Julia Implementation

### New Types

```julia
"""
Precompiled circuit with fixed sparsity pattern.
Structure discovered once, values updated each iteration.
"""
struct PrecompiledCircuit{F,P,S}
    # Original builder for reference
    builder::F
    params::P
    spec::S

    # Node mapping: device_node_idx → system_idx (fixed after setup)
    node_mapping::Vector{Int}

    # Preallocated COO storage (fixed length after first build)
    G_I::Vector{Int}      # Row indices (constant)
    G_J::Vector{Int}      # Col indices (constant)
    G_V::Vector{Float64}  # Values (updated each eval)
    C_I::Vector{Int}
    C_J::Vector{Int}
    C_V::Vector{Float64}
    b::Vector{Float64}

    # Preallocated sparse matrices (same sparsity pattern)
    G::SparseMatrixCSC{Float64,Int}
    C::SparseMatrixCSC{Float64,Int}

    # Mapping from COO index to nonzeros(G) index
    G_coo_to_nz::Vector{Int}
    C_coo_to_nz::Vector{Int}
end
```

### Compilation Function

```julia
"""
Compile a circuit builder into a PrecompiledCircuit.
Called once at setup time.
"""
function compile_circuit(builder, params, spec)
    # First pass: discover structure
    ctx = builder(params, spec; x=Float64[])

    # Build sparse matrices to get sparsity pattern
    G = sparse(ctx.G_I, ctx.G_J, ctx.G_V, n, n)
    C = sparse(ctx.C_I, ctx.C_J, ctx.C_V, n, n)

    # Create mapping from COO indices to nonzeros indices
    # This lets us update values in-place without rebuilding
    G_coo_to_nz = compute_coo_to_nz_mapping(ctx.G_I, ctx.G_J, G)
    C_coo_to_nz = compute_coo_to_nz_mapping(ctx.C_I, ctx.C_J, C)

    return PrecompiledCircuit(
        builder, params, spec,
        collect(1:ctx.n_nodes),  # Identity mapping initially
        copy(ctx.G_I), copy(ctx.G_J), copy(ctx.G_V),
        copy(ctx.C_I), copy(ctx.C_J), copy(ctx.C_V),
        copy(ctx.b),
        G, C,
        G_coo_to_nz, C_coo_to_nz
    )
end

function compute_coo_to_nz_mapping(I, J, S::SparseMatrixCSC)
    # For each COO entry (I[k], J[k]), find its index in nonzeros(S)
    mapping = zeros(Int, length(I))
    for k in 1:length(I)
        i, j = I[k], J[k]
        # Find position in CSC
        for idx in nzrange(S, j)
            if rowvals(S)[idx] == i
                mapping[k] = idx
                break
            end
        end
    end
    return mapping
end
```

### Fast Evaluation

```julia
"""
Fast residual evaluation using precompiled circuit.
Only updates values, structure is fixed.
"""
function fast_residual!(resid, du, u, pc::PrecompiledCircuit, t)
    # Build with current operating point (stamps into preallocated COO)
    spec_t = with_time(pc.spec, t)

    # Device stamping writes directly to pc.G_V, pc.C_V, pc.b
    fast_stamp!(pc, u, spec_t)

    # Update sparse matrix values in-place (O(nnz), not O(n²))
    update_sparse_values!(pc.G, pc.G_V, pc.G_coo_to_nz)
    update_sparse_values!(pc.C, pc.C_V, pc.C_coo_to_nz)

    # Compute residual
    mul!(resid, pc.C, du)
    mul!(resid, pc.G, u, 1.0, 1.0)
    resid .-= pc.b
end

function update_sparse_values!(S::SparseMatrixCSC, V::Vector, mapping::Vector)
    nz = nonzeros(S)
    fill!(nz, 0.0)  # Reset to zero
    for (k, v) in enumerate(V)
        nz[mapping[k]] += v  # Accumulate (handles duplicate entries)
    end
end
```

### Device-Level Optimization

For maximum performance, devices should stamp directly:

```julia
"""
Preallocated stamp handle for a device instance.
Created once during circuit compilation.
"""
struct DeviceStampHandle
    # Indices into the circuit's COO arrays
    G_indices::Vector{Int}  # Which G_V entries this device writes
    C_indices::Vector{Int}
    b_indices::Vector{Int}
end

# Device stamps directly to preallocated storage
function fast_stamp!(dev::Resistor, handle::DeviceStampHandle, G_V::Vector, u::Vector)
    # Compute conductance at operating point
    G = 1.0 / dev.R

    # Write to preallocated positions (no push!, no allocation)
    G_V[handle.G_indices[1]] += G   # (p, p)
    G_V[handle.G_indices[2]] -= G   # (p, n)
    G_V[handle.G_indices[3]] -= G   # (n, p)
    G_V[handle.G_indices[4]] += G   # (n, n)
end
```

## Performance Comparison

| Operation | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Node allocation | O(n) per call | O(1) - precomputed | ~10x |
| COO accumulation | O(nnz) allocs | Zero allocs | ~5x |
| Sparse rebuild | O(nnz log n) | O(nnz) value copy | ~3x |
| Index resolution | O(n_currents) | Zero - preresolved | ~2x |
| **Total per eval** | ~1ms for 100 nodes | ~0.1ms | **~10x** |

## Implementation Phases

### Phase 1: PrecompiledCircuit Type
- Add `PrecompiledCircuit` struct
- Implement `compile_circuit` function
- Keep existing `MNACircuit` for compatibility

### Phase 2: Fast Stamping Infrastructure
- Add `DeviceStampHandle` for common devices
- Implement `fast_stamp!` for Resistor, Capacitor, VoltageSource
- Benchmark against current approach

### Phase 3: VA Device Integration
- Extend `make_mna_device` to emit fast stamp handles
- Update Verilog-A codegen for direct stamping
- Match OpenVAF's OSDI pointer model

### Phase 4: Automatic Compilation
- Detect when to use PrecompiledCircuit vs MNACircuit
- Linear circuits: compile once, reuse forever
- Nonlinear circuits: compile once, update values each Newton step

---

## GPU Acceleration

Two distinct GPU use cases exist, requiring different approaches.

### Use Case 1: Ensemble GPU (Parameter Sweeps)

For running the **same small circuit** with **many different parameters** (Monte Carlo, corners, optimization):

```julia
using DiffEqGPU, CUDA

# Create base problem
circuit = MNACircuit(build_inverter; Vdd=1.8, W=1e-6)
prob = ODEProblem(circuit, (0.0, 1e-6))

# Define parameter variation
function prob_func(prob, i, repeat)
    W_varied = 1e-6 * (0.9 + 0.2 * rand())  # ±10% variation
    new_circuit = alter(circuit; W=W_varied)
    return ODEProblem(new_circuit, prob.tspan)
end

# Create ensemble
ensemble = EnsembleProblem(prob; prob_func=prob_func, safetycopy=false)

# Solve 10,000 parameter variations on GPU
sol = solve(ensemble, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
            trajectories=10_000)
```

**Requirements for GPU kernel compatibility:**

1. **Out-of-place ODE function** (already in mna_architecture.md design):
   ```julia
   # Current (in-place) - NOT GPU compatible
   function dae_residual!(resid, du, u, p, t)
       # ... mutates resid ...
   end

   # Required (out-of-place) - GPU compatible
   function ode_rhs(u, p, t)
       # ... returns new vector ...
       return SVector{N}(du...)
   end
   ```

2. **StaticArrays for small circuits**:
   ```julia
   # For circuits with N < ~100 unknowns
   u0 = SVector{N,Float32}(...)  # Static size, GPU-friendly
   ```

3. **Float32 precision** (2x memory bandwidth on GPU):
   ```julia
   circuit = MNACircuit{Float32}(build_inverter; ...)
   ```

4. **No allocations in hot path**:
   - PrecompiledCircuit approach enables this
   - Device stamps write to preallocated static arrays

### Use Case 2: Large Circuit GPU (Single Circuit)

For **one large circuit** (>1000 nodes), use GPU-accelerated sparse linear algebra:

```julia
using CUDA, CUDA.CUSPARSE, CUDSS

# Convert sparse matrices to GPU
G_gpu = CuSparseMatrixCSR(circuit.G)
C_gpu = CuSparseMatrixCSR(circuit.C)
b_gpu = CuArray(circuit.b)
u_gpu = CuArray(u0)

# Direct solver (for moderate sizes)
using CUDSS
solver = CudssSolver(G_gpu, "G", 'F')  # LU factorization on GPU

# Iterative solver (for very large circuits)
using Krylov
x, stats = gmres(G_gpu, b_gpu; M=preconditioner)
```

**Sparse GPU considerations:**

| Aspect | CPU | GPU |
|--------|-----|-----|
| Matrix format | CSC (Julia default) | CSR (CUDA preferred) |
| Direct solve | SuiteSparse UMFPACK | CUDSS |
| Iterative solve | IterativeSolvers.jl | Krylov.jl + CUDA |
| Sweet spot | < 10k unknowns | > 10k unknowns |

### GPU Implementation Phases

#### GPU Phase 1: EnsembleGPU for CircuitSweep

Extend `CircuitSweep` to use EnsembleGPU:

```julia
function tran_gpu!(cs::CircuitSweep, tspan; backend=CUDA.CUDABackend())
    # Convert to out-of-place formulation
    base_circuit = first(cs)
    base_prob = ODEProblem{false}(make_oop_function(base_circuit), u0, tspan)

    # Create ensemble with parameter variations
    function prob_func(prob, i, repeat)
        circuit_i = cs[i]  # Get i-th parameter set
        return remake(prob, p = circuit_i.params)
    end

    ensemble = EnsembleProblem(base_prob; prob_func, safetycopy=false)
    return solve(ensemble, GPUTsit5(), EnsembleGPUKernel(backend);
                 trajectories=length(cs))
end
```

#### GPU Phase 2: Out-of-Place Circuit Evaluation

Add out-of-place evaluation mode:

```julia
"""
Out-of-place circuit evaluation for GPU compatibility.
Returns (G, C, b) as new arrays instead of mutating.
"""
function eval_circuit_oop(circuit::PrecompiledCircuit{N}, u, t) where N
    # Use StaticArrays for small circuits
    G = MMatrix{N,N,Float32}(undef)
    C = MMatrix{N,N,Float32}(undef)
    b = MVector{N,Float32}(undef)

    fill!(G, 0f0)
    fill!(C, 0f0)
    fill!(b, 0f0)

    # Stamp devices (inlined by compiler)
    stamp_devices_static!(G, C, b, circuit, u, t)

    return (SMatrix(G), SMatrix(C), SVector(b))
end
```

#### GPU Phase 3: Sparse GPU Backend

For large circuits, add GPU sparse support:

```julia
struct GPUCircuit{T}
    G::CuSparseMatrixCSR{T}
    C::CuSparseMatrixCSR{T}
    b::CuVector{T}
    solver::CudssSolver
end

function solve_dc_gpu!(circuit::GPUCircuit)
    cudss_solve!(circuit.solver, circuit.b)
end
```

### Performance Expectations

| Scenario | CPU Time | GPU Time | Speedup |
|----------|----------|----------|---------|
| 1000x parameter sweep (10-node circuit) | 10s | 0.1s | 100x |
| Single 10k-node DC solve | 100ms | 20ms | 5x |
| Single 10k-node transient (1000 steps) | 100s | 10s | 10x |

### Current Blockers for GPU

1. **In-place formulation**: Current `dae_residual!` mutates - need out-of-place version
2. **Dynamic allocation**: `MNAContext` allocates vectors - need static version
3. **Sparse matrix rebuild**: GPU prefers fixed sparsity - need PrecompiledCircuit
4. **Float64 only**: MOSFET models use Float64 - need Float32 variants

All of these are addressed by the PrecompiledCircuit optimization, making GPU support a natural extension.

## References

- OpenVAF source: `/home/user/OpenVAF/`
- OSDI 0.4 header: `/home/user/OpenVAF/openvaf/osdi/header/osdi_0_4.h`
- OpenVAF internals: `/home/user/OpenVAF/internals.md`
- Melange OSDI device: `/home/user/OpenVAF/melange/core/src/veriloga/osdi_device.rs`
- SciML GPU docs: https://docs.sciml.ai/Overview/stable/showcase/massively_parallel_gpu/
- DiffEqGPU.jl: https://github.com/SciML/DiffEqGPU.jl
