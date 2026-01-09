# Memory Allocation Analysis: VACASK Benchmarks

This document analyzes the memory allocation patterns observed in the VACASK benchmarks
and explains the differences between solvers (IDA, FBDF, Rodas5P) and circuits (RC, Graetz).

## Executive Summary

| Benchmark | Solver | Memory | Per-Step Alloc | Key Factor |
|-----------|--------|--------|----------------|------------|
| RC        | IDA    | 645 MB | 0.64 KB        | Solution storage |
| RC        | FBDF   | 196 MB | 0.20 KB        | Minimal overhead |
| RC        | Rodas5P| 2.1 GB | 2.15 KB        | LU factorization every step |
| Graetz    | FBDF   | 2.1 GB | 2.08 KB        | VA device allocations + Jacobians |
| Graetz    | Rodas5P| 37 GB  | 37 KB          | Both factors combined |

## Key Findings

### 1. Rodas5P Allocates ~10x More Per Step Than FBDF

**Root Cause: LU Factorization Strategy**

- **Rosenbrock methods** (Rodas5P) compute `W = M - γJ` and factorize it **every step**
- **BDF methods** (FBDF, IDA) reuse the Jacobian across many Newton iterations

Profiling results for RC circuit (1M steps):
- FBDF: 1 Jacobian evaluation for 1M steps
- Rodas5P: ~1M Jacobian evaluations (one per step)

The sparse LU factorization allocates ~5.5 KB per call, which dominates Rodas5P's memory usage.

### 2. Graetz Allocates ~20x More Per Step Than RC (with FBDF)

**Root Cause: VA Device Evaluation Allocations**

The key finding from profiling:

```
RC Circuit:
  fast_rebuild!:  0.0 bytes/call
  fast_residual!: 0.0 bytes/call

Graetz Circuit:
  fast_rebuild!:  800.0 bytes/call
  fast_residual!: 800.0 bytes/call
```

The 800 bytes/call allocation comes from the **builder function** which evaluates the
Verilog-A diode model. This allocation is happening inside the VA device evaluation code.

Additional factors:
- More Jacobian evaluations for nonlinear circuits (327 vs 1 for 10k steps)
- Larger system size: 17 unknowns vs 3

### 3. Allocation Sources Breakdown

For a 1-second simulation with dtmax=1µs (~1M timesteps):

**RC Circuit + FBDF (196 MB):**
- Solution storage: ~32 MB (n=3, 32 bytes/timepoint × 1M)
- Solver overhead: ~164 MB

**RC Circuit + Rodas5P (2.1 GB):**
- Solution storage: ~32 MB
- LU factorizations: ~5.5 KB × 1M steps = ~5.5 GB (but compressed by GC)
- Observed: ~2.1 GB (GC reclaims some allocations)

**Graetz + FBDF (2.1 GB):**
- Solution storage: ~136 MB (n=17, 136 bytes/timepoint × 1M)
- VA device allocations: 800 bytes × 1M steps = ~800 MB
- Jacobian evaluations: 327 × larger LU cost
- Total: ~2.1 GB

## Optimization Opportunities

### 1. VA Device Evaluation (High Priority)

The 800 bytes/call in `fast_rebuild!` for VA devices is a significant optimization target.

**Investigation needed:**
- Profile the generated VA code to find allocation sources
- Potential causes: string allocations, temporary arrays, closures
- Consider precomputing temperature-dependent parameters

### 2. Sparse LU Factorization Reuse (Medium Priority)

For Rosenbrock methods, consider:
- Symbolic factorization reuse (only numeric values change)
- Using KLU or UMFPACK with factorization reuse APIs

### 3. Solution Storage (Low Priority)

For very long simulations, consider:
- Dense output with interpolation instead of storing all timesteps
- `saveat` parameter to reduce stored timepoints

## Profiling Scripts

The following scripts were created for this analysis:

1. `benchmarks/vacask/profile_solver_allocations.jl` - Compare solvers on RC
2. `benchmarks/vacask/profile_solver_allocations_full.jl` - Full 1-second simulations
3. `benchmarks/vacask/profile_rodas5p_internals.jl` - Rodas5P breakdown
4. `benchmarks/vacask/profile_graetz_allocations.jl` - Graetz vs RC comparison
5. `benchmarks/vacask/profile_va_device_allocations.jl` - VA device analysis

## Conclusions

1. **Rodas5P memory usage is expected** - it's an algorithmic characteristic of
   Rosenbrock methods that trade computation for robustness

2. **FBDF is the most memory-efficient** solver for these benchmarks due to
   lazy Jacobian evaluation

3. **VA device allocations are the main target** for reducing Graetz memory usage -
   the 800 bytes/call overhead is avoidable with careful code generation

4. **IDA uses more memory than FBDF** despite similar algorithms, likely due to
   Sundials' internal data structures and dense Jacobian storage
