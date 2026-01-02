# VADistiller Integration Tests Memory Analysis

## Executive Summary

The vadistiller integration tests cause OOM errors due to **cumulative JIT compilation memory** that is never returned to the OS. The primary culprit is the combination of:

1. **Triple-nested ForwardDiff duals** for voltage-dependent capacitor detection
2. **ODE solver compilation** for each unique circuit type
3. **Julia's memory management** (RSS never decreases even after GC)

## Profiling Results

### Baseline Memory
- Julia startup: ~464 MB RSS
- After loading CedarSim: **~1,110 MB RSS** (baseline)

### Model Loading Memory (sequential, all 16 models)
| Model Group | RSS After | Delta |
|-------------|-----------|-------|
| Initial | 1,106 MB | - |
| 3 passives (resistor, capacitor, inductor) | 1,190 MB | +84 MB |
| 4 small devices (mes1, jfet1, diode, jfet2) | 1,256 MB | +66 MB |
| 7 medium devices (mos1-9, bjt, vdmos) | 1,348 MB | +92 MB |
| bsim3v3 (4K lines) | 1,383 MB | +35 MB |
| bsim4v8 (10K lines) | 1,520 MB | +137 MB |
| **Total for model loading** | 1,520 MB | **+414 MB** |

### Simulation Memory (the real problem)
| Operation | RSS After | Delta | Time |
|-----------|-----------|-------|------|
| Load 4 models | 1,271 MB | - | - |
| RC circuit + Rodas5P | 1,335 MB | +64 MB | 12s |
| RC circuit + QNDF | 1,356 MB | +21 MB | 3s |
| **MOSFET + Rodas5P** | **2,952 MB** | **+1,596 MB** | **176s** |
| MOSFET + QNDF | 2,952 MB | +0 MB | 2s |
| MOSFET + IDA | 2,952 MB | +0 MB | 3s |

The MOSFET transient simulation with Rodas5P causes a **1.6 GB memory spike** due to JIT compilation of the ODE solver for the complex dual number types.

## Root Cause Analysis

### 1. Triple-Nested Duals (Charge Formulation)

The voltage-dependent capacitor detection in `src/mna/contrib.jl` uses three nested ForwardDiff tags:

```julia
# Precedence: CapacitanceDerivTag ≺ JacobianTag ≺ ContributionTag
# Result: Dual{ContributionTag, Dual{JacobianTag, Dual{CapacitanceDerivTag, Float64}}}
```

This creates complex types like:
```
Dual{ContributionTag, Dual{JacobianTag, Dual{CapacitanceDerivTag, Float64, 1}, N}, M}
```

Each unique type combination requires separate method specialization by Julia's JIT compiler.

### 2. ODE Solver Compilation

Implicit ODE solvers (Rodas5P, QNDF) require compilation for:
- Each unique circuit builder function
- Each unique dual number type used in the circuit
- Each unique Jacobian structure

When a MOSFET circuit with voltage-dependent capacitors is simulated, the solver must compile methods for all the nested dual types, causing massive memory allocation.

### 3. Julia Memory Management

Julia's GC does not return memory to the OS. The RSS stays high even when:
- Live bytes are low (~60-80 MB)
- GC has run and freed objects

This is a known Julia behavior, not a bug. RSS represents "high water mark" memory usage.

## Test Memory Budget

Based on profiling, the vadistiller integration tests require approximately:

| Phase | Memory Required |
|-------|-----------------|
| CedarSim baseline | 1,110 MB |
| All 16 models loaded | +414 MB |
| First RC transient (Rodas5P+QNDF) | +85 MB |
| First MOSFET transient (Rodas5P) | +1,600 MB |
| Additional circuit types | +200-500 MB each |
| **Estimated peak for full test suite** | **3,500-4,500 MB** |

## Recommendations

### Short-term (CI Fixes)

1. **Increase CI memory limit** to 8GB if possible
2. **Split tests into separate processes** using Julia's `--project` with individual test files:
   ```bash
   julia --project=. test/mna/vadistiller_tier6.jl
   julia --project=. test/mna/vadistiller_tier7.jl
   julia --project=. test/mna/vadistiller_tier8.jl
   ```
3. **Reduce solver diversity in tests** - not every circuit needs all 3 solvers

### Medium-term (Code Optimizations)

1. **Lazy charge formulation**: Only use triple duals when voltage-dependent capacitance is actually present
2. **Precompilation**: Add solver specializations to precompile.jl
3. **Type-stable circuits**: Use a single circuit builder pattern to reduce type diversity

### Long-term (Architecture)

1. **Consider worker processes**: Run each test group in a subprocess
2. **Memory-aware test ordering**: Run memory-heavy tests first, before memory accumulates
3. **Explicit memory limits**: Use `Base.Sys.set_process_limit()` to fail early on memory issues

## Conclusion

The OOM issue is **not a regression** in the traditional sense - it's the natural consequence of:
1. More complex type structures (charge formulation with triple duals)
2. More comprehensive tests (multiple solvers × multiple circuit types)
3. Julia's compile-once-per-type behavior

The memory usage is fundamentally driven by JIT compilation overhead, which scales with type complexity. The charge formulation feature (commit 33ae4be) added necessary complexity for voltage-dependent capacitors, but also increased the compilation burden.
