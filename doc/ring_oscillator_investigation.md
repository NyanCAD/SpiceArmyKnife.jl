# Ring Oscillator Benchmark Investigation

This document summarizes the investigation into why the ring oscillator benchmark "hangs and blows up."

## Summary of Findings

### 1. JIT Compilation Time (Root Cause of "Hang")

The PSP103 Verilog-A model generates a very large Julia `stamp!` method. First-call overhead:

| Phase | Time |
|-------|------|
| Model loading (parse + eval) | ~38s |
| JIT compilation of stamp! | ~150s |
| Actual stamp execution | ~0.02s |
| **Total first-call** | **~190s** |

**This is why the benchmark appears to "hang"** - it's waiting for JIT compilation, not stuck in an infinite loop.

### 2. System Properties

After assembly, the ring oscillator circuit has:
- **System size**: 371 unknowns
- **G matrix**: 2846 nonzeros, 137 zero diagonal entries
- **C matrix**: 1296 stored nonzeros, only 162 actual nonzeros (>1e-20)
- **C rank**: 72 / 371 (highly rank deficient)
- **DAE index**: The system is a DAE with 72 differential and 299 algebraic variables

### 3. Numerical Issues (Root Cause of "Blows Up")

The simulation fails due to:
1. **Singular Jacobian**: G has zero diagonal entries, making the Jacobian (G + C/dt) singular at small dt
2. **No stable DC equilibrium**: Ring oscillators don't have a stable DC operating point
3. **Very high condition number**: Even with GMIN regularization, cond(G) ≈ 6.57e18

### 4. Solver Behaviors

| Solver | Result |
|--------|--------|
| IDA | Segfault during transient |
| FBDF | "Newton steps could not converge" |
| ImplicitEuler | "Solver failed" (singular matrix) |

## Root Causes

1. **JIT compilation time**: The PSP103 model is ~3000 lines of Verilog-A that expands to a massive Julia function. Julia's JIT compiler takes ~150s for first compile.

2. **Singular Jacobian**: The MNA formulation creates a system where:
   - Many rows have zero diagonal (current variables, internal nodes)
   - The mass matrix C is rank-deficient (only capacitive nodes have dynamics)
   - Without regularization, the Jacobian is singular

3. **DAE structure**: Ring oscillators are true DAEs (not ODEs). They require:
   - Proper DAE solvers (IDA should work but has a bug)
   - Consistent initialization (no stable DC point to start from)
   - GMIN or source stepping to regularize during init

## GMIN Regularization Test

Adding GMIN = 1e-12 to all G diagonal entries:
- Removes all zero diagonal entries
- Enables linear solve to succeed
- Condition number still high but finite (6.57e18)
- V(vdd) correctly solved to 1.2V

## Recommendations

### Short-term Fixes

1. **Increase benchmark timeout**: Account for 190s+ JIT compilation
2. **Precompile PSP103**: Add to precompile workload or use a precompiled device package
3. **Add GMIN option**: Implement GMIN stepping in CedarDCOp/CedarUICOp

### Medium-term Improvements

1. **Fix IDA integration**: The segfault during transient needs investigation
2. **Implement source stepping**: Homotopy method for difficult convergence
3. **Use BSIM4 instead**: The test BSIM4 model works and compiles in 76s (vs 150s for PSP103)

### Code Changes Needed

1. **dcop.jl**: Add GMIN option to CedarUICOp:
```julia
struct CedarUICOp <: DiffEqBase.DAEInitializationAlgorithm
    warmup_steps::Int
    dt::Float64
    gmin::Float64  # New: conductance to add to diagonal
end
CedarUICOp(; warmup_steps=10, dt=1e-12, gmin=0.0) = CedarUICOp(warmup_steps, dt, gmin)
```

2. **solve.jl**: Apply GMIN during assembly when spec.mode == :dcop

3. **Benchmark**: Use BSIM4 or add explicit precompile step before timing

## Test Results

The following test confirms GMIN regularization enables solving:
```julia
# After adding GMIN = 1e-12 to G diagonal
DC solve succeeded!
Max voltage: 16.18V  # Note: linear solve at u=0, not physically meaningful
V(vdd): 1.2V  # Correct supply voltage
```

## Files Created During Investigation

- `diagnose_ring.jl` - Initial diagnostic
- `diagnose_ring_simple.jl` - Simplified diagnostic
- `diagnose_ring_builder.jl` - Builder isolation test
- `diagnose_single_psp.jl` - Single PSP103 test
- `diagnose_single_bsim4.jl` - Single BSIM4 test
- `test_ring_gmin.jl` - GMIN regularization test
