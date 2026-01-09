# VACASK Benchmarks - CedarSim Integration Status

This document tracks the progress of adapting VACASK benchmarks to run with CedarSim.

## Overview

The VACASK benchmark suite (from https://codeberg.org/arpadbuermen/VACASK) tests circuit simulator performance with various circuit types. We've created `cedarsim/` folders alongside the original `ngspice/` folders to run these benchmarks with CedarSim's MNA backend.

All benchmarks use vadistiller-ported SPICE models equivalent to ngspice's built-in models.

## Benchmark Status

| Benchmark | Model | Status | Notes |
|-----------|-------|--------|-------|
| `rc` | None (R, C) | ✅ Working | ~1M timepoints in ~12s |
| `graetz` | `sp_diode` | ✅ Working | ~1M timepoints in ~24s |
| `mul` | `sp_diode` | ✅ Working | ~500k timepoints in ~11s |
| `ring` | `PSP103VA` | ✅ Working | 9-stage ring oscillator with PSP103 MOSFETs |
| `c6288` | `PSP103VA` | ❌ Broken | Requires sparse Jacobian fixes (see notes below) |

## Models Used

- **Diode** (`test/vadistiller/models/diode.va`): Used by graetz and mul benchmarks
- **PSP103** (`test/vadistiller/models/psp103v4/psp103.va`): Used by ring and c6288 benchmarks

## Fixed Issues

### 1. Parameter Name Collision (BoundsError)

The original error (`BoundsError(0.0, 2)`) was caused by PSP103 having a local variable `x = 0.0` (line 833) that shadowed the MNA framework's `x` parameter (solution vector). When voltage extraction tried `x[node_idx]`, it was doing `0.0[2]`.

**Fix:** Renamed MNA framework parameters to use `_mna_` prefix:
- `x` → `_mna_x_` (solution vector)
- `t` → `_mna_t_` (simulation time)
- `_sim_mode_` → `_mna_mode_` (simulation mode)
- `spec` → `_mna_spec_` (MNA spec)

### 2. Nested Variable Declarations (UndefVarError: Igdov)

PSP103 uses Verilog-A named blocks (`begin : evaluateblock`) with local variable declarations inside them. These nested declarations were not being extracted during code generation, causing `UndefVarError` for variables like `Igdov` when they were referenced in detection closures.

**Fix:** Added `collect_nested_var_decls!` function in `src/vasim.jl` that walks the analog block AST to find `IntRealDeclaration` nodes inside named `AnalogSeqBlock` nodes and adds them to `var_types` for proper initialization.

## Running Benchmarks

```bash
# All benchmarks are now working
~/.juliaup/bin/julia --project=. benchmarks/vacask/rc/cedarsim/runme.jl
~/.juliaup/bin/julia --project=. benchmarks/vacask/graetz/cedarsim/runme.jl
~/.juliaup/bin/julia --project=. benchmarks/vacask/mul/cedarsim/runme.jl
~/.juliaup/bin/julia --project=. benchmarks/vacask/ring/cedarsim/runme.jl
~/.juliaup/bin/julia --project=. benchmarks/vacask/c6288/cedarsim/runme.jl

# Or run all benchmarks with the CI script
~/.juliaup/bin/julia --project=. benchmarks/vacask/run_benchmarks.jl
```

## Code Changes Made

1. **Export `@sp_str` macro** (`src/CedarSim.jl`)
2. **Add `imported_hdl_modules` to `parse_spice_to_mna`** (`src/spc/interface.jl`)
3. **Fix `srcline` keyword** (`src/spc/interface.jl`)
4. **Fix stamp! keyword argument mismatch** (`src/spc/codegen.jl`, `src/mna/devices.jl`) - Changed `mode` to `_sim_mode_`
5. **Case-insensitive VA module lookup** (`src/spc/codegen.jl`) - Check both lowercase and original case
6. **Preserve parameter case for VA modules** (`src/spc/codegen.jl`) - VA modules may use uppercase params
7. **Pass `x` to subcircuit builders** (`src/spc/codegen.jl`) - For nonlinear devices that need solution vector
8. **Rename MNA parameters to avoid VA variable collision** (`src/vasim.jl`, `src/spc/codegen.jl`) - Use `_mna_x_`, `_mna_t_`, `_mna_mode_`, `_mna_spec_` to avoid collision with VA model local variables like `x = 0.0` in PSP103
9. **Extract nested variable declarations** (`src/vasim.jl`) - Added `collect_nested_var_decls!` to find and initialize variables declared inside named analog blocks

## Known Issues

### C6288 Benchmark (154k variables)

The C6288 multiplier benchmark does not currently work due to several interrelated issues:

1. **DC Operating Point Failure**: The circuit has no valid DC solution (typical for digital circuits). ngspice handles this with 'uic' (use initial conditions) which skips DC solve and starts from zeros.

2. **Memory Requirements**: With 154k variables, a dense Jacobian would require 154k² × 8 bytes = ~189GB. The KLU sparse solver is required.

3. **Sparse Jacobian Pattern Mismatch**: When using KLU, Sundials reports "Sparsity Pattern in receiving SUNMatrix doesn't match sending SparseMatrix". This occurs because:
   - The `jac_prototype` (G+C pattern) we provide to IDA has a certain sparsity structure
   - The actual Jacobian filled by `fast_jacobian!` may have a different structure due to:
     - G and C having different sparsity patterns
     - Numerical cancellation when G[i,j] = -C[i,j]
     - Broadcast operations (`.+=`) potentially creating new sparse matrices

**Potential Solutions** (not yet implemented):
- Precompute index mappings from G.nzval and C.nzval to J.nzval indices
- Use `abs.(G) .+ abs.(C)` for jac_prototype to prevent numerical cancellation
- Investigate if ShampineCollocationInit's internal finite-difference Jacobian causes pattern mismatches
- Consider passing `u0`/`du0` parameters to `tran!` to skip DC solve (like ngspice's 'uic')

See: https://sciml.ai/news/2025/09/17/sundials_v5_release/ for Sundials v5 IDA initialization options
