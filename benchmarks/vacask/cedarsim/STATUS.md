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
| `ring` | `PSP103VA` | ⚠️ Fails | PSP103 needs solution vector init |
| `c6288` | `PSP103VA` | ⚠️ Fails | PSP103 needs solution vector init |

## Models Used

- **Diode** (`test/vadistiller/models/diode.va`): Used by graetz and mul benchmarks
- **PSP103** (`test/vadistiller/models/psp103v4/psp103.va`): Used by ring and c6288 benchmarks

## PSP103 Issue

The ring and c6288 benchmarks fail because PSP103 is a complex compact model that accesses `V(node)` during stamping. The stamp! function tries to access `x[node_idx]` but the solution vector `x` is empty during initial assembly.

This is a fundamental limitation that needs to be addressed in the MNA solver's handling of nonlinear device initialization.

## Running Benchmarks

```bash
# Working benchmarks
~/.juliaup/bin/julia --project=. benchmarks/vacask/rc/cedarsim/runme.jl
~/.juliaup/bin/julia --project=. benchmarks/vacask/graetz/cedarsim/runme.jl
~/.juliaup/bin/julia --project=. benchmarks/vacask/mul/cedarsim/runme.jl

# Failing benchmarks (PSP103 issue)
~/.juliaup/bin/julia --project=. benchmarks/vacask/ring/cedarsim/runme.jl
~/.juliaup/bin/julia --project=. benchmarks/vacask/c6288/cedarsim/runme.jl
```

## Code Changes Made

1. **Export `@mna_sp_str` macro** (`src/CedarSim.jl`)
2. **Add `imported_hdl_modules` to `parse_spice_to_mna`** (`src/spc/interface.jl`)
3. **Fix `srcline` keyword** (`src/spc/interface.jl`)
4. **Fix stamp! keyword argument mismatch** (`src/spc/codegen.jl`, `src/mna/devices.jl`) - Changed `mode` to `_sim_mode_`
5. **Case-insensitive VA module lookup** (`src/spc/codegen.jl`) - Check both lowercase and original case
6. **Preserve parameter case for VA modules** (`src/spc/codegen.jl`) - VA modules may use uppercase params
7. **Pass `x` to subcircuit builders** (`src/spc/codegen.jl`) - For nonlinear devices that need solution vector
