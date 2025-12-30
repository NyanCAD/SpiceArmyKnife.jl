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
| `ring` | `PSP103VA` | ⚠️ Fails | PSP103 branch current issue (`_I_branch_NOII`) |
| `c6288` | `PSP103VA` | ⚠️ Fails | PSP103 branch current issue (`_I_branch_NOII`) |

## Models Used

- **Diode** (`test/vadistiller/models/diode.va`): Used by graetz and mul benchmarks
- **PSP103** (`test/vadistiller/models/psp103v4/psp103.va`): Used by ring and c6288 benchmarks

## PSP103 Issue

### Fixed: Parameter Name Collision (BoundsError)

The original error (`BoundsError(0.0, 2)`) was caused by PSP103 having a local variable `x = 0.0` (line 833) that shadowed the MNA framework's `x` parameter (solution vector). When voltage extraction tried `x[node_idx]`, it was doing `0.0[2]`.

**Fix:** Renamed MNA framework parameters to use `_mna_` prefix:
- `x` → `_mna_x_` (solution vector)
- `t` → `_mna_t_` (simulation time)
- `_sim_mode_` → `_mna_mode_` (simulation mode)
- `spec` → `_mna_spec_` (MNA spec)

### Current Issue: Branch Current Handling (`_I_branch_NOII`)

After fixing the parameter collision, PSP103 now fails with:
```
UndefVarError: `_I_branch_NOII` not defined in `Main.PSP103VA_module`
```

This is a different issue related to how PSP103 uses named branches for internal currents. The model uses `I(<branch_name>)` syntax for branches like `NOII` (no-impact ionization?), but the vasim code generator isn't properly defining these branch current variables.

**Investigation needed:**
1. PSP103 uses named branches: `branch (DI, DG) NOII;` (or similar)
2. The VA-to-Julia translation needs to define `_I_branch_NOII` for `I(NOII)` access
3. Check vasim.jl branch handling code to ensure named branches are properly translated

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
8. **Rename MNA parameters to avoid VA variable collision** (`src/vasim.jl`, `src/spc/codegen.jl`) - Use `_mna_x_`, `_mna_t_`, `_mna_mode_`, `_mna_spec_` to avoid collision with VA model local variables like `x = 0.0` in PSP103
