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

#### Root Cause Analysis

PSP103 uses named branches for **correlated noise modeling** (`PSP103_module.include` lines 46-49):

```verilog
electrical NOI;           // Internal node for noise correlation
branch (NOI) NOII;        // Named branch from NOI to ground (NOII = NOI-I-I for gate current noise)
branch (NOI) NOIR;        // Named branch (NOIR = NOI-R for resistive element)
branch (NOI) NOIC;        // Named branch (NOIC = NOI-C for capacitive element)
```

These branches are used in the noise section (lines 2781-2786):

```verilog
I(NOII) <+  white_noise((nt / mig), "igig");           // Noise current contribution to NOII
I(NOIR) <+  V(NOIR) / mig;                             // Resistive element
I(NOIC) <+  ddt(CGeff * V(NOIC));                      // Capacitive element
I(DI,SI) <+ sigVds * sqrt(MULT_i) * migid * I(NOII);  // READS I(NOII) as probe!
```

The critical issue is the last line: `I(NOII)` is used as a **probe** (reading the branch current) on the RHS of another contribution.

#### Code Flow in vasim.jl

1. **Contribution collection** (`vasim.jl:1219`): When `I(NOII) <+ expr` is parsed, it correctly identifies `is_branch=true` and `branch_name=:NOII`

2. **Contribution processing** (`vasim.jl:1350-1355`): Current contributions go to `branch_contribs` keyed by `(p, n)` tuple. The `branch_name` is **lost**:
   ```julia
   if c.kind == :current
       branch = (c.p, c.n)  # Only uses node pair, ignores branch_name!
       branch_contribs[branch] = [expr]
   ```

3. **Branch current variable allocation** (`vasim.jl:1382-1388`): `branch_current_vars` is only populated for **voltage contributions** (e.g., inductors):
   ```julia
   for (branch_name, _) in voltage_branch_contribs  # Only voltage contributions!
       branch_current_vars[branch_name] = I_var
   ```

4. **Branch current extraction** (`vasim.jl:1595-1600`): `_I_branch_NAME` is only defined for branches in `branch_current_vars` - which excludes current-contribution branches like NOII.

5. **Probe access** (`vasim.jl:541`): When `I(NOII)` is used as a probe, it returns `Symbol("_I_branch_NOII")` - but this variable was never defined!

#### Fix Required

The vasim.jl code generator needs to:

1. **Track named branches with current contributions** that are also used as probes
2. **Define `_I_branch_NAME`** for such branches, initialized to the sum of their contributions
3. Since `white_noise()` returns 0 in DC/transient (`vasim.jl:654-655`), the simplest fix is to initialize `_I_branch_NOII = 0.0` for noise-only branches

A more complete fix would:
- Accumulate all current contributions to named branches into `_I_branch_NAME`
- Make this accumulated value available before any code that reads `I(branch_name)`

#### Related Branches

PSP103 has three noise branches (NOII, NOIR, NOIC). The same fix applies to all branches that:
- Have current contributions (`I(branch) <+ expr`)
- Are read as probes (`I(branch)` on RHS)

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
