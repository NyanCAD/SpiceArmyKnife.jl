# VACASK Benchmarks - CedarSim Integration Status

This document tracks the progress of adapting VACASK benchmarks to run with CedarSim.

## Overview

The VACASK benchmark suite (from https://codeberg.org/arpadbuermen/VACASK) tests circuit simulator performance with various circuit types. We've created `cedarsim/` folders alongside the original `ngspice/` folders to run these benchmarks with CedarSim's MNA backend.

## Working Benchmarks ✅

### 1. RC Circuit (`rc/cedarsim/`)
- **Status**: Fully working
- **Description**: RC circuit excited by a pulse train (simple linear circuit)
- **Performance**: ~1M timepoints in ~12s
- **Files**: `runme.jl`, `runme.sp`

### 2. Graetz Bridge (`graetz/cedarsim/`)
- **Status**: Fully working
- **Description**: Full-wave rectifier with 4 diodes, capacitor, and resistor load
- **Performance**: ~1M timepoints in ~24s
- **Files**: `runme.jl`, `runme.sp`
- **Notes**: Uses inline VA diode model (`sp_diode`) defined in `runme.jl`

### 3. Diode Voltage Multiplier (`mul/cedarsim/`)
- **Status**: Fully working
- **Description**: Voltage multiplier with 4 diodes and 4 capacitors
- **Performance**: ~500k timepoints in ~11s
- **Files**: `runme.jl`, `runme.sp`
- **Notes**: Uses inline VA diode model with transit time support

## Failing Benchmark ⚠️

### 4. BJT Ring Oscillator (`vadistiller/bjtring/cedarsim/`)
- **Status**: Fails at runtime
- **Description**: 9-stage ring oscillator using BJT transistors
- **Files**: `runme.jl`, `runme.sp`
- **Error**: `UndefVarError: t2n2222_mna_builder not defined`

#### Root Cause
The SPICE file uses `.model` to create an alias for a VA module:
```spice
.model t2n2222 sp_bjt type=1 subs=1 is=19f bf=150 ...

.subckt invcell in out vcc vee
  xq1 out b vee 0 t2n2222   ; References the model alias
  ...
.ends
```

The codegen incorrectly looks for `t2n2222_mna_builder` (as if it were a user-defined subcircuit) instead of recognizing that `t2n2222` is a model alias for the `sp_bjt` VA module.

#### Fix Required
In `src/spc/codegen.jl`, the `cg_mna_instance!` function needs to:
1. Check if the subcircuit name matches a `.model` that references a VA module
2. If so, generate a VA device stamp call with the model's parameters
3. Instead of looking for a `_mna_builder` function

## Not Adapted (No Julia Runner) 📋

### 5. Ring Oscillator (`ring/cedarsim/`)
- **Blocker**: Requires PSP103 MOSFET VA model
- **Notes**: Original uses complex MOSFET model from external file

### 6. C6288 Multiplier (`c6288/cedarsim/`)
- **Blocker**: Requires native SPICE MOSFET support
- **Notes**: Uses `.model ... nmos/pmos` built-in models

### 7. VADistiller C6288 (`vadistiller/c6288/cedarsim/`)
- **Blocker**: Requires vadistiller-generated MOSFET model
- **Notes**: Uses MOSFET model from VADistiller

### 8. JFET Ring (`vadistiller/jfetring/cedarsim/`)
- **Blocker**: Requires vadistiller-generated JFET model

### 9. VADistiller Multiplier (`vadistiller/mul/cedarsim/`)
- **Blocker**: Requires vadistiller-generated diode model
- **Notes**: Could potentially use the inline diode from `mul/cedarsim/`

## Code Changes Made

### 1. Export `@mna_sp_str` macro (`src/CedarSim.jl`)
```julia
export @mna_sp_str
```

### 2. Add `imported_hdl_modules` to `parse_spice_to_mna` (`src/spc/interface.jl`)
```julia
function parse_spice_to_mna(spice_code::String; circuit_name::Symbol=:circuit,
                            imported_hdl_modules::Vector{Module}=Module[])
```

### 3. Fix `srcline` keyword (`src/spc/interface.jl`)
Changed `line_offset` to `srcline` in the `@mna_sp_str` macro.

### 4. Fix stamp! keyword argument mismatch (`src/spc/codegen.jl`, `src/mna/devices.jl`)
Changed `mode = spec.mode` to `_sim_mode_ = spec.mode` in all stamp! calls for VA modules and built-in time-dependent sources. This matches the VA-generated stamp! signature which uses `_sim_mode_` to avoid conflicts with VA parameter names.

## VADistiller Models

The VADistiller-generated models are in `benchmarks/vacask/models/`:
- `diode.va` - SPICE diode model in Verilog-A
- `bjt.va` - SPICE BJT model in Verilog-A

These models use advanced Verilog-A features like `$limit`, `$simparam`, etc. that may require parser/codegen support.

## Running Benchmarks

```bash
# RC circuit
~/.juliaup/bin/julia --project=. benchmarks/vacask/rc/cedarsim/runme.jl

# Graetz bridge
~/.juliaup/bin/julia --project=. benchmarks/vacask/graetz/cedarsim/runme.jl

# Diode multiplier
~/.juliaup/bin/julia --project=. benchmarks/vacask/mul/cedarsim/runme.jl
```

## Next Steps

1. **Fix `.model` alias for VA modules**: Update codegen to recognize when an X-device references a model that aliases a VA module

2. **Add native MOSFET support**: For c6288 and ring benchmarks that use built-in SPICE MOSFETs

3. **Validate vadistiller models**: Check which Verilog-A features are missing for the vadistiller-generated models

4. **Performance comparison**: Compare CedarSim results with ngspice reference results
