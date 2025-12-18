# MNA Backend Test Status

## Summary

The MNA backend is a work-in-progress replacement for DAECompiler-based simulation.
Current status as of commit.

## Working Features

### DC Analysis
- ✅ Linear circuits (R, V, I sources)
- ✅ Voltage dividers
- ✅ RC circuits (capacitor as open circuit in DC)
- ✅ BSIMCMG basic DC

### VA Device Support
- ✅ VA resistor (`I(p,n) <+ V(p,n)/R`)
- ✅ VA capacitor with ddt (`I(p,n) <+ ddt(C*V(p,n))`)
- ✅ BSIMCMG loading and evaluation
- ✅ Alias parameters
- ✅ `$param_given` checks

### Parameterization
- ✅ Immutable params struct (enables constant folding)
- ✅ ParamLens integration (selective sweeps without recompilation)
- ✅ Nested lens structure for device hierarchy

### Integration
- ✅ MNA-Spectre environment layer
- ✅ Named device wrapper
- ✅ ParallelInstances (m parameter)
- ✅ spicecall helper

## Not Yet Implemented

### Analysis Types
- ❌ AC Analysis
- ❌ Noise Analysis
- ❌ Monte Carlo (agauss, etc.)

### Transient
- ⚠️ Basic transient structure exists
- ⚠️ Companion model integration (ddt) implemented
- ❌ Numerical stability issues with BSIMCMG transient

### Other
- ❌ Solution observability (sys.R.I, sys.R.V style access)
- ❌ Debug IR logging
- ❌ CairoMakie plotting integration

## Test Files

| File | Status | Notes |
|------|--------|-------|
| test/test_mna_spectre.jl | ✅ 24/24 pass | MNA integration layer |
| test/basic.jl | ❌ | Uses DAECompiler API |
| test/transients.jl | ❌ | Uses DAECompiler API |
| test/ac.jl | ❌ | AC not implemented |
| test/inverter_noise.jl | ❌ | Noise not implemented |
| test/bsimcmg/*.jl | ❌ | Uses DAECompiler API |

## Comparison with DAECompiler

| Feature | DAECompiler | MNA |
|---------|-------------|-----|
| DC Analysis | ✅ | ✅ |
| Transient | ✅ | ⚠️ |
| AC | ✅ | ❌ |
| Noise | ✅ | ❌ |
| ParamLens | ✅ | ✅ |
| Solution Access | ✅ sys.dev.V | ❌ |
| BSIMCMG | ✅ | ✅ DC only |
| Compile Time | Slow | Fast |
| Runtime | Optimized | Good |

## Next Steps

1. Fix transient numerical stability with BSIMCMG
2. Add solution observability (sys.dev.V style)
3. Consider AC analysis implementation
4. Port more tests to MNA-compatible format
