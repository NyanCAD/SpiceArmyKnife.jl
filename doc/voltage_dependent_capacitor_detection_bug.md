# Voltage-Dependent Capacitor Detection Bug in Complex VA Models

## Summary

The charge detection mechanism in `vasim.jl` fails to correctly detect voltage-dependent
capacitances in complex Verilog-A models like PSP103. This causes all charges to be
classified as "linear" and stamped into the C matrix directly, rather than using the
charge formulation that provides a constant mass matrix.

## Investigation Results

### Observed Behavior

When running the VACASK ring oscillator benchmark with PSP103 MOSFETs:

```
=== MNA Context Summary ===
Nodes: 154
Currents: 127
Charges: 0  <-- Should be non-zero for PSP103's nonlinear capacitors
System size: 281
Charge is vdep cache: Bool[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

The detection cache has 18 entries (one per MOSFET), all classified as `false`
(not voltage-dependent).

### Root Cause

The detection lambda in `vasim.jl` (lines 1569-1593) is structured as:

```julia
function (_Vpn)
    let D = 0.0 * _Vpn, G = 0.31 * _Vpn, S = 0.57 * _Vpn, ...
        # sum_expr references Qfgd, Igdov, CGeff, CHNL_TYPE, MULT_i
        (CHNL_TYPE * MULT_i) * Igdov + va_ddt((CHNL_TYPE * MULT_i) * Qfgd) + ...
    end
end
```

The problem: Variables like `Qfgd`, `Igdov`, `CGeff`, `CHNL_TYPE`, `MULT_i` are
**captured from the outer scope**, not recomputed within the lambda. These values
are computed by the PSP103 model based on the ORIGINAL node voltages, not the
test voltages (`_Vpn`) provided by the detection mechanism.

When the detection algorithm varies `_Vpn` to test if capacitance changes:
1. The local node variables (D, G, S, B, etc.) ARE updated based on `_Vpn`
2. But `Qfgd`, `Igdov`, etc. remain CONSTANT (captured from outer scope)
3. So the charge appears constant, and detection returns `false`

### Why Simple Models Work

For simple contribution expressions like `va_ddt(C0 * V^2)`, the charge value
`C0 * V^2` is computed directly from the local voltage variable `V`, which IS
updated when `_Vpn` changes. So detection works correctly.

### Impact

- All PSP103 capacitances (Qg, Qb, Qd, junction caps) are treated as linear
- The C matrix contains voltage-dependent capacitances that should be constant
- Rosenbrock solvers may have convergence issues due to non-constant mass matrix
- Junction capacitance nonlinearity is not properly handled

## Verification Tests

1. **Unit tests pass**: The charge detection tests in `test/mna/charge_formulation.jl`
   use simple inline expressions where the charge directly depends on the voltage
   variable, so they correctly detect voltage-dependence.

2. **Complex models fail**: PSP103 and similar compact models have many intermediate
   calculations that depend on voltages. The detection lambda captures the final
   computed values rather than the computation chain.

## Potential Fixes

### Option 1: Full Model Re-evaluation (Expensive)

Include all PSP103 model calculations inside the detection lambda so they're
re-executed with the test voltages. This requires significant code generation
changes and would increase compilation time.

### Option 2: Conservative Detection (Always Assume Nonlinear)

Assume all `ddt()` calls in VA models produce voltage-dependent capacitances.
Use charge formulation for all of them. This adds extra state variables but
guarantees correctness.

### Option 3: Static Analysis

Analyze the VA AST to determine if charge expressions contain direct voltage
references only (linear) or involve intermediate computed values (potentially
nonlinear).

### Option 4: Runtime Sampling at Actual Operating Points

Instead of testing at fixed voltages, sample the capacitance at multiple
actual operating points during simulation startup. This would require changes
to the simulation initialization logic.

## Affected Files

- `src/vasim.jl` (lines 1566-1593): Detection lambda generation
- `src/mna/contrib.jl`: `is_voltage_dependent_charge()` implementation
- `benchmarks/vacask/`: All benchmarks using PSP103 MOSFETs

## Implemented Fix: Q/V Ratio Detection

As of January 2026, we implemented **Q/V ratio comparison** across multiple operating points:

The key insight is that while `∂Q/∂V` from ForwardDiff doesn't capture intermediate
value dependencies (because intermediates are computed before JacobianTag duals exist),
the actual **Q value itself DOES change correctly** when x changes.

### Detection Algorithm

1. **Pass 1**: Use `ZERO_VECTOR` to discover structure and get initial system size
2. **Pass 2-3**: Use random voltages (0-0.8V), compare Q/V ratios:
   - For linear capacitor: Q = C*V, so Q/V = C (constant across operating points)
   - For nonlinear capacitor: Q = f(V), so Q/V varies

If the Q/V ratio differs significantly between passes, the charge is marked as
voltage-dependent and uses charge formulation.

### Results

```
=== PSP103 Ring Oscillator ===
Nodes: 154
Currents: 127
Charges (nonlinear): 90   <-- Only truly nonlinear charges!
System size: 371
```

Compared to conservative detection (144 charges), this correctly identifies only
90 nonlinear charges, saving 54 state variables.

### Trade-offs

- **Pro**: Accurate detection - only nonlinear caps use charge formulation
- **Pro**: Fewer state variables than conservative approach
- **Con**: Requires 3 builder passes (minor compilation overhead)
- **Note**: Simple inline expressions still use accurate lambda-based detection

### Files Modified

- `src/mna/contrib.jl`: `detect_or_cached!` compares Q/V ratios
- `src/mna/solve.jl`: `assemble!(::MNACircuit)` runs 3 detection passes
- `src/mna/context.jl`: Store `charge_Q_values` and `charge_V_values` for comparison
- `src/vasim.jl`: Pass `V_branch` and `q_val` to detection
