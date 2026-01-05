# JAX-SPICE vs SpiceArmyKnife.jl: C6288 Benchmark Comparison

This document analyzes how jax-spice handles the c6288 benchmark (16x16 bit multiplier with 10k+ MOSFETs) and compares it to SpiceArmyKnife.jl's current approach, identifying priorities for getting c6288 working.

## Circuit Characteristics

The c6288 is a challenging benchmark:
- **Size**: ~86k nodes in jax-spice / 154k variables in SpiceArmyKnife.jl
- **Devices**: 10k+ PSP103 MOSFETs
- **Memory**: Dense Jacobian would need 189GB (154k² × 8 bytes)
- **Solver**: Requires sparse linear solver (KLU, UMFPACK, etc.)
- **DC Solution**: Digital circuits often have no valid DC solution

## JAX-SPICE Approach

### Convergence Techniques Implemented

jax-spice implements a comprehensive VACASK-style homotopy chain (`jax_spice/analysis/homotopy.py`):

1. **GMIN Stepping** (two modes):
   - `gdev`: Extra conductance added to device Jacobian diagonals
   - `gshunt`: Shunt conductance from all nodes to ground
   - Adaptive stepping: starts at 1e-3 S, steps down to 1e-13 S
   - Factor adjustment based on NR iteration count

2. **Source Stepping**:
   - Ramps voltage/current sources from 0% to 100%
   - Falls back to GMIN stepping at source_factor=0 if initial solve fails
   - Adaptive step adjustment: starts at 0.1, min 0.001

3. **Homotopy Chain**: `gdev → gshunt → src`
   - Tries each method in sequence until one succeeds
   - Uses best solution from failed attempt as starting point for next

### Newton-Raphson Configuration

From `solver.py`:
```python
@dataclass
class NRConfig:
    max_iterations: int = 50
    abstol: float = 1e-12
    reltol: float = 1e-3
    damping: float = 1.0        # Damping factor for updates
    max_step: float = 2.0       # Maximum voltage change per iteration
```

Key features:
- **Step limiting**: `step_scale = min(damping, max_step / delta_norm)`
- **Regularization**: Adds 1e-14 * I to Jacobian for stability
- **Dual convergence**: Checks both residual norm AND delta convergence

### C6288-Specific Fixes (from git history)

1. **Voltage source initialization** (commit `2b53809`):
   - Previously only nodes with 'vdd'/'vcc' in name were initialized
   - Now ALL vsource positive nodes are set to their DC target values
   - c6288 has 34 voltage sources with various names

2. **Tolerance adjustment** (same commit):
   - Changed `DEFAULT_ABSTOL` from 1e-3 A to 1e4 A (10kA)
   - With G=1e12 for vsources, 1mA residual = 1fV voltage error (too strict)
   - 10kA corresponds to ~10nV voltage accuracy (reasonable)

3. **UIC mode initialization** (commit `82cdf26`):
   - Initialize internal CMOS nodes to mid-rail (VDD/2) instead of zeros
   - Set NOI (Node Of Interest) nodes to 0V for PSP103 devices
   - Set body internal nodes to match bulk voltage
   - VDD/VCC nodes to supply, GND/VSS to ground

4. **Sparse solver support**:
   - UMFPACK for CPU (fast)
   - Spineax for GPU (with COO→CSR precomputation)
   - cuDSS for NVIDIA GPUs

### Critical Insight: JAX-SPICE Ignores Convergence Failures

**This is the key difference from SciML/IDA.**

jax-spice uses a simple forward/backward Euler timestepping loop that **does not require Newton convergence**. From `engine.py` (lines 3095-3108):

```python
def step_fn(carry, source_vals):
    V, Q_prev = carry
    vsource_vals, isource_vals = source_vals
    V_new, iterations, converged, max_f, Q = nr_solve_fn(
        V, vsource_vals, isource_vals, Q_prev, inv_dt_val, device_arrays_arg
    )
    # Uses V_new regardless of whether converged is True or False!
    return (V_new, Q), (V_new[:n_ext], iterations, converged)
```

They track convergence as a **statistic**, not a stopping condition:

```python
non_converged = int(jnp.sum(~all_converged))
stats = {
    'non_converged_count': non_converged,
    'convergence_rate': 1.0 - non_converged / max(num_timesteps, 1),
    ...
}
```

**Why this works for them:**
1. **Fixed small timestep** (e.g., 0.05ns) - errors don't accumulate catastrophically
2. **Capacitors provide damping** - even unconverged Newton gives bounded updates
3. **No consistency requirement** - they don't need `F(du0, u0, 0) = 0`
4. **Backward Euler is unconditionally stable** - won't blow up even with large errors

**Why this doesn't work with SciML/IDA:**
1. **IDA requires consistent initial conditions**: `F(du0, u0, 0) = 0` must be satisfied
2. **IDA fails on Newton non-convergence**: The entire simulation stops
3. **IDA uses adaptive timestep control**: Based on local error estimates that assume convergence
4. **IDA is a proper DAE solver**: It solves the algebraic constraints exactly

**Implications for SpiceArmyKnife.jl:**

To match jax-spice's behavior, we would need to either:
1. **Implement a custom Euler integrator** that ignores convergence like jax-spice
2. **Use `initializealg = NoInit()`** to skip IDA's consistency check (risky)
3. **Pre-compute very good initial conditions** so IDA's first Newton succeeds

The jax-spice approach is less rigorous but pragmatic for digital circuits where:
- The capacitances naturally damp oscillations
- Small timesteps keep local errors bounded
- Exact algebraic constraint satisfaction isn't critical

### Current Status in JAX-SPICE

From `registry.py`:
```python
if name == "c6288":
    info.max_steps = 20
    info.xfail = True
    info.xfail_reason = "Node count mismatch - need node collapse"
```

The DC operating point is now correct (after PHI node translation fix), but there are still node count issues related to node collapse.

## SpiceArmyKnife.jl Current State

### What's Implemented

1. **GMIN Support** (`src/mna/build.jl`):
   - `assemble_G(ctx; gmin=0.0)` adds GMIN from each voltage node to ground
   - Default `gmin = 1e-12` in MNASpec
   - Applied during assembly, not dynamic stepping

2. **DC Solver** (`src/mna/solve.jl`):
   - Uses NonlinearSolve.jl with `RobustMultiNewton()`
   - Provides explicit Jacobian (G matrix) for efficiency
   - Linear solve first, then Newton iteration if needed

3. **DAE Integration**:
   - Uses Sundials IDA with KLU sparse solver
   - Explicit Jacobian (G + gamma*C) provided via `jac_prototype`

### Current Issues (from STATUS.md)

1. **DC Operating Point Failure**:
   - `SingularException` - expected for digital circuits
   - No `uic` (use initial conditions) mode to skip DC

2. **Sparse Jacobian Pattern Mismatch**:
   - KLU reports "Sparsity Pattern in receiving SUNMatrix doesn't match"
   - `jac_prototype` (G+C pattern) doesn't match actual Jacobian from `fast_jacobian!`
   - Root causes:
     - G and C have different sparsity patterns
     - Numerical cancellation when G[i,j] = -C[i,j]
     - Broadcast operations (`.+=`) potentially creating new patterns

### Missing Features Compared to JAX-SPICE

| Feature | JAX-SPICE | SpiceArmyKnife.jl |
|---------|-----------|-------------------|
| GMIN stepping | ✅ Adaptive | ❌ Static only |
| Source stepping | ✅ Adaptive | ❌ Not implemented |
| Homotopy chain | ✅ gdev→gshunt→src | ❌ Not implemented |
| UIC mode | ✅ Mid-rail init | ❌ Not implemented |
| Step limiting | ✅ max_step=2.0 | ❌ Relies on solver |
| Damping | ✅ Configurable | ❌ Relies on solver |
| Voltage source init | ✅ To DC values | ❌ Not implemented |
| NOI node handling | ✅ Mask residuals | ❌ Not implemented |
| Sparse solver | ✅ UMFPACK/cuDSS | ⚠️ KLU (pattern issues) |

## Priorities for SpiceArmyKnife.jl

### Priority 1: Fix Sparse Jacobian Pattern (Blocking)

**Problem**: KLU rejects Jacobian due to pattern mismatch

**Solutions**:
1. Use structural union for jac_prototype: `abs.(G) .+ abs.(C)` prevents numerical cancellation
2. Precompute index mappings from G.nzval/C.nzval to J.nzval indices
3. Consider using fixed sparsity pattern throughout simulation

### Priority 2: Implement UIC Mode

**Problem**: DC solve fails for digital circuits

**Solutions**:
1. Add `u0`/`du0` parameters to `tran!` to skip DC solve
2. Implement smart initialization like jax-spice:
   - VDD/VCC nodes → supply voltage
   - GND/VSS → 0V
   - Other nodes → mid-rail (VDD/2)
   - NOI nodes → 0V
   - Body internal nodes → bulk voltage

### Priority 3: Implement Homotopy Chain (Nice to Have)

**Benefit**: Robust DC convergence for difficult circuits

**Implementation**:
1. GMIN stepping with adaptive factor
2. Source stepping with adaptive step
3. Chain: try gdev, then gshunt, then src

### Priority 4: Voltage Limiting (Optional)

**Benefit**: Prevents Newton divergence on large updates

**Implementation**:
- Add `max_step` parameter (default 2.0V)
- Scale delta_V if too large: `step_scale = min(1.0, max_step / norm(delta_V))`

## Recommended Implementation Order

1. **Quick win**: Fix jac_prototype pattern (use abs.(G) .+ abs.(C))
2. **Medium effort**: Implement UIC mode (skip DC, use smart init)
3. **Larger effort**: Implement GMIN/source stepping if needed
4. **Long term**: Consider custom Newton solver with limiting

## Key Learnings from JAX-SPICE

1. **Tolerance matters**: Very strict abstol causes unnecessary iterations
2. **Initialization matters**: Mid-rail + supply voltages converge faster than zeros
3. **Adaptive stepping**: Fixed step sizes often too aggressive or too conservative
4. **Multiple fallbacks**: When one method fails, try another
5. **Explicit is better**: Providing Jacobian avoids finite differencing issues

## References

- jax-spice homotopy: `/home/user/jax-spice/jax_spice/analysis/homotopy.py`
- jax-spice solver: `/home/user/jax-spice/jax_spice/analysis/solver.py`
- jax-spice engine: `/home/user/jax-spice/jax_spice/analysis/engine.py`
- VACASK: https://codeberg.org/arpadbuermen/VACASK
