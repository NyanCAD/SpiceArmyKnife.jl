# API Unification Plan for Cadnip.jl

This document outlines a comprehensive refactoring plan to unify the circuit simulation APIs in Cadnip.jl. The goal is to create a single, consistent API that supports the full feature set across all solvers and is easy to use.

---

## Part 1: Exploration - Active Test and Benchmark Entry Points

### 1.1 GitHub Actions Workflows

**CI Workflow** (`.github/workflows/ci.yml`):
- Runs `Pkg.test(test_args=["mna"])` for MNA tests
- Uses Julia 1.10 and 1.11

**Benchmark Workflow** (`.github/workflows/benchmark.yml`):
- Runs `benchmarks/vacask/run_benchmarks.jl`
- Tests IDA, FBDF, and Rodas5P solvers on 5 VACASK circuits

### 1.2 Test Entry Points

**Main Entry**: `test/runtests.jl`

| Test File | Description | API Used |
|-----------|-------------|----------|
| `test/basic.jl` | Basic SPICE parsing and DC/transient | `sp"""..."""` macro, `MNACircuit`, `dc!`, `tran!` |
| `test/transients.jl` | Transient analysis tests | `MNACircuit`, `tran!`, various solvers |
| `test/sweep.jl` | Parameter sweeps | `CircuitSweep`, `ProductSweep`, `dc!`, `tran!` |
| `test/params.jl` | Parameter handling | `MNACircuit`, `ParamLens`, `alter()` |
| `test/mna/core.jl` | Low-level MNA operations | `MNAContext`, `stamp!`, `assemble!`, `solve_dc` |
| `test/mna/va.jl` | VA integration | `va"""..."""` macro, `make_mna_module()` |
| `test/mna/vadistiller_integration.jl` | Complex VA models | VA parsing, PSP103 integration |
| `test/mna/audio_integration.jl` | BJT amplifier tests | `va"""..."""`, `parse_spice_to_mna()`, `CircuitSweep` |
| `test/testpdk/pdk_test.jl` | PDK loading | `load_mna_modules()`, `load_mna_va_module()` |

### 1.3 Benchmark Entry Points

**VACASK Benchmarks** (`benchmarks/vacask/`):

| Benchmark | Circuit | Solvers Tested |
|-----------|---------|----------------|
| RC Circuit | Simple RC | IDA, FBDF, Rodas5P |
| Graetz Bridge | Diode bridge | IDA, FBDF, Rodas5P |
| Voltage Multiplier | Diode multiplier | IDA, FBDF, Rodas5P |
| Ring Oscillator | 9-stage ring with PSP103 | IDA, FBDF, Rodas5P |
| C6288 Multiplier | Large digital circuit | IDA, FBDF, Rodas5P |

Each benchmark uses:
- `parse_spice_to_mna()` with `imported_hdl_modules` for VA models
- `MNACircuit()` wrapper
- `tran!()` for transient analysis
- `MNA.assemble!()` for circuit preparation

---

## Part 2: Categorization - API Chains and Feature Support

### 2.1 Circuit Definition APIs

#### **API Chain 1: Direct MNA Context (Low-Level)**

**Entry Point**: `MNAContext()`, `get_node!()`, `stamp!()`, `assemble!()`

```julia
ctx = MNAContext()
vcc = get_node!(ctx, :vcc)
stamp!(VoltageSource(5.0), ctx, vcc, 0)
stamp!(Resistor(1000.0), ctx, vcc, out)
sys = assemble!(ctx)
sol = solve_dc(sys)
```

| Feature | Supported |
|---------|-----------|
| Verilog-A models | ✅ Via `stamp!(device, ctx, ...)` |
| Initial conditions | ❌ Manual only |
| Parameterization | ❌ Must handle manually |
| Nonlinear devices | ✅ Via `x` parameter |
| Time-varying sources | ✅ Via `t` parameter |
| Nonlinear capacitors | ✅ Via charge stamping |
| SciML configuration | ❌ Not applicable |
| **Usage**: Low-level stamping, internal tests |

#### **API Chain 2: MNACircuit Wrapper (Recommended)**

**Entry Point**: `MNACircuit(builder; params...)`, `dc!()`, `tran!()`

```julia
circuit = MNACircuit(build_function; Vcc=5.0, R=1000.0)
sol_dc = dc!(circuit)
sol_tran = tran!(circuit, (0.0, 1e-3); solver=Rodas5P())
```

| Feature | Supported |
|---------|-----------|
| Verilog-A models | ✅ Via builder function |
| Initial conditions | ✅ `CedarDCOp` algorithm |
| Parameterization | ✅ Via kwargs + `alter()` |
| Nonlinear devices | ✅ Automatic Newton iteration |
| Time-varying sources | ✅ Via `spec.time` |
| Nonlinear capacitors | ✅ Multi-pass detection |
| SciML configuration | ✅ All solver options |
| **Usage**: Main API for all analysis |

#### **API Chain 3: SPICE Parsing**

**Entry Point**: `sp"""..."""` macro, `parse_spice_to_mna()`, `make_mna_circuit()`

```julia
# Macro form (returns MNACircuit)
circuit = sp"""
V1 vcc 0 5.0
R1 vcc out 1k
R2 out 0 1k
"""

# Parse form (returns builder function code)
code = parse_spice_to_mna(spice_text; circuit_name=:my_circuit)
eval(code)
circuit = MNACircuit(my_circuit; params...)
```

| Feature | Supported |
|---------|-----------|
| Verilog-A models | ✅ Via `imported_hdl_modules` |
| Initial conditions | ✅ Via MNACircuit |
| Parameterization | ✅ Via `.param` + kwargs |
| Nonlinear devices | ✅ BSIM, PSP103, etc. |
| Time-varying sources | ✅ SIN, PWL, PULSE, etc. |
| Nonlinear capacitors | ✅ Via MNACircuit detection |
| SciML configuration | ✅ All solver options |
| **Usage**: Loading SPICE netlists |

#### **API Chain 4: Verilog-A Parsing**

**Entry Point**: `va"""..."""` macro, `make_mna_module()`, `load_mna_va_module()`

```julia
# Macro form (creates module with device)
va"""
module resistor(p, n);
    parameter real R = 1000.0;
    inout p, n;
    electrical p, n;
    analog I(p,n) <+ V(p,n)/R;
endmodule
"""
# Creates resistor_module with resistor device

# File loading form
const device_mod = load_mna_va_module(@__MODULE__, "device.va")
```

| Feature | Supported |
|---------|-----------|
| VA contributions | ✅ `I() <+`, `V() <+` |
| Parameters | ✅ Via device kwargs |
| DDT (charges) | ✅ Via `va_ddt()` |
| Nonlinear devices | ✅ Full support |
| Branch aliasing | ⚠️ Partial (no named branches) |
| **Usage**: Custom device models |

#### **API Chain 5: Parameter Sweeps**

**Entry Point**: `CircuitSweep`, `ProductSweep`, `TandemSweep`, `SerialSweep`

```julia
sweep = ProductSweep(R1 = 100:100:1000, R2 = [500, 1000])
cs = CircuitSweep(builder, sweep; default_params...)
solutions = dc!(cs)
solutions = tran!(cs, tspan)
```

| Feature | Supported |
|---------|-----------|
| Product sweeps | ✅ Cartesian product |
| Tandem sweeps | ✅ Parallel iteration |
| Serial sweeps | ✅ Sequential iteration |
| Nested parameters | ✅ Via `var"a.b"` strings |
| All analysis types | ✅ dc!, tran! |
| **Usage**: Parameter exploration |

### 2.2 Analysis Function APIs

#### **DC Analysis**

| Function | Input | Features |
|----------|-------|----------|
| `solve_dc(sys::MNASystem)` | Assembled system | Linear only, no iteration |
| `solve_dc(builder, params, spec)` | Builder function | Newton iteration for nonlinear |
| `solve_dc(ctx::MNAContext)` | Context | Assemble + linear solve |
| `dc!(circuit::MNACircuit)` | Circuit wrapper | Full support, recommended |
| `dc!(cs::CircuitSweep)` | Sweep | Returns vector of solutions |

#### **Transient Analysis**

| Function | Input | Solver Types |
|----------|-------|--------------|
| `tran!(circuit, tspan)` | MNACircuit | Auto-dispatch DAE/ODE |
| `DAEProblem(circuit, tspan)` | MNACircuit | IDA, DFBDF, DABDF2 |
| `ODEProblem(circuit, tspan)` | MNACircuit | Rodas5P, QNDF, FBDF |
| `tran!(cs, tspan)` | CircuitSweep | Vector of solutions |

**Solver Feature Matrix:**

| Feature | IDA (DAE) | DFBDF (DAE) | Rodas5P (ODE) | FBDF (ODE) |
|---------|-----------|-------------|---------------|------------|
| Voltage-dependent C | ✅ | ✅ | ⚠️ Fixed at DC | ⚠️ Fixed at DC |
| Explicit Jacobian | ✅ | ❌ | ✅ | ✅ |
| Sparse matrices | ✅ | ✅ | ✅ | ✅ |
| Stiff systems | ✅ | ✅ | ✅ | ✅ |
| Speed (relative) | 1.0x | ~0.7x | ~1.5-2x | ~1.2x |
| Memory | Medium | Medium | Lower | Lower |

**Note:** ODE solvers use constant mass matrix (C evaluated at DC point), so voltage-dependent capacitors are not accurate during transient.

#### **AC Analysis**

| Function | Input | Features |
|----------|-------|----------|
| `solve_ac(sys, freqs)` | MNASystem + freq vector | Linear, no iteration |
| `solve_ac(sys; fstart, fstop, ppd)` | MNASystem + range | Log-spaced frequencies |

**Not yet in MNACircuit wrapper** - must use assembled system.

### 2.3 Initialization Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| `CedarDCOp()` | DC solve in :dcop mode + BrownFullBasicInit | Default for DAE, handles V-dependent C |
| `BrownFullBasicInit()` | SciML default DAE init | Ring oscillators (no equilibrium) |
| `NoInit()` | Skip initialization | ODE solvers (u0 already valid) |

### 2.4 PDK and Device Loading

| Function | Purpose | Output |
|----------|---------|--------|
| `load_mna_modules(mod, path)` | Load SPICE PDK | Dict of corner modules |
| `load_mna_pdk(path; section)` | Load single section | Module expression |
| `load_mna_va_module(mod, path)` | Load VA device | Device module |

### 2.5 Unused/Deprecated APIs

| API | Status | Reason |
|-----|--------|--------|
| `CircuitIRODESystem` | Deprecated | DAECompiler legacy, replaced by MNACircuit |
| `ParamSim` | Deprecated | Replaced by MNACircuit |
| `solve_mna_spice_code()` | Active but internal | Convenience wrapper |
| `solve_mna_spectre_code()` | Active but internal | Convenience wrapper |
| `make_ode_function(sys)` | Active | Low-level, use ODEProblem(circuit) |
| `make_dae_function(sys)` | Active | Low-level, use DAEProblem(circuit) |
| `PrecompiledCircuit` | Active | Internal optimization, auto-used by MNACircuit |

---

## Part 3: Unification - Proposed API Design

### 3.1 Design Goals

1. **Single Entry Point**: One way to define circuits that works everywhere
2. **Consistent Analysis API**: Same functions for all circuit types
3. **Full Feature Support**: All features available on all solvers
4. **Easy Configuration**: Sensible defaults with clear override paths
5. **Backward Compatibility**: Existing tests should work with minimal changes

### 3.2 Proposed Unified API

#### **Circuit Definition: `Circuit` (renamed from `MNACircuit`)**

```julia
# From builder function
circuit = Circuit(build_function; Vcc=5.0, R=1000.0)

# From SPICE text
circuit = Circuit(sp"""...""")

# From SPICE file
circuit = Circuit(load_spice("circuit.sp"; imported_hdl_modules=[...]))

# From Verilog-A (creates device, not circuit)
device = Device(va"""...""")
```

**Key changes:**
- Rename `MNACircuit` → `Circuit` (cleaner name)
- Add `Circuit(::SpiceAST)` constructor for direct SPICE use
- Deprecate `sp"""..."""` returning circuit, make it return builder only

#### **Analysis Functions: Unified Interface**

```julia
# DC Analysis
sol = dc(circuit)                    # Basic DC
sol = dc(circuit; maxiters=200)      # With options

# Transient Analysis
sol = tran(circuit, tspan)           # Auto-select solver
sol = tran(circuit, tspan; solver=Rodas5P())  # Explicit solver
sol = tran(circuit, tspan; dt=1e-9)  # With fixed timestep

# AC Analysis (NEW: add to Circuit wrapper)
sol = ac(circuit, freqs)             # Frequency vector
sol = ac(circuit; fstart=1, fstop=1e9, ppd=10)  # Range

# Sweeps
sols = dc(sweep)
sols = tran(sweep, tspan)
sols = ac(sweep, freqs)
```

**Key changes:**
- Remove `!` suffix (not mutating in a meaningful way)
- Add `ac()` to Circuit wrapper
- Consistent signature: `analysis(circuit, ...; options...)`

#### **Solver Configuration: `SolverSpec`**

```julia
# Default (auto-selects based on circuit)
sol = tran(circuit, tspan)

# DAE solvers for nonlinear capacitors
sol = tran(circuit, tspan; solver=DAE())        # IDA with defaults
sol = tran(circuit, tspan; solver=DAE(:IDA))    # Explicit IDA
sol = tran(circuit, tspan; solver=DAE(:DFBDF))  # DFBDF

# ODE solvers for speed (fixed C)
sol = tran(circuit, tspan; solver=ODE())        # Rodas5P default
sol = tran(circuit, tspan; solver=ODE(:FBDF))   # FBDF

# Full control
sol = tran(circuit, tspan;
    solver=DAE(:IDA),
    abstol=1e-12,
    reltol=1e-10,
    dtmax=1e-9,
    maxiters=1_000_000
)
```

#### **Initial Conditions: `InitSpec`**

```julia
# Default (CedarDCOp)
sol = tran(circuit, tspan)

# Custom IC
sol = tran(circuit, tspan; u0=custom_u0)

# Oscillators (no stable DC)
sol = tran(circuit, tspan; init=:relaxed)

# Skip init (already consistent)
sol = tran(circuit, tspan; init=:none)
```

#### **Solution Access: Unified `Solution` Type**

```julia
# All analysis types return Solution
sol = dc(circuit)
sol = tran(circuit, tspan)
sol = ac(circuit, freqs)

# Access voltages
voltage(sol, :out)           # DC: scalar, Tran: at all times, AC: at all freqs
voltage(sol, :out, 1e-3)     # Tran: at specific time
voltage(sol, :out, 1e6)      # AC: at specific frequency

# Access currents
current(sol, :I_V1)

# Time/frequency arrays
sol.t                        # Tran: time points
sol.f                        # AC: frequency points

# Raw data
sol.u                        # State vector(s)
```

### 3.3 Migration Plan

#### **Phase 1: API Aliasing (Non-Breaking)**

1. Create `Circuit` as alias for `MNACircuit`
2. Create `dc`, `tran`, `ac` as wrappers for `dc!`, `tran!`, `solve_ac`
3. Add deprecation warnings to old names
4. Add `ac()` method to Circuit wrapper

**Files to modify:**
- `src/mna/solve.jl`: Add `ac()` for MNACircuit
- `src/sweeps.jl`: Create non-mutating aliases
- `src/CedarSim.jl`: Export new names

#### **Phase 2: Test Migration**

Convert tests to new API one file at a time:

| File | Changes Needed |
|------|----------------|
| `test/basic.jl` | `dc!` → `dc`, `tran!` → `tran` |
| `test/transients.jl` | `dc!` → `dc`, `tran!` → `tran` |
| `test/sweep.jl` | `dc!` → `dc`, `tran!` → `tran` |
| `test/params.jl` | `MNACircuit` → `Circuit` |
| `test/mna/core.jl` | Keep low-level (internal tests) |
| `test/mna/va.jl` | Keep as-is (VA-specific) |
| `test/mna/audio_integration.jl` | `tran!` → `tran` |
| `test/testpdk/pdk_test.jl` | Keep as-is (PDK-specific) |

#### **Phase 3: Deprecation Removal**

1. Remove `dc!`, `tran!` (replaced by `dc`, `tran`)
2. Rename `MNACircuit` → `Circuit` (keep alias for 1 version)
3. Update all documentation

#### **Phase 4: Feature Parity**

1. Add `ac()` to Circuit wrapper (currently only on MNASystem)
2. Add solver auto-selection based on circuit analysis
3. Add `SolverSpec` abstraction for cleaner configuration

### 3.4 APIs to Deprecate/Remove

| API | Action | Replacement |
|-----|--------|-------------|
| `dc!(circuit)` | Deprecate | `dc(circuit)` |
| `tran!(circuit, tspan)` | Deprecate | `tran(circuit, tspan)` |
| `solve_mna_spice_code()` | Keep internal | N/A |
| `solve_mna_spectre_code()` | Keep internal | N/A |
| `make_ode_function(sys)` | Keep internal | Auto-used by ODEProblem |
| `make_dae_function(sys)` | Keep internal | Auto-used by DAEProblem |

### 3.5 Estimated Effort

| Phase | Effort | Risk |
|-------|--------|------|
| Phase 1: Aliasing | 1-2 days | Low |
| Phase 2: Test Migration | 2-3 days | Low |
| Phase 3: Deprecation Removal | 1 day | Medium |
| Phase 4: Feature Parity | 3-5 days | Medium |

**Total: ~1-2 weeks**

---

## Appendix A: Complete API Inventory

### Exported from `CedarSim`

**Circuit Definition:**
- `sp"""..."""` - SPICE string macro
- `va"""..."""` - Verilog-A string macro
- `parse_spice_to_mna()` - Parse SPICE to builder code
- `make_mna_circuit()` - Create builder from AST
- `load_mna_modules()` - Load SPICE PDK
- `load_mna_pdk()` - Load single PDK section
- `load_mna_va_module()` - Load VA device

**Parameters:**
- `ParamLens` - Hierarchical parameter access
- `IdentityLens` - Default parameter handler
- `ParamObserver` - Parameter observation

**Sweeps:**
- `Sweep` - Single parameter sweep
- `ProductSweep` - Cartesian product
- `TandemSweep` - Parallel iteration
- `SerialSweep` - Sequential iteration
- `CircuitSweep` - Circuit parameter sweep
- `sweepvars()` - Get sweep variables
- `split_axes()` - Split product sweep

**Analysis:**
- `dc!()` - DC operating point
- `tran!()` - Transient analysis
- `alter()` - Modify parameters

### Exported from `CedarSim.MNA`

**Context:**
- `MNAContext` - Circuit builder context
- `MNASpec` - Simulation specification
- `MNASystem` - Assembled system
- `MNACircuit` - Circuit wrapper
- `MNASolutionAccessor` - Solution access

**Nodes:**
- `get_node!()` - Get/create node
- `get_current_idx!()` - Get/create current variable
- `get_charge_idx!()` - Get/create charge variable

**Devices:**
- `Resistor`, `Capacitor`, `Inductor`
- `VoltageSource`, `CurrentSource`
- `VCVS`, `CCCS`, `VCCS`, `CCVS`
- `Diode`, `MOSFET`
- Time-dependent: `SinSource`, `PWLSource`, `PulseSource`

**Stamping:**
- `stamp!()` - Stamp device into context
- `reset_for_restamping!()` - Clear for rebuild

**Assembly:**
- `assemble!()` - Build MNASystem
- `system_size()` - Get system dimension

**Analysis:**
- `solve_dc()` - DC solution (multiple signatures)
- `solve_ac()` - AC analysis
- `DCSolution`, `ACSolution` - Result types
- `voltage()`, `current()` - Access results

**SciML Integration:**
- `DAEProblem()` - Create DAE problem
- `ODEProblem()` - Create ODE problem

**Compilation:**
- `compile()` - Precompile circuit
- `PrecompiledCircuit` - Compiled result
- `EvalWorkspace` - Evaluation workspace

---

## Appendix B: Test Coverage by Feature

| Feature | Test Files |
|---------|------------|
| Basic DC | `basic.jl`, `mna/core.jl` |
| Transient | `transients.jl`, `mna/audio_integration.jl` |
| Parameter sweeps | `sweep.jl`, `mna/audio_integration.jl` |
| ParamLens | `params.jl` |
| SPICE parsing | `basic.jl`, `testpdk/pdk_test.jl` |
| VA integration | `mna/va.jl`, `mna/vadistiller_integration.jl` |
| PDK loading | `testpdk/pdk_test.jl` |
| Nonlinear devices | `mna/va.jl`, benchmarks |
| Time-varying sources | `mna/audio_integration.jl`, `transients.jl` |
| Multiple solvers | `transients.jl`, benchmarks |

---

## Appendix C: Benchmark Reference

**VACASK Benchmarks (from `benchmarks/vacask/run_benchmarks.jl`):**

All benchmarks use the pattern:
```julia
va = VerilogAParser.parsefile(va_path)
Core.eval(@__MODULE__, CedarSim.make_mna_module(va))

circuit_code = parse_spice_to_mna(spice_code;
    circuit_name=:circuit_name,
    imported_hdl_modules=[VA_module])
eval(circuit_code)

circuit = MNACircuit(circuit_name)
MNA.assemble!(circuit)
sol = tran!(circuit, tspan; solver=solver, dense=false)
```

**Solver comparison (typical results):**
- IDA: Most robust, medium speed
- FBDF: Fast for stiff systems
- Rodas5P: Fastest overall, good for most circuits
