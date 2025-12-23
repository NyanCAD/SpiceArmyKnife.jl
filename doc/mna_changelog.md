# MNA Migration Changelog

This document tracks progress on the MNA (Modified Nodal Analysis) migration as defined in `mna_design.md`.

---

## Phase Overview

| Phase | Description | Status | LOC Target |
|-------|-------------|--------|------------|
| 0 | Dependency Cleanup | **Complete** | - |
| 1 | MNA Core | **Complete** | ~200 |
| 2 | Simple Device Stamps | **Complete** (merged into Phase 1) | ~200 |
| 3 | DC & Transient Solvers | **Complete** (merged into Phase 1) | ~300 |
| 4 | SPICE Codegen | **In Progress** | ~300 |
| 5 | VA Contribution Functions | **Core Complete** | ~400 |
| 6 | Complex VA & DAE | Not Started | ~400 |
| 7 | Advanced Features | Not Started | ~300 |
| 8 | Cleanup | Not Started | - |

---

## Phase 0: Dependency Cleanup

**Status:** Complete
**Date:** 2024-12-21
**Branch:** `claude/implement-mna-phase-0-8mGFW`

### Goal
Clean baseline on Julia 1.12 with minimal dependencies. Simulation won't work yet, but parsing/codegen should.

### Changes Made

#### New Files
- `src/mna/stubs.jl` - DAECompiler stub implementations
  - Placeholder types: `Scope`, `GenScope`, `equation`, `StubVariable`, `IRODESystem`
  - Stub functions: `variable()`, `equation!()`, `ddt()`, `sim_time()`, etc.
  - Returns placeholder values for codegen testing

#### Modified Files

**CedarSim.jl**
- Added `USE_DAECOMPILER = false` flag for conditional loading
- Include stubs module when DAECompiler disabled
- Skip `dcop.jl` and `ac.jl` when disabled (simulation code)

**Project.toml**
- Commented out DAECompiler dependency
- Removed test/docs/benchmarks from workspace (CedarEDA-specific deps)
- Added local source paths for: SpectreNetlistParser, VerilogAParser, Lexers
- Removed CedarEDA fork references from sources

**VerilogAParser.jl/Project.toml**
- Commented out CMC, SciMLBase fork, SciMLSensitivity fork
- Removed workspace with CedarEDA test deps

**spectre.jl**
- Updated to use `NumberLiteral` (parser API changed)
  - Removed references to: `FloatLiteral`, `IntLiteral`, `NumericValue`
- Updated to use `ControlledSource{:V,:V}` etc. (parameterized types)
  - Removed references to: `VCVS`, `VCCS`, `CCVS`, `CCCS`
- Fixed Spectre number parsing for units like "23pf" → 23e-12
- Removed duplicate `LineNumberNode` method definition

**Other guarded files** (added `@static if USE_DAECOMPILER` blocks):
- `simulate_ir.jl`, `util.jl`, `vasim.jl`, `simpledevices.jl`
- `circuitodesystem.jl`, `spectre_env.jl`, `sweeps.jl`
- `netlist_utils.jl`, `deprecated.jl`, `circsummary.jl`, `va_env.jl`

### Exit Criteria Status

| Criterion | Status |
|-----------|--------|
| Clean baseline on Julia 1.12 | ✅ |
| Minimal dependencies | ✅ |
| Parsing works | ✅ |
| Codegen works | ✅ |
| DAECompiler compile-time deps commented out | ✅ |
| Simulation disabled (expected) | ✅ |

### What Works
- Package loads and precompiles on Julia 1.12
- Spectre/SPICE netlist parsing
- Verilog-A parsing
- Circuit code generation (produces valid Julia code)

### What Doesn't Work (Expected)
- Actual circuit simulation (requires Phase 1+ MNA implementation)
- Tests that eval generated circuits or call `solve_circuit`
- PDK-dependent tests (BSIM4, Sky130PDK, etc.)

### Test Infrastructure Changes

**test/runtests.jl**
- Added `PHASE0_MINIMAL` flag for conditional test execution
- Phase 0 runs only: `spectre_expr.jl`, `sweep.jl`
- Full test suite runs when all dependencies are available

**test/common.jl**
- Conditional `DAECompiler` import with `HAS_DAECOMPILER` flag
- Stub implementations of `solve_circuit`, `solve_spice_*`, `solve_spectre_*`
- `sim_time` falls back to stubs when DAECompiler unavailable

**test/spectre_expr.jl**
- Added `HAS_SIMULATION` guards around `eval(fn)` and variable tests
- Parsing and codegen tests work without simulation

**test/sweep.jl**
- Added `HAS_SIMULATION` guards around `dc!` tests
- Sweep data structure tests work without simulation

**test/Project.toml**
- Removed heavy dependencies: BSIM4, Sky130PDK, GF180MCUPDK, etc.
- Minimal deps for parsing tests only

**.github/workflows/ci.yml**
- Updated to run `Pkg.test("CedarSim")` instead of smoke tests
- Develops local packages in test environment

### Notes
- Stubs return placeholder values to allow codegen testing
- `equation` struct is both a type and callable (matches DAECompiler API)
- Phase 0 tests validate parsing and code generation only

---

## Phase 1: MNA Core (includes Phases 1-3 from design doc)

**Status:** Complete
**Date:** 2024-12-21
**Branch:** `claude/implement-mna-engine-pOTJ7`
**LOC:** ~1640 (implementation) + ~600 (tests)

### Goal
Complete MNA infrastructure: context, stamping, devices, and solvers.
Phases 1-3 from the design doc were merged into a single implementation
since they share common data structures and are tightly coupled.

### Implementation Summary

The MNA module provides a standalone circuit simulation engine that:
- Assembles circuits via stamping (like classical SPICE)
- Supports DC, AC, and transient analysis
- Integrates with DifferentialEquations.jl for time-domain simulation

### New Files

#### `src/mna/MNA.jl` (58 LOC)
Module wrapper that includes all MNA components.

#### `src/mna/context.jl` (385 LOC)
Core MNA context and stamping primitives:

```julia
mutable struct MNAContext
    node_names::Vector{Symbol}      # Node name registry
    node_to_idx::Dict{Symbol,Int}   # Name → index mapping
    n_nodes::Int                    # Number of voltage nodes
    current_names::Vector{Symbol}   # Current variable names
    n_currents::Int                 # Number of current variables
    G_I, G_J, G_V::Vector           # COO format for G matrix
    C_I, C_J, C_V::Vector           # COO format for C matrix
    b::Vector{Float64}              # RHS vector
    finalized::Bool
end
```

Key functions:
- `get_node!(ctx, name)` - Allocate/lookup node index (0 = ground)
- `alloc_current!(ctx, name)` - Allocate current variable for V-sources/inductors
- `stamp_G!(ctx, i, j, val)` - Stamp into G matrix (conductance)
- `stamp_C!(ctx, i, j, val)` - Stamp into C matrix (capacitance)
- `stamp_b!(ctx, i, val)` - Stamp into RHS vector (sources)
- `stamp_conductance!(ctx, p, n, G)` - 2-terminal conductance pattern
- `stamp_capacitance!(ctx, p, n, C)` - 2-terminal capacitance pattern

#### `src/mna/build.jl` (269 LOC)
Sparse matrix assembly from COO format:

```julia
struct MNASystem{T<:Real}
    G::SparseMatrixCSC{T,Int}       # Conductance matrix
    C::SparseMatrixCSC{T,Int}       # Capacitance matrix
    b::Vector{T}                     # RHS vector
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    n_nodes::Int
    n_currents::Int
end
```

Key functions:
- `assemble!(ctx)` - Build MNASystem from context
- `assemble_G(ctx)`, `assemble_C(ctx)` - Individual matrix assembly
- `get_rhs(ctx)` - Get properly-sized RHS vector
- Matrix visualization utilities for debugging

#### `src/mna/devices.jl` (502 LOC)
Device types and stamp! methods:

| Device | Type | Stamps | Current Variable? |
|--------|------|--------|-------------------|
| Resistor | `Resistor(r)` | G only | No |
| Capacitor | `Capacitor(c)` | C only | No |
| Inductor | `Inductor(l)` | G + C | Yes (I_L is state) |
| Voltage Source | `VoltageSource(v)` | G + b | Yes (I unknown) |
| Current Source | `CurrentSource(i)` | b only | No |
| VCVS | `VCVS(gain)` | G | Yes |
| VCCS | `VCCS(gm)` | G | No |
| CCVS | `CCVS(rm)` | G | Yes (×2) |
| CCCS | `CCCS(gain)` | G | Yes |

All devices follow the MNA stamping conventions:
- Ground is node 0 (implicit, not in matrices)
- Current leaving node is positive (sign convention)
- Voltage sources and inductors need current variables

#### `src/mna/solve.jl` (429 LOC)
Analysis solvers:

**DC Analysis:**
```julia
sol = solve_dc(sys)           # Solve G*x = b
V = voltage(sol, :node_name)  # Access by name
I = current(sol, :I_V1)       # Access current variables
```

**AC Analysis:**
```julia
ac = solve_ac(sys, [1e3, 1e4, 1e5])  # Specify frequencies
ac = solve_ac(sys; fstart=1, fstop=1e6, points_per_decade=10)
Vout = voltage(ac, :out)      # Complex voltage at all freqs
mag = magnitude_db(ac, :out)  # dB magnitude
```

**Transient (ODEProblem setup):**
```julia
prob = make_ode_problem(sys, (0.0, 1e-3))
# Returns NamedTuple with:
#   f, u0, tspan, mass_matrix, jac, jac_prototype
# Use with OrdinaryDiffEq:
# f = ODEFunction(prob.f; mass_matrix=prob.mass_matrix, ...)
# ode = ODEProblem(f, prob.u0, prob.tspan)
# sol = solve(ode, Rodas5())
```

### Test Coverage

#### `test/mna/core.jl` (606 LOC)
Comprehensive test suite covering:

1. **Context and Node Allocation**
   - Node creation and lookup
   - Ground node handling
   - Current variable allocation

2. **Stamping Primitives**
   - G, C, b stamping
   - Ground entry skipping
   - Value accumulation

3. **Matrix Assembly**
   - COO to CSC conversion
   - Correct matrix structure

4. **Device Stamps**
   - Each device type validated against textbook patterns
   - Resistor, Capacitor, Inductor stamps
   - Voltage/Current source stamps
   - Controlled source stamps (VCVS, VCCS, CCVS, CCCS)

5. **DC Analysis (Analytical Validation)**
   - Voltage divider (equal and unequal)
   - Current source into resistor
   - Two voltage sources
   - VCCS amplifier
   - VCVS inverting amplifier
   - Multi-node star network (superposition)

6. **AC Analysis**
   - RC low-pass filter (cutoff frequency, 3dB point)
   - RL high-pass filter

7. **Transient Setup**
   - ODE problem creation
   - Mass matrix and Jacobian

8. **Edge Cases**
   - Empty context
   - Single floating node
   - Very large/small resistances

### Modified Files

**CedarSim.jl**
- Added MNA module include and export
- `include("mna/MNA.jl")` after stubs
- `export MNA`

**test/runtests.jl**
- Added MNA tests to Phase 0/1 test section
- `using CedarSim` for MNA access
- `include("mna/core.jl")` in Phase 1 testset

### Exit Criteria

| Criterion | Status |
|-----------|--------|
| MNAContext tracks nodes and currents | ✅ |
| stamp_G!/stamp_C!/stamp_b! work correctly | ✅ |
| assemble_* produces correct sparse matrices | ✅ |
| All basic devices implemented (R, C, L, V, I) | ✅ |
| Controlled sources implemented (VCVS, VCCS, CCVS, CCCS) | ✅ |
| DC solver works with analytical validation | ✅ |
| AC solver works with filter validation | ✅ |
| Transient ODEProblem setup ready | ✅ |
| Comprehensive unit tests | ✅ |

### Usage Example

```julia
using CedarSim.MNA

# Create voltage divider
ctx = MNAContext()
vcc = get_node!(ctx, :vcc)
out = get_node!(ctx, :out)

stamp!(VoltageSource(5.0), ctx, vcc, 0)
stamp!(Resistor(1000.0), ctx, vcc, out)
stamp!(Resistor(1000.0), ctx, out, 0)

# DC analysis
sol = solve_dc(ctx)
@assert voltage(sol, :out) ≈ 2.5

# AC analysis
sys = assemble!(ctx)
ac = solve_ac(sys; fstart=100, fstop=1e6)
```

### Architecture Notes

The implementation follows patterns from VACASK (modern C++ MNA):
- Two-phase stamping: topology first, then values
- COO format for incremental assembly
- Separate G (resistive) and C (reactive) matrices
- Current variables only for voltage-defined elements

Integration with SciML follows SciMLBase.jl patterns:
- Mass matrix ODE formulation: `C * dx/dt = b - G*x`
- Jacobian is constant: `-G`
- Compatible with stiff solvers (Rodas5, QNDF, etc.)

### Bug Fixes (2024-12-21)

After initial implementation, testing with Julia 1.12 revealed several issues:

1. **Missing Dependencies**
   - Added `Printf` and `SparseArrays` to Project.toml

2. **Missing Exports**
   - Added `stamp_conductance!`, `stamp_capacitance!` to context.jl exports
   - Added `voltage`, `current` helper functions to solve.jl exports

3. **VCCS Sign Convention Error**
   - Original: stamps `+gm` into `G[out_p, in_p]` (incorrect)
   - Fixed: stamps `-gm` into `G[out_p, in_p]` (current INTO out_p = negative)
   - MNA convention: "current leaving node is positive"
   - When I = gm * V(in) flows into out_p, it contributes -gm to G[out_p, :]

4. **Test Import Conflicts**
   - Changed from `using CedarSim` to explicit imports from `CedarSim.MNA`
   - Avoids conflicts between CedarSim.VoltageSource and MNA.VoltageSource

5. **RL High-Pass Filter Test**
   - Fixed test circuit topology (R in series, L to ground)
   - Original test had incorrect topology (L in series, R to ground)

All 92 MNA tests now pass on Julia 1.12.

### Transient Analysis Enhancement (2024-12-21)

Added comprehensive transient simulation with actual ODE/DAE solving:

#### New Functions
- `make_dae_function(sys)` - Create implicit DAE residual F(du,u,p,t)=0
- `make_dae_problem(sys, tspan)` - Set up DAEProblem with consistent ICs

#### Transient Tests Added (~320 LOC, 5 testsets)
1. **RC Charging (Mass Matrix ODE)** - Validates exponential charging V(t)=Vcc*(1-exp(-t/τ))
2. **RC Charging (Implicit DAE)** - Same circuit solved with Sundials IDA
3. **RL Circuit** - Inductor current rise I(t)=I_ss*(1-exp(-t/τ))
4. **RLC Oscillator** - Underdamped 2nd-order system demonstrating overshoot
5. **ODE vs DAE Comparison** - Verifies both methods produce identical results

All tests validate against analytical solutions for first-order RC/RL response
and second-order RLC oscillation. Total: **141 tests passing**.

### Parameterization & Mode Switching (2024-12-21)

Added ParamSim-style parameterization wrapper and mode switching:

#### New Types
- `MNASim{F,P}` - Parameterized simulation wrapper (similar to ParamSim)
- `TimeDependentVoltageSource{F}` - Mode-aware voltage source with time function
- `PWLVoltageSource` - Piecewise-linear voltage source

#### New Functions
- `alter(sim; kwargs...)` - Create new sim with modified parameters
- `with_mode(sim, mode)` - Create new sim with different mode (:tran, :dcop, :tranop)
- `get_source_value(src, t, mode)` - Get mode-aware source value
- `pwl_value(src, t)` - Evaluate PWL source at time t
- `make_dc_initialized_ode_problem(sys, tspan)` - ODE problem with DC steady-state ICs
- `make_dc_initialized_dae_problem(sys, tspan)` - DAE problem with DC steady-state ICs

#### MNASim Usage
```julia
# Define parameterized circuit builder
function build_rc(p)
    ctx = MNAContext()
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # p.mode is automatically included in params
    voltage = p.mode == :dcop ? p.Vdc : p.Vss
    stamp!(VoltageSource(voltage), ctx, vcc, 0)
    stamp!(Resistor(p.R), ctx, vcc, out)
    stamp!(Capacitor(p.C), ctx, out, 0)
    return ctx
end

# Create parameterized simulation
sim = MNASim(build_rc; Vdc=0.0, Vss=5.0, R=1000.0, C=1e-6)

# Alter parameters
sim2 = alter(sim; R=2000.0)

# Switch mode
sim_dcop = with_mode(sim, :dcop)

# Solve
sol = solve_dc(sim)
```

#### Mode-Aware Initialization
The typical SPICE initialization flow is now supported:
1. Set mode to `:dcop`, solve for DC operating point
2. Switch to `:tran`, use DC solution as initial conditions
3. Run transient with time-dependent sources

#### Tests Added (~220 LOC, 3 testsets)
1. **Parameterized circuit with mode switching** - Tests `alter()`, `with_mode()`, chained alterations
2. **Time-dependent source with mode** - Tests `TimeDependentVoltageSource`, `PWLVoltageSource`
3. **Mode-aware parameterized simulation** - Tests full SPICE-style init flow (DC → transient)

### Architecture Refactor (2024-12-22)

Major refactor based on analysis of CedarSim patterns, OpenVAF/VACASK interfaces, and GPU requirements.

#### Design Decisions (see `doc/mna_architecture.md`)

1. **Out-of-Place Evaluation**: Circuit evaluation returns new matrices for GPU compatibility
   - Enables `ODEProblem{false}` formulation for DiffEqGPU.jl
   - Enables ensemble GPU solving for parameter sweeps
   - No mutation means no reset needed

2. **Explicit Parameter Passing**: Parameters via function arguments, not ScopedValue
   - DAECompiler's ScopedValue optimization not available in plain Julia
   - Avoids Julia's closure boxing bug
   - Enables full JIT optimization

3. **Separated Spec from Params**:
   - `MNASpec`: Simulation specification (temp, mode)
   - `params`: Circuit parameters via ParamLens pattern
   - Builder signature: `(params, spec) -> MNAContext`

#### New Types and Functions

```julia
# Simulation specification
struct MNASpec
    temp::Float64  # Temperature (Celsius)
    mode::Symbol   # :dcop, :tran, :tranop, :ac
end

# MNASim now separates spec from params
struct MNASim{F,S,P}
    builder::F
    spec::S      # MNASpec
    params::P    # Circuit parameters
end

# Out-of-place evaluation (GPU-compatible)
eval_circuit(builder, params, spec; t=0.0, u=nothing) -> MNASystem
eval_circuit(sim::MNASim; t=0.0, u=nothing) -> MNASystem

# Spec manipulation
with_spec(sim, spec) -> MNASim
with_temp(sim, temp) -> MNASim  # Convenience
with_mode(sim, mode) -> MNASim  # Convenience
```

#### Research Findings

**OpenVAF/OSDI Interface** (`refs/OpenVAF/melange/core/src/veriloga/osdi_0_4.rs`):
- `OsdiSimInfo.abstime`: Time passed every iteration
- `setup_instance(temp, ...)`: Temperature explicit at setup
- `JACOBIAN_ENTRY_*_CONST` flags: Mark constant vs. variable Jacobian entries
- Separation of resist/react for C ABI efficiency (not needed in Julia with JIT)

**VACASK** (`refs/VACASK/`):
- ContextStack for hierarchical parameter resolution
- Device eval called every Newton iteration
- Temperature converted from Celsius to Kelvin for Verilog-A

**GPU Support** (DiffEqGPU.jl):
- Out-of-place required for `EnsembleGPUKernel`
- `CuSparseMatrixCSR` for sparse matrices on GPU
- Iterative solvers via Krylov.jl

### Next Steps

Phase 4 (SPICE Codegen) will:
- Modify `src/spc/*.jl` to emit `stamp!` calls (NOT spectre.jl which is the old backend)
- Connect parsed netlists to MNA stamping
- Enable end-to-end SPICE simulation

### CircuitSweep Integration (2024-12-22)

Ported the high-level sweep API to work with MNA backend.

#### Changes to `src/sweeps.jl`

**Removed DAECompiler-specific code:**
- Removed `IRODESystem` references
- Removed `DAEProblem`, `ParamSim` patterns
- Removed broadcast overrides for `dc!`/`tran!`

**Simplified `CircuitSweep` struct:**
```julia
struct CircuitSweep{T<:Function}
    builder::T       # Builder function (params, spec) -> MNAContext
    sim::MNASim      # Base simulation for alter()
    iterator::SweepLike
end
```

**New methods:**
- `CircuitSweep(builder, sweep; default_params...)` - Create sweep from builder
- `dc!(cs::CircuitSweep)` - DC analysis over entire sweep
- `tran!(cs::CircuitSweep, tspan)` - Transient analysis over entire sweep
- Iteration returns `MNASim` objects via `alter()`

**MNA `alter()` enhanced for nested params:**
- Uses Accessors for lens-based path resolution
- Supports var-strings like `var"inner.R1"` for nested NamedTuples
- Auto-converts numeric types to Float64

#### Test Changes (`test/sweep.jl`)

**Converted to MNA:**
- `build_two_resistor(params, spec)` builder function with defaults via `merge()`
- All CircuitSweep tests use MNA backend
- Nested params test demonstrates ParamLens pattern

**ParamLens demonstration:**
```julia
function build_nested_resistor(params, spec)
    lens = ParamLens(params)
    p = lens.inner(; R1=1000.0, R2=1000.0)  # Defaults merged with overrides
    # ... stamp circuit using p.R1, p.R2 ...
end

sweep = ProductSweep(var"inner.params.R1" = 100.0:200.0, ...)
cs = CircuitSweep(build_nested_resistor, sweep;
                  inner = (params = (R1=100.0, R2=100.0),))
```

**Disabled tests:**
- SPICE codegen sweep test commented out (TODO: re-enable after Phase 4)

#### Tests Passing
- 555 sweep tests (Sweep types, CircuitSweep, dc! sweeps, nested params)
- 246 MNA core tests

---

## Phase 2: Simple Device Stamps

**Status:** Complete (merged into Phase 1)

See Phase 1 for device implementations.

---

## Phase 3: DC & Transient Solvers

**Status:** Complete (merged into Phase 1)

See Phase 1 for solver implementations.

---

## Phase 4: SPICE Codegen

**Status:** In Progress
**Date Started:** 2024-12-22
**Branch:** `claude/integrate-mna-code-generation-yOpfg`
**LOC:** ~350 (implementation) + ~100 (tests)

### Goal
Update spc/codegen.jl to emit stamp! calls instead of DAECompiler primitives.
Connect parsed SPICE netlists to the MNA backend for simulation.

### Implementation Summary

#### Core Changes

**`src/spc/codegen.jl`** (~280 LOC added)

Added MNA-specific codegen functions that transform SPICE AST into MNA stamp! calls:

1. **Device Instance Generation:**
   - `cg_mna_instance!(state, instance)` - Dispatch for each device type
   - Resistor: `stamp!(Resistor(r/m), ctx, p, n)` with multiplicity
   - Capacitor: `stamp!(Capacitor(c*m), ctx, p, n)` with multiplicity
   - Inductor: `stamp!(Inductor(l), ctx, p, n)`
   - Voltage Source: `stamp!(VoltageSource(v), ctx, p, n)`
   - Current Source: `stamp!(CurrentSource(i), ctx, p, n)`
   - VCVS: `stamp!(VCVS(gain), ctx, out_p, out_n, in_p, in_n)`
   - VCCS: `stamp!(VCCS(gm), ctx, out_p, out_n, in_p, in_n)`

2. **Builder Function Generation:**
   - `codegen_mna!(state)` - Generate MNA builder function body
   - `make_mna_circuit(ast; circuit_name)` - Top-level code generation
   - Returns code that creates a function: `(params, spec) -> MNAContext`

3. **Expression Handling:**
   - Reuses existing `cg_expr!` for parameter expressions
   - Handles temperature option: updates spec if `.TEMP` specified
   - Handles parameter dependencies via topological sort

**`src/spc/sema.jl`** (~10 LOC)

Added `sema_nets` methods for controlled sources:
- `sema_nets(SNode{SP.VCVS})` - 4-terminal: (pos, neg, cpos, cneg)
- `sema_nets(SNode{SP.VCCS})` - 4-terminal: (pos, neg, cpos, cneg)
- `sema_nets(SNode{SP.CCVS})` - 2-terminal (current control separate)
- `sema_nets(SNode{SP.CCCS})` - 2-terminal (current control separate)

**`src/spc/interface.jl`** (~60 LOC)

Added high-level MNA functions:
- `parse_spice_to_mna(code; circuit_name)` - Parse and return MNA builder code
- `solve_spice_mna(code; temp)` - Parse, build, and solve DC
- Updated `sp_str` macro to generate MNA builder functions instead of SpCircuit

4. **Subcircuit Support:**
   - `codegen_mna_subcircuit(sema, name)` - Generate subcircuit builder function
   - Subcircuit builders have signature: `name_mna_builder(params, spec, ctx, ports...)`
   - Main `make_mna_circuit` generates all subcircuit builders before main circuit

**`src/CedarSim.jl`**

- Added includes for spc/*.jl files (Phase 4 integration)
- Exported: `make_mna_circuit`, `parse_spice_to_mna`, `solve_spice_mna`

#### Test Changes

**`test/sweep.jl`** (~90 LOC added)

Added MNA SPICE codegen tests:

1. **Basic resistor circuit test:**
   - Voltage divider: `V1 vcc 0 DC 5`, `R1 vcc out 1k`, `R2 out 0 1k`
   - Validates `voltage(sol, :out) ≈ 2.5`

2. **RLC circuit test:**
   - DC analysis of series RLC
   - Validates inductor=short, capacitor=open at DC

3. **Current source test:**
   - `I1 0 out DC 1m`, `R1 out 0 1k`
   - Validates V = IR = 1V

### Generated Code Pattern

The MNA codegen generates builder functions like:

```julia
function circuit(params, spec::MNASpec=MNASpec())
    ctx = MNAContext()

    # Node allocation
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # Parameter handling
    var"*params#" = params isa ParamLens ? getfield(params, :nt) : params
    r1 = getfield(var"*params#", :r1, 1000.0)

    # Device stamps
    stamp!(VoltageSource(5.0; name=:v1), ctx, vcc, 0)
    stamp!(Resistor(r1; name=:r1), ctx, vcc, out)
    stamp!(Resistor(1000.0; name=:r2), ctx, out, 0)

    return ctx
end
```

### Key Differences from DAECompiler Codegen

| Aspect | DAECompiler (old) | MNA (new) |
|--------|-------------------|-----------|
| Output | `Named(spicecall(...))(nets...)` | `stamp!(Device(...), ctx, nodes...)` |
| Node allocation | Implicit via `net()` call | Explicit via `get_node!(ctx, name)` |
| Parameters | Via ScopedValue | Via function argument + ParamLens |
| Result | Circuit function | MNAContext |

### What Works

- Basic device parsing (R, C, L, V, I)
- Controlled sources (E, G)
- Parameter expressions
- Unit suffixes (k, m, u, n, p, f)
- DC analysis of parsed circuits

### What's Pending

- Model statements (`.MODEL`)
- Transient sources (PWL, PULSE, SIN)
- CCVS/CCCS (current-controlled sources - require reference to V source)
- More comprehensive test coverage

### Time-Dependent Sources (PWL/SIN) - 2024-12-22

Added support for time-dependent transient sources (PWL, SIN, PULSE).

#### New Device Types (`src/mna/devices.jl`)

Added ~150 LOC for new time-dependent source types:

```julia
# Sinusoidal Voltage Source
struct SinVoltageSource
    vo::Float64      # DC offset
    va::Float64      # Amplitude
    freq::Float64    # Frequency (Hz)
    td::Float64      # Delay
    theta::Float64   # Damping factor
    phase::Float64   # Phase (degrees)
    name::Symbol
end

# Sinusoidal Current Source
struct SinCurrentSource  # Same fields as SinVoltageSource

# PWL Current Source (complementing existing PWLVoltageSource)
struct PWLCurrentSource
    times::Vector{Float64}
    values::Vector{Float64}
    name::Symbol
end
```

Key functions:
- `sin_value(src, t)` - Evaluate sinusoidal source at time t
- `get_source_value(src, t, mode)` - Mode-aware value (DC vs transient)
- `stamp!(src, ctx, p, n; t=0.0, mode=:dcop)` - Time-parameterized stamping

#### Solver Updates (`src/mna/solve.jl`)

Added ~90 LOC for time-dependent ODE support:

```julia
# MNASpec now includes time
struct MNASpec
    temp::Float64   # Temperature
    mode::Symbol    # :dcop, :tran
    time::Float64   # Current simulation time
end

# Time-dependent ODE function
make_ode_function_timed(builder, params, base_spec)
# Rebuilds b(t) at each timestep by calling builder with updated spec

# Complete ODE problem for time-dependent circuits
make_ode_problem_timed(builder, params, tspan; temp=27.0)
```

#### SPICE Codegen Updates (`src/spc/codegen.jl`)

Updated `cg_mna_instance!` for Voltage/Current sources (~120 LOC):

- **PWL sources:** Parse time-value pairs, emit `PWLVoltageSource`/`PWLCurrentSource`
- **SIN sources:** Parse vo, va, freq, td, theta, phase; emit `SinVoltageSource`/`SinCurrentSource`
- **PULSE sources:** Convert to PWL approximation for first period

Generated code pattern:
```julia
# PWL(1m 0 9m 5)
let vals = [1e-3, 0.0, 9e-3, 5.0]
    times = vals[1:2:end]
    values = vals[2:2:end]
    stamp!(PWLVoltageSource(times, values; name=:V1),
           ctx, vcc, 0; t=spec.time, mode=spec.mode)
end
```

#### Test Coverage (`test/mna/core.jl`)

Added ~180 LOC of tests:

1. **SinVoltageSource evaluation** - Validates against analytical sine formula
2. **PWLVoltageSource evaluation** - Validates interpolation and DC mode
3. **PWL/SIN stamp! methods** - Validates time-parameterized stamping
4. **Time-dependent ODE with builder** - PWL ramp → RC circuit
5. **SIN transient simulation** - RC low-pass with sinusoidal source, validates attenuation

### Current-Controlled Sources (CCVS/CCCS) - 2024-12-22

Added support for current-controlled voltage sources (CCVS/H) and current-controlled current sources (CCCS/F).

#### Implementation (`src/spc/codegen.jl`)

CCVS (H element) generates:
```julia
stamp!(CCVS(rm; name=:H1), ctx, out_p, out_n, ctrl_current_idx)
```

CCCS (F element) generates:
```julia
stamp!(CCCS(gain; name=:F1), ctx, out_p, out_n, ctrl_current_idx)
```

Key feature: Control current index lookup via `sys.current_names` to find the voltage source current variable.

#### Bug Fixes - 2024-12-22

**Inductor/Capacitor Value Extraction:**
- Fixed codegen to extract values from `instance.val` when not using named parameters
- Example: `L1 vin n1 1.5` (value) vs `L1 vin n1 l=1.5` (named param)
- Affects both `cg_mna_instance!` for Inductor and Capacitor

**VCVS/VCCS Gain Extraction:**
- Fixed to access gain through `instance.val` (VoltageControl node)
- Was incorrectly trying to access `instance.params` which doesn't exist

**SubcktCall Parameter Handling:**
- Fixed to iterate through AST children to find Parameter nodes
- Was incorrectly trying to access `instance.params`

**Current Source Sign Convention:**
- SPICE convention: `I n+ n-` injects current into n- (second terminal)
- MNA convention: `stamp!(CurrentSource, ctx, p, n)` injects into p
- Fix: Swap p/n for current sources in codegen
- Affects: DC current sources, PWL, SIN, PULSE sources

**World Age Issues:**
- Added `Base.invokelatest` for dynamically generated circuit functions
- Required when using `Base.eval` to create circuit builders in tests

### Exit Criteria Status

| Criterion | Status |
|-----------|--------|
| Basic device stamps (R, C, L, V, I) | ✅ |
| Voltage-controlled sources (E, G) | ✅ |
| Current-controlled sources (H, F) | ✅ |
| Parameter expression evaluation | ✅ |
| Unit suffix parsing | ✅ |
| DC analysis of parsed circuits | ✅ |
| Test infrastructure | ✅ |
| Subcircuit support | ⚠️ Port handling needs work |
| High-level sp_str macro | ✅ |
| CircuitSweep with SPICE circuits | ⚠️ Subcircuit params pending |
| PWL transient sources | ✅ |
| SIN transient sources | ✅ |
| PULSE transient sources (PWL approx) | ✅ |
| Time-dependent ODE solver | ✅ |

### Test Suite Updates

**`test/common.jl`**
- Added MNA-based solve helpers:
  - `solve_mna_spice_code(code)` - Parse SPICE and solve DC with MNA
  - `solve_mna_circuit(builder)` - Solve MNA builder function
  - `tran_mna_circuit(builder, tspan)` - Transient analysis with MNA
- Added `Base.invokelatest` wrapper for world age issues

**`test/basic.jl`** (complete rewrite - 19 tests)
- All tests now use MNA backend instead of DAECompiler
- Passing tests:
  - Simple VR Circuit, Simple IR circuit, Simple VRC circuit
  - Simple SPICE sources, Simple SPICE controlled sources
  - SPICE multiplicities, .option
  - SPICE CCVS (H element), SPICE CCCS (F element)
- Skipped tests (known limitations):
  - Subcircuit parameter passing
  - .LIB include handling
  - Parameter scoping in subcircuits
  - Unit suffixes (mAmp, MegQux)
  - SPICE functions (int, floor, ceil)
  - .if/.else conditionals

**`test/transients.jl`** (18 tests)
- PWL test (12 tests): Piecewise linear current source through resistor
- Butterworth Filter test (6 tests): Third-order low-pass with SIN source
- Uses both SPICE codegen and direct MNA API for validation

**`test/sweep.jl`**
- Fixed MNA SPICE codegen tests with `invokelatest`
- Skipped: dc! sweep on SPICE-generated circuit (subcircuit params pending)

**`test/runtests.jl`**
- Phase 0/1 (reduced): basic.jl + transients.jl now run without DAECompiler
- All 37 MNA-related tests passing

### Usage Example

```julia
using CedarSim
using CedarSim.MNA: voltage

spice_code = """
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
"""

# Parse and generate builder
ast = SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
code = make_mna_circuit(ast)

# Evaluate and solve
circuit_fn = eval(code)
ctx = circuit_fn((;), MNASpec())
sol = solve_dc(ctx)

@assert voltage(sol, :out) ≈ 2.5  # Voltage divider
```

### Subcircuit Parameter Handling Fix (2024-12-23)

Fixed subcircuit parameter handling to properly use ParamLens merge semantics.

#### The Problem

The initial implementation created a `MergedLens` type that merged explicit subcircuit
parameters with the lens hierarchy. This was incorrect because:
- Merging is already what a lens does via the function call pattern
- MergedLens broke the optimization path for constant folding
- Parameter precedence was inverted

#### The Solution: Proper Lens Semantics

Refactored to use the correct lens pattern:

1. **Subcircuit builders accept explicit params as function kwargs with defaults:**
   ```julia
   function myres_mna_builder(lens, spec::MNASpec, ctx::MNAContext, p, n;
                               r=1000.0)  # Default from subcircuit definition
       r = (lens(; r=r)).r  # Merge: lens overrides take precedence
       stamp!(Resistor(r), ctx, p, n)
   end
   ```

2. **Subcircuit calls pass explicit params as kwargs:**
   ```julia
   let subckt_lens = getproperty(var"*lens#", :myres)
       myres_mna_builder(subckt_lens, spec, ctx, vcc, out; r=2000.0)
   end
   ```

3. **Parameter precedence (highest to lowest):**
   - Sweep overrides (in `lens.params`)
   - Netlist explicit params (kwargs at call site)
   - Subcircuit defaults (kwargs with defaults in function signature)

#### Changes Made

**`src/spc/codegen.jl`:**
- `codegen_mna_subcircuit()`: Build kwargs for subcircuit params with defaults
- `cg_mna_instance!()`: Pass explicit params as kwargs to subcircuit builder
- `codegen_mna!()`: Added `is_subcircuit` flag to distinguish contexts
- Top-level: params use literal default in lens call
- Subcircuit: params are function kwargs, use variable name in lens call

**`src/spectre.jl`:**
- Removed `MergedLens` type entirely

#### Tests Enabled

- ParamObserver test now passes
- `.if/.else/.endif` conditionals test now passes

See `doc/mna_architecture.md` for detailed documentation of the lens pattern.

### .if/.else/.endif Conditionals (2024-12-23)

Added support for conditional blocks in SPICE netlists.

#### Implementation

**`src/spc/codegen.jl`:**
- Added `cg_mna_if_else!()` to handle conditional blocks
- Generates Julia `if`/`elseif`/`else` blocks
- Condition expressions evaluated via `cg_expr!`

**`src/spc/interface.jl` - `make_mna_circuit()`:**
- Changed to use `sema()` instead of `sema_file_or_section()`
- `sema()` calls `resolve_scopes!()` which populates `parameter_order`
- This is required for conditional parameter handling

#### Example

```spice
.param mode=1
.if (mode==1)
R1 out 0 1k
.else
R1 out 0 2k
.endif
```

Generates:
```julia
if mode == 1
    stamp!(Resistor(1000.0), ctx, out, 0)
else
    stamp!(Resistor(2000.0), ctx, out, 0)
end
```

### Remaining Work (Phase 4)

**Low Priority:**
1. **Spectre subcircuit codegen** - Native Spectre subcircuits (`subckt`/`ends`) not yet supported
2. **Edge case testing** - Verify all device types and parameter combinations work correctly

**Completed:**
- ~~Subcircuit port handling~~ ✅
- ~~Subcircuit parameter passing~~ ✅
- ~~.if/.else conditionals~~ ✅
- ~~ParamObserver support~~ ✅
- ~~Unit suffix parsing~~ ✅ (mAmp, MegQux, Mil, etc. - implemented in `cg_expr!` for `SP.NumberLiteral`)
- ~~SPICE functions~~ ✅ (int, nint, floor, ceil, pow, ln - implemented in `SpectreEnvironment`)
- ~~Spectre basic device codegen~~ ✅ (resistor, capacitor, inductor, vsource, isource, vcvs, vccs)
- ~~.LIB include handling~~ ✅ (self-referential includes via path comparison in sema)

**Next Steps:**
- Phase 5 (VA Contribution Functions) can begin in parallel

---

## Phase 5: VA Contribution Functions

**Status:** Core Implementation Complete
**Date:** 2024-12-23
**Branch:** `claude/study-mna-backend-M0fjM`
**LOC:** ~350 (implementation) + ~220 (tests)
**Design Document:** `doc/phase5_implementation_plan.md`

### Goal
Implement s-dual contribution stamping for VA-style contributions.

### Implementation Summary

The s-dual approach uses ForwardDiff to automatically separate resistive and reactive contributions:

1. **S-Dual Representation:**
   - Laplace variable `s` represented as `Dual{ContributionTag}(0, 1)`
   - `va_ddt(x) = s * x` transforms time derivatives to frequency domain
   - `value(result)` → resistive current (stamps into G)
   - `partials(result, 1)` → charge (stamps into C via capacitance)

2. **Nested Dual Structure:**
   - Voltage dual (`Dual{Nothing}`) for ∂I/∂V Jacobian extraction
   - ContributionTag dual for resist/react separation
   - Tag ordering defined: ContributionTag < other tags

3. **Three Result Structures Handled:**
   - Pure resistive: `Dual{Nothing}(I, dI/dVp, dI/dVn)`
   - Pure reactive: `Dual{ContributionTag}(Dual{Nothing}(0,...), Dual{Nothing}(q,...))`
   - Mixed: `Dual{Nothing}(Dual{ContributionTag}(...), ...)`

### New Files

#### `src/mna/contrib.jl` (~110 LOC)

Core contribution stamping primitives:

```julia
# Tag for s-dual
struct ContributionTag end

# Laplace domain ddt
@inline function va_ddt(x::Real)
    return Dual{ContributionTag}(zero(x), x)
end

# Evaluate contribution and extract all derivatives
function evaluate_contribution(contrib_fn, Vp::Real, Vn::Real) -> NamedTuple
    # Returns: I, dI_dVp, dI_dVn, q, dq_dVp, dq_dVn
end

# Main stamping entry point
function stamp_current_contribution!(ctx, p, n, contrib_fn, x)
    # Evaluates contrib_fn at operating point
    # Stamps G (conductance) and C (capacitance) matrices
end
```

Key features:
- ForwardDiff tag ordering to handle nested duals
- Automatic Jacobian extraction for nonlinear devices
- Handles pure resistive, pure reactive, and mixed contributions

### Modified Files

#### `src/mna/MNA.jl`
Added `include("contrib.jl")` after solve.jl.

#### `src/vasim.jl` (~550 LOC added)
Added MNA device generation alongside existing DAECompiler codegen:

- `make_mna_device(vm)` - Generate MNA-compatible device struct
- `MNAScope` struct for MNA-specific code generation
- Translation methods for expressions (BinaryExpression, FunctionCall, etc.)
- `mna_collect_contributions!()` - Extract contribution statements
- `generate_mna_stamp_method_2term()` - Two-terminal device stamp method
- `generate_mna_stamp_method_nterm()` - N-terminal device stamp method
- `make_mna_module(va)` - Generate complete VA module

Updated `@va_str` macro:
```julia
macro va_str(str)
    @static if CedarSim.USE_DAECOMPILER
        esc(make_module(va))
    else
        esc(make_mna_module(va))
    end
end
```

### Test Coverage

#### `test/mna/va.jl` (~220 LOC, 41 tests)

1. **va_ddt s-dual basics** (5 tests)
   - Dual structure validation
   - Nested dual handling

2. **evaluate_contribution** (20 tests)
   - Resistor: `V/R` → correct I and dI/dV
   - Capacitor: `C*ddt(V)` → correct q and dq/dV
   - Parallel RC: `V/R + C*ddt(V)` → mixed contributions
   - Diode: `Is*(exp(V/Vt)-1)` → nonlinear conductance

3. **stamp_current_contribution!** (16 tests)
   - Matrix structure validation
   - Ground node handling
   - Nonlinear operating point stamping

### Exit Criteria Status

| Criterion | Status |
|-----------|--------|
| va_ddt creates s-dual | ✅ |
| Resistor contribution (V/R) | ✅ |
| Capacitor contribution (C*ddt) | ✅ |
| Mixed RC contribution | ✅ |
| Nonlinear diode contribution | ✅ |
| Nested dual handling | ✅ |
| stamp_current_contribution! works | ✅ |
| Core MNA tests still pass (297) | ✅ |

### Known Limitations

1. **VA string parsing with includes:**
   - `va_str` macro with `include "disciplines.vams"` doesn't work from IOBuffer
   - Full VA→MNA pipeline requires loading .va files, not inline strings
   - Contribution primitives work independently (tested)

2. **VA codegen not fully integrated:**
   - `make_mna_device()` generates device structs but full integration pending
   - Translation of complex VA statements (conditionals, loops) not complete

### Usage Example

```julia
using CedarSim.MNA
using CedarSim.MNA: va_ddt, stamp_current_contribution!, evaluate_contribution

# Define a parallel RC contribution function
R, C = 1000.0, 1e-6
contrib_fn(V) = V/R + C * va_ddt(V)

# Evaluate at operating point
result = evaluate_contribution(contrib_fn, 5.0, 0.0)
# result.I ≈ 0.005 (current)
# result.dI_dVp ≈ 0.001 (conductance)
# result.q ≈ 5e-6 (charge)
# result.dq_dVp ≈ 1e-6 (capacitance)

# Or stamp directly into MNA context
ctx = MNAContext()
p = get_node!(ctx, :p)
n = get_node!(ctx, :n)
stamp_current_contribution!(ctx, p, n, contrib_fn, [5.0, 0.0])
sys = assemble!(ctx)
# sys.G has conductance, sys.C has capacitance
```

### Architecture Notes

The s-dual approach is inspired by:
- **OpenVAF OSDI interface**: Separate resist/react loading functions
- **ForwardDiff dual numbers**: Automatic differentiation for Jacobians
- **Laplace domain**: `ddt(x) = s*x` in frequency domain

Key insight: By representing `s` as a ForwardDiff dual with `value=0, partials=1`,
evaluating a contribution function naturally separates:
- `value(result)` = contribution at s=0 = DC/resistive
- `partials(result)` = coefficient of s = charge (capacitive)

This avoids AST analysis to determine which branches are resistive vs reactive.

---

## Phase 6: Complex VA & DAE

**Status:** Not Started
**LOC Target:** ~400

### Goal
Full VA support including nonlinear capacitors.

---

## Phase 7: Advanced Features

**Status:** Not Started
**LOC Target:** ~300

### Goal
ParamSim, sweeps, sensitivity.

---

## Phase 8: Cleanup

**Status:** Not Started

### Goal
Remove dead code, DAECompiler remnants, and unused stubs.

---

## References

- `doc/mna_design.md` - Main design document
- `doc/mna_architecture.md` - Detailed architecture and parameterization patterns
- `doc/mna_ad_stamping.md` - AD-based stamping approach (if exists)
- DAECompiler.jl - Original backend (for reference)
- ngspice - Validation reference

### External Documentation
- [Virtuoso Spectre Circuit Simulator Reference](https://amarketplaceofideas.com/wp-content/uploads/2015/09/Virtuoso-Spectre-Circuit-Simulator-Reference.pdf) - Cadence Spectre 11.1 manual (see Chapter 4 for syntax)
- [SPICE Quick Reference](https://web.stanford.edu/class/ee133/handouts/general/spice_ref.pdf) - Stanford EE133 reference
