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
| 4 | SPICE Codegen | Not Started | ~300 |
| 5 | VA Contribution Functions | Not Started | ~400 |
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

### Next Steps

Phase 4 (SPICE Codegen) will:
- Modify `spc/codegen.jl` to emit `stamp!` calls
- Connect parsed netlists to MNA stamping
- Enable end-to-end SPICE simulation

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

**Status:** Not Started
**LOC Target:** ~300

### Goal
Update spc/codegen.jl to emit stamp! calls instead of DAECompiler primitives.

---

## Phase 5: VA Contribution Functions

**Status:** Not Started
**LOC Target:** ~400

### Goal
Update vasim.jl to emit contribution functions with s-dual ddt().

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
- `doc/mna_ad_stamping.md` - AD-based stamping approach (if exists)
- DAECompiler.jl - Original backend (for reference)
- ngspice - Validation reference
