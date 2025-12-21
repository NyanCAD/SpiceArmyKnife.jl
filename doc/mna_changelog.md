# MNA Migration Changelog

This document tracks progress on the MNA (Modified Nodal Analysis) migration as defined in `mna_design.md`.

---

## Phase Overview

| Phase | Description | Status | LOC Target |
|-------|-------------|--------|------------|
| 0 | Dependency Cleanup | **Complete** | - |
| 1 | MNA Core | Not Started | ~200 |
| 2 | Simple Device Stamps | Not Started | ~200 |
| 3 | DC & Transient Solvers | Not Started | ~300 |
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

### Notes
- Stubs return placeholder values to allow codegen testing
- `equation` struct is both a type and callable (matches DAECompiler API)
- Some tests will fail because they try to execute generated code

---

## Phase 1: MNA Core

**Status:** Not Started
**LOC Target:** ~200

### Goal
Standalone MNA context and stamping primitives, tested in isolation.

### Planned Work
- [ ] Create `src/mna/context.jl` with MNAContext struct
- [ ] Node allocation: `get_node!(ctx, name)`
- [ ] Current variable allocation: `alloc_current!(ctx, name)`
- [ ] Stamping primitives: `stamp_G!`, `stamp_C!`, `stamp_b!`
- [ ] Matrix assembly: `assemble_G`, `assemble_C`
- [ ] Unit tests for 2-node, 3-node circuits

### Exit Criteria
- [ ] MNAContext can track nodes and currents
- [ ] stamp_G!/stamp_C!/stamp_b! accumulate correctly
- [ ] assemble_* produce correct sparse matrices
- [ ] Unit tests pass for simple stamp patterns

---

## Phase 2: Simple Device Stamps

**Status:** Not Started
**LOC Target:** ~200

### Goal
stamp! methods for basic devices.

### Planned Work
- [ ] `stamp!(::Resistor, ctx, p, n)`
- [ ] `stamp!(::Capacitor, ctx, p, n)`
- [ ] `stamp!(::Inductor, ctx, p, n)` (with current variable)
- [ ] `stamp!(::VoltageSource, ctx, p, n)` (with current variable)
- [ ] `stamp!(::CurrentSource, ctx, p, n)`
- [ ] Test each device against analytical solution

---

## Phase 3: DC & Transient Solvers

**Status:** Not Started
**LOC Target:** ~300

### Goal
Working DC and transient analysis.

### Planned Work
- [ ] DC solver: Newton iteration with G matrix
- [ ] Initial condition handling
- [ ] Transient: wrap as ODEProblem for DifferentialEquations.jl
- [ ] Validate RC circuit against analytical solution
- [ ] Validate against ngspice

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
