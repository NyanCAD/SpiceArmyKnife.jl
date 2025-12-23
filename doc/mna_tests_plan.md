# MNA test/basic.jl Implementation Plan

## Overview

This document provides a detailed plan to implement the missing functionality in `test/basic.jl` for the MNA backend. Currently, 10 out of 19 tests are working; 9 are skipped due to missing features.

## Current Status Summary

| Test | Status | Blocking Issue |
|------|--------|----------------|
| Simple VR Circuit | ✅ Working | - |
| Simple IR circuit | ✅ Working | - |
| Simple VRC circuit | ✅ Working | - |
| Simple SPICE sources | ✅ Working | - |
| Simple SPICE controlled sources | ✅ Working | - |
| SPICE multiplicities | ✅ Working | - |
| .option | ✅ Working | - |
| SPICE CCVS (H element) | ✅ Working | - |
| SPICE CCCS (F element) | ✅ Working | - |
| Simple Spectre sources | ⏸️ Skipped | Spectre MNA codegen |
| Simple SPICE subcircuit | ⏸️ Skipped | Subcircuit port/param handling |
| SPICE include .LIB | ⏸️ Skipped | .LIB include not implemented |
| SPICE parameter scope | ⏸️ Skipped | Subcircuit param scoping |
| units and magnitudes | ⏸️ Skipped | Unit suffix parsing (mAmp, MegQux, Mil) |
| functions | ⏸️ Skipped | SPICE functions (int, floor, ceil, etc.) |
| ifelse | ⏸️ Skipped | .if/.else/.endif conditionals |

---

## Task Breakdown

### Task 1: Fix Subcircuit Port Handling (HIGH PRIORITY)

**Problem**: When a subcircuit is instantiated, its internal ports get redefined with `get_node!` instead of using the passed-in port indices from the parent circuit.

**Files to modify**: `src/spc/codegen.jl`

**Current behavior (broken)**:
```julia
function myres_mna_builder(params, spec, ctx, port_1, port_2)
    # BUG: This creates NEW nodes instead of using port_1, port_2
    vcc = get_node!(ctx, :vcc)
    gnd = get_node!(ctx, :gnd)
    # Stamps into wrong nodes!
    stamp!(Resistor(r), ctx, vcc, gnd)
end
```

**Expected behavior**:
```julia
function myres_mna_builder(params, spec, ctx, port_1, port_2)
    # Use the passed port indices directly
    stamp!(Resistor(r), ctx, port_1, port_2)
end
```

**Implementation steps**:

1. In `codegen_mna_subcircuit()`, after extracting ports, create a mapping from internal port names to function parameter names:
   ```julia
   port_mapping = Dict(:vcc => :port_1, :gnd => :port_2)
   ```

2. Modify `codegen_mna!()` to accept an optional port mapping parameter

3. When generating net code, check if the net name is in the port mapping:
   - If yes: use the mapped parameter name directly (no `get_node!`)
   - If no: call `get_node!` as normal

4. Handle the special case of ground (node `0`) - it should never be remapped

**Test case**:
```julia
@testset "Simple SPICE subcircuit" begin
    spice_code = """
    .subckt myres vcc gnd
    .param r=1k
    R1 vcc gnd 'r'
    .ends

    V1 vcc 0 DC 1
    X1 vcc 0 myres r=2k
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    @test isapprox_deftol(voltage(sol, :vcc), 1.0)
    # I = V/R = 1V / 2kΩ = 0.5mA
    @test isapprox_deftol(-current(sol, :I_v1), 0.5e-3)
end
```

**Estimated LOC**: ~50

---

### Task 2: Fix Subcircuit Parameter Passing (HIGH PRIORITY)

**Problem**: Parameters passed to subcircuit instances aren't being properly merged with the subcircuit's default parameters.

**Files to modify**: `src/spc/codegen.jl`

**Current behavior (broken)**:
```julia
# X1 vcc 0 myres r=2k
# Currently generates:
let subckt_params = (;)  # Missing r=2k!
    myres_mna_builder(subckt_params, spec, ctx, vcc, 0)
end
```

**Expected behavior**:
```julia
let subckt_params = (r=2000.0,)
    myres_mna_builder(subckt_params, spec, ctx, vcc, 0)
end
```

**Implementation steps**:

1. Fix `cg_mna_instance!` for `SP.SubcktCall` to properly extract parameters

2. The issue is that `SubcktCall` doesn't have a `.params` field - parameters are AST children:
   ```julia
   # Fix: iterate through children to find Parameter nodes
   for child in SpectreNetlistParser.RedTree.children(instance)
       if child isa SNode{SP.Parameter}
           # Extract name and value
       end
   end
   ```

3. Merge passed parameters with defaults from the subcircuit definition

4. In the subcircuit builder, use `getfield(params, :name, default)` pattern:
   ```julia
   r = hasfield(typeof(params), :r) ? getfield(params, :r) : 1000.0
   ```

**Test case**: Same as Task 1

**Estimated LOC**: ~40

---

### Task 3: Implement SPICE Unit Suffix Parsing (MEDIUM PRIORITY)

**Problem**: Extended SPICE unit suffixes like `mAmp`, `MegQux`, `Mil` are not parsed correctly.

**Files to modify**: `src/spc/codegen.jl`

**Current `spice_magnitudes` dict**:
```julia
const spice_magnitudes = Dict(
    "t" => d"1e12",
    "g" => d"1e9",
    "meg" => d"1e6",
    "k" => d"1e3",
    "m" => d"1e-3",
    "u" => d"1e-6",
    "mil" => d"25.4e-6",
    "n" => d"1e-9",
    "p" => d"1e-12",
    "f" => d"1e-15",
    "a" => d"1e-18",
);
```

**Missing patterns**:
- `mAmp`, `mV`, etc. - magnitude prefix + unit name
- `MegOhm`, `MegQux`, etc. - `Meg` + arbitrary suffix
- Proper `Mil` handling (case-insensitive)

**Implementation steps**:

1. Expand regex to match magnitude + optional unit suffix:
   ```julia
   const spice_regex = r"(t|g|meg|k|m|u|mil|n|p|f|a)[a-z]*$"i
   ```

2. Extract just the magnitude prefix from the match:
   ```julia
   m = match(spice_regex, txt)
   if m !== nothing
       # Find longest matching prefix
       prefix = lowercase(m.match)
       for len in length(prefix):-1:1
           candidate = prefix[1:len]
           if haskey(spice_magnitudes, candidate)
               sf = spice_magnitudes[candidate]
               txt = txt[begin:end-length(m.match)]
               break
           end
       end
   end
   ```

3. Handle case-insensitivity properly (convert to lowercase before lookup)

**Test case**:
```julia
@testset "units and magnitudes" begin
    spice_code = """
    i1 vcc 0 DC -1mAmp
    r1 vcc 0 1MegQux
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # V = I*R = 1e-3 * 1e6 = 1000V
    @test isapprox(voltage(sol, :vcc), 1000.0; atol=deftol*10)
end
```

**Estimated LOC**: ~30

---

### Task 4: Add SpectreEnvironment Import to MNA Eval Module (TRIVIAL FIX)

**Problem**: The functions `int`, `nint`, `floor`, `ceil`, `pow`, `ln` **already exist** in `SpectreEnvironment` (lines 112-121 of `src/spectre_env.jl`):

```julia
const ln = log
const pow = ^
int(x) = trunc(Int, x)
nint(x) = Base.round(Int, x)

export int, nint, floor, ceil, pow, ln, log, exp, sqrt, ...
```

The `cg_expr!` function (line 154-171) correctly generates `GlobalRef(SpectreEnvironment, :int)` for function calls:
```julia
elseif isdefined(CedarSim.SpectreEnvironment, Symbol(id))
    args = map(x->cg_expr!(state, x.item), stmt.args)
    Expr(:call, GlobalRef(SpectreEnvironment, Symbol(id)), args...)
```

**The actual issue**: The evaluation module doesn't import SpectreEnvironment!

**Files to modify**: `test/common.jl`, `src/spc/interface.jl`

**Current code** (`test/common.jl:162-166`):
```julia
m = Module()
Base.eval(m, :(using CedarSim.MNA))
Base.eval(m, :(using CedarSim: ParamLens))
# MISSING: SpectreEnvironment import!
```

**Fix** - add one line:
```julia
m = Module()
Base.eval(m, :(using CedarSim.MNA))
Base.eval(m, :(using CedarSim: ParamLens))
Base.eval(m, :(using CedarSim.SpectreEnvironment))  # ADD THIS
```

Same fix needed in `src/spc/interface.jl:225-228`.

**Test case**: Same as before - just needs the import to work.

**Estimated LOC**: ~4 lines total

---

### Task 5: Implement .if/.else/.endif Conditionals (LOW PRIORITY)

**Problem**: Conditional device instantiation with `.if`/`.else`/`.endif` blocks is not fully working.

**Files to modify**: `src/spc/codegen.jl`

**Analysis**: The SemaResult already tracks conditionals in `state.sema.conditionals`. The `cond` field on instances indicates which conditional they depend on:
- `cond == 0`: Always active
- `cond > 0`: Active when condition N is true
- `cond < 0`: Active when condition N is false

**Current codegen already handles conditionals** but there may be bugs in expression evaluation.

**Implementation steps**:

1. Debug the conditional expression evaluation in `codegen_mna!`

2. Ensure `cond_syms` vector is properly initialized for all conditionals

3. Verify the conditional expressions are being evaluated correctly:
   ```julia
   # For: .if (switch == 1)
   # Should generate: cond_1 = (switch == 1)
   ```

4. Test with parameter-based conditions

**Test case**:
```julia
@testset "ifelse" begin
    spice_code = """
    .param switch=1
    v1 vcc 0 1
    .if (switch == 1)
    R1 vcc 0 1
    .else
    R1 vcc 0 2
    .endif
    """
    ctx, sol = solve_mna_spice_code(spice_code)
    # With switch=1, R1=1Ω, I = 1A
    @test isapprox(-current(sol, :I_v1), 1.0; atol=deftol*10)
end
```

**Estimated LOC**: ~30 (mostly debugging/fixes)

---

### Task 6: Implement .LIB Include Handling (LOW PRIORITY)

**Problem**: `.LIB` definitions and includes are not processed during MNA codegen.

**Files to modify**: `src/spc/codegen.jl`, `src/spc/sema.jl`

**Analysis**: The semantic analysis already handles `.LIB` statements (see `analyze_imports!`), but the MNA codegen doesn't process library definitions.

**Implementation steps**:

1. In `make_mna_circuit`, process library definitions before the main circuit

2. Handle `.LIB name ... .ENDL` blocks - these define named library sections

3. Handle `.LIB "filename" name` - these include a named section from a file

4. The challenge is that libraries can reference external files - need to resolve paths

**Test case**:
```julia
@testset "SPICE include .LIB" begin
    mktempdir() do dir
        spice_file = joinpath(dir, "selfinclude.cir")
        write(spice_file, """
        V1 vdd 0 1
        .LIB my_lib
        r1 vdd 0 1337
        .ENDL
        .LIB "selfinclude.cir" my_lib
        """)

        ast = SpectreNetlistParser.parse(spice_file; start_lang=:spice)
        code = CedarSim.make_mna_circuit(ast)
        # ... evaluate and check r1 exists with R=1337
    end
end
```

**Estimated LOC**: ~80

---

### Task 7: Subcircuit Parameter Scoping (LIKELY ALREADY WORKING)

**Problem (reported)**: Complex parameter expressions that reference parent scope parameters don't resolve correctly.

**Example**:
```spice
.subckt subcircuit1 vss gnd l=11
.param
+ par_l=1
+ par_leff='l-par_l'  ; References subcircuit parameter 'l'
r1 vss gnd 'par_leff'
.ends
```

**Analysis - The semantic analysis already handles this!**

Looking at `src/spectre.jl:984-1020` (`subckt_macro`), the old codegen:
1. Uses `toposort(deps)` to order parameters by dependencies
2. Collects variables from expressions via `let_to_julia.variables`
3. Properly handles lexical scoping

The semantic analysis (`SemaResult`) tracks:
- `formal_parameters`: Parameters declared on subcircuit line (like `l=11`)
- `exposed_parameters`: Parameters exposed to parent
- `parameter_order`: Already topologically sorted!

The MNA codegen at `codegen_mna!:1063-1095` iterates through `state.sema.parameter_order`, which comes from the sema phase.

**Actual issue**: This test is probably blocked by Tasks 1+2 (subcircuit port/param handling). Once those are fixed, parameter scoping should work because the sema phase already does the heavy lifting.

**Action**: After fixing Tasks 1+2, re-test parameter scoping. If still failing, debug the codegen's use of sema results.

**Estimated LOC**: ~0-20 (likely just debugging)

---

### Task 8: Spectre MNA Codegen (DEFERRED)

**Problem**: Spectre netlist format uses different syntax and requires separate codegen.

**Analysis**: The current focus is SPICE. Spectre codegen can be added later by:
1. Creating Spectre-specific `cg_mna_instance!` methods
2. Adding a Spectre path through `make_mna_circuit`

**Recommendation**: Defer to Phase 5+ unless specifically needed.

---

## Recommended Implementation Order

### Phase 4 Completion (Current Sprint)

1. **Task 1 + Task 2** (Subcircuit port/param handling) - These are tightly coupled
2. **Task 3** (Unit suffix parsing) - Quick win
3. **Task 4** (SPICE functions) - Quick win

### Future Work

4. **Task 5** (Conditionals) - Debug existing code
5. **Task 6** (.LIB handling) - More complex
6. **Task 7** (Parameter scoping) - Requires refactoring

---

## Implementation Details for Priority Tasks

### Detailed Fix for Subcircuit Ports (Tasks 1 & 2)

**Location**: `src/spc/codegen.jl:1156-1180` (`codegen_mna_subcircuit`)

**Current code**:
```julia
function codegen_mna_subcircuit(sema::SemaResult, subckt_name::Symbol)
    state = CodegenState(sema)
    body = codegen_mna!(state)  # BUG: This generates get_node! for ports

    subckt_ports = extract_subcircuit_ports(sema)
    port_args = [Symbol("port_", i) for i in 1:length(subckt_ports)]
    port_mappings = [...]  # Maps internal name = port_arg
    ...
end
```

**Fixed approach**:
```julia
function codegen_mna_subcircuit(sema::SemaResult, subckt_name::Symbol)
    state = CodegenState(sema)

    # Extract ports FIRST
    subckt_ports = extract_subcircuit_ports(sema)
    port_args = [Symbol("port_", i) for i in 1:length(subckt_ports)]

    # Create port mapping: internal_name => arg_symbol
    port_map = Dict(port => arg for (port, arg) in zip(subckt_ports, port_args))

    # Generate body WITH port mapping
    body = codegen_mna!(state; port_map)

    builder_name = Symbol(subckt_name, "_mna_builder")

    return quote
        function $(builder_name)(params, spec::$(MNASpec), ctx::$(MNAContext), $(port_args...))
            $body  # No port_mappings assignment - ports used directly
            return nothing
        end
    end
end
```

**Then modify `codegen_mna!`**:
```julia
function codegen_mna!(state::CodegenState; port_map::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    block = Expr(:block)
    ...

    # Codegen nets - skip ports that are in port_map
    for (net, _) in state.sema.nets
        net_name = cg_net_name!(state, net)
        if haskey(port_map, net_name)
            # Port comes from function argument, assign directly
            push!(block.args, :($net_name = $(port_map[net_name])))
        else
            # Internal node - allocate normally
            push!(block.args, :($net_name = get_node!(ctx, $(QuoteNode(net_name)))))
        end
    end
    ...
end
```

---

## Testing Strategy

1. **Unit tests**: Each task adds specific test cases (shown above)

2. **Integration tests**: Run full `test/basic.jl` after each task

3. **Regression tests**: Ensure existing passing tests still work

4. **ngspice validation**: Compare results with ngspice for complex circuits

---

## Dependencies

### External References (if needed)

- **SciML/DifferentialEquations.jl**: For ODE solvers (already integrated)
- **ngspice-sf-mirror**: For SPICE syntax reference
- **VACASK**: For MNA implementation patterns (already referenced in design)

### No cloning required for Phase 4 tasks

The remaining Phase 4 work is primarily:
1. Fixing existing codegen bugs (subcircuits)
2. Expanding existing parsers (unit suffixes)
3. Adding function definitions (SPICE functions)

All necessary context is already in the codebase.

---

## Summary

| Task | Priority | LOC | Blocks Tests |
|------|----------|-----|--------------|
| 1. Subcircuit ports | HIGH | ~50 | subcircuit, param scope |
| 2. Subcircuit params | HIGH | ~40 | subcircuit, param scope |
| 3. Unit suffixes | MEDIUM | ~30 | units and magnitudes |
| 4. SpectreEnvironment import | TRIVIAL | ~4 | functions |
| 5. Conditionals | LOW | ~30 | ifelse |
| 6. .LIB handling | LOW | ~80 | .LIB include |
| 7. Param scoping | LIKELY DONE | ~0-20 | param scope (blocked by 1+2) |
| 8. Spectre codegen | DEFERRED | ~200+ | Spectre sources |

**Total for HIGH+MEDIUM priority**: ~124 LOC (previously ~180)
**Total for all Phase 4**: ~234+ LOC (well within 300 LOC target)

## Key Findings

1. **SPICE functions already exist** in `SpectreEnvironment` - just need to import the module in MNA eval context
2. **Parameter scoping is handled by semantic analysis** - `SemaResult.parameter_order` is already topologically sorted
3. **Main blockers are subcircuit port/param handling** (Tasks 1+2) - these gate most other issues
4. **ParamObserver should just work** - it's defined in `src/spectre.jl:199-242` with no DAECompiler dependencies. The test/basic.jl comment claiming it "requires ParamSim and ParamObserver" is incorrect - these are pure Julia types that work independently of the simulation backend.
