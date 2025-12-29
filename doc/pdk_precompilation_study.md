# PDK Precompilation Study

## Overview

This document analyzes how PDK packages (like GF180MCUPDK.jl) use precompilation and whether the MNA backend can support precompiled PDK modules.

## GF180MCUPDK.jl Precompilation Technique

### Module Structure

The PDK has a layered structure:

```julia
module GF180MCUPDK

module design
    using CedarSim
    sp"""
    .include "../model/design.ngspice"
    """i
    export cap_mc_skew, fnoicor, mc_skew, ...
end

module sm141064
    using CedarSim, BSIM4, ..design
    path = joinpath(@__DIR__, "../model/sm141064.ngspice")
    eval(CedarSim.load_spice_modules(path;
        pdk_include_paths=[dirname(path)],
        names=["typical", "statistical"],
        preload=[CedarSim, BSIM4, design],
        exports=[...]
    ))
end

# PrecompileTools workload
@setup_workload begin
    (;d, g, s, b) = nets()
    @compile_workload begin
        @ckt_nfet_03v3(nodes=(g, d, s, b), w=1e-6, l=1e-6)
        @ckt_pfet_03v3(nodes=(g, d, s, b), w=1e-6, l=1e-6)
        ...
    end
end

end
```

### Key Functions

1. **`load_spice_modules(file; ...)`** (`src/spectre.jl:1694`):
   - Parses PDK SPICE files
   - Calls `make_spectre_modules()` to generate Julia modules

2. **`make_spectre_modules()`** (`src/spectre.jl:1664`):
   - Iterates through `.lib` sections
   - Generates a Julia module for each section
   - Uses `make_spectre_netlist()` to generate the content

3. **`CedarParseCache`** (`src/spc/cache.jl`):
   - Caches parsed SPICE/VA files per module
   - Uses `include_dependency()` for recompilation invalidation
   - Accessed via `module.var"#cedar_parse_cache#"`

### jlpkg:// Path Convention

SPICE files can reference Julia packages using `jlpkg://`:

```spice
.LIB "jlpkg://GF180MCUPDK/sm141064.ngspice" typical
```

Resolution in `src/spc/sema.jl:458-523`:
1. Extract package name from path
2. Load the package's parse cache
3. Use cached sema results or parse fresh

## MNA Codegen Architecture

### Builder Function Pattern

The MNA backend generates builder functions:

```julia
function circuit(params, spec::MNASpec)
    ctx = MNAContext()
    lens = ParamLens(params)

    # Node allocation
    node_vcc = get_node!(ctx, :vcc)
    node_out = get_node!(ctx, :out)

    # Device stamping
    stamp!(resistor(1000.0), ctx, spec, node_vcc, node_out)

    return ctx
end
```

### Key Codegen Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `make_mna_circuit()` | `src/spc/codegen.jl:2343` | Top-level circuit codegen |
| `codegen_mna!()` | `src/spc/codegen.jl:1989` | Generate builder body |
| `codegen_mna_subcircuit()` | `src/spc/codegen.jl:2256` | Generate subcircuit builder |
| `make_mna_device()` | `src/vasim.jl:139` | Generate VA device `stamp!` method |

### Include/Import Handling

The sema phase handles `jlpkg://` imports:

```julia
# In sema.jl during .LIB/.INCLUDE processing:
if startswith(str, JLPATH_PREFIX)
    path = str[sizeof(JLPATH_PREFIX)+1:end]
    components = splitpath(path)
    imp = Symbol(components[1])
    imp_mod = sema_resolve_import(scope, imp)  # Load PDK module
    parse_cache = imp_mod.var"#cedar_parse_cache#"  # Get its cache
    # ... merge sema results
end
```

## Comparison: DAECompiler vs MNA

| Aspect | DAECompiler | MNA |
|--------|-------------|-----|
| Device API | `@ckt_*` macros | `stamp!()` methods |
| Subcircuits | Macro expansion | Builder functions |
| Precompilation | Macros + workload | Builder compilation |
| PDK structure | Exports macros | Could export builders |
| Time integration | DAECompiler handles | DifferentialEquations.jl |

## MNA Precompilation Support Assessment

### What Already Works

| Component | Status | Notes |
|-----------|--------|-------|
| SPICE parsing cache | ✅ | `CedarParseCache` works with MNA |
| Sema result cache | ✅ | Stored in `spc_cache` dict |
| VA device generation | ✅ | `make_mna_device()` creates precompilable `stamp!` methods |
| `jlpkg://` resolution | ✅ | Sema phase resolves and caches |

### What Needs Work

| Component | Status | Issue |
|-----------|--------|-------|
| Subcircuit builders | ⚠️ | Generated inline, not exported |
| PDK module structure | ⚠️ | Creates `@ckt_*` macros, not MNA builders |
| PrecompileTools workload | ⚠️ | Compiles macro-based API, not MNA API |

## Recommendation

### Option A: Minimal Changes (Recommended)

Use the existing `jlpkg://` infrastructure without modifying PDK packages.

**How it works**:
- MNA sema already handles `jlpkg://` imports
- Parse caching provides speedup on repeated use
- Tests just need to use MNA API instead of DAECompiler API

**Test conversion example**:
```julia
# Old (DAECompiler)
sa = SPICENetlistParser.parsefile(path)
code = CedarSim.make_spectre_circuit(sa, [dffdir])
circuit = eval(code)
sys = CircuitIRODESystem(circuit)
prob = DAEProblem(sys, nothing, nothing, (0.0, 7e-7))
sol = solve(prob, IDA())

# New (MNA)
sa = SpectreNetlistParser.parsefile(path)
builder_code = CedarSim.make_mna_circuit(sa)
builder = eval(builder_code)
sim = MNASim(builder; <params>)
dae_data = MNA.make_dae_problem(sim; tspan=(0.0, 7e-7))
prob = DAEProblem(dae_data.f!, dae_data.du0, dae_data.u0, dae_data.tspan)
sol = solve(prob, IDA())
```

### Option B: Full MNA Precompilation

Add MNA-specific PDK module generation:

1. New `make_mna_modules()` function that exports builder functions
2. Update PDK packages to use MNA API
3. PrecompileTools workload for MNA builders

This provides maximum precompilation but requires more changes.

## Conclusion

The MNA backend **can** support PDK precompilation through the existing `jlpkg://` infrastructure. The key insight is that **parsing and sema caching already work** - the PDK files are parsed once and cached.

For enabling PDK tests now, Option A (minimal changes) is recommended:
1. Convert tests to use MNA API (`MNASim`, `make_mna_circuit`)
2. Keep using `jlpkg://` paths in SPICE files
3. No PDK package modifications needed

Full MNA precompilation (Option B) can be pursued later if performance analysis shows it's needed.
