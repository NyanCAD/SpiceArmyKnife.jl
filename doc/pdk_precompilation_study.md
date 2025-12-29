# PDK Precompilation Study

## Overview

This document analyzes how PDK packages (like GF180MCUPDK.jl) use precompilation and how the MNA backend supports precompiled PDK modules.

## MNA PDK Precompilation API

### Primary API: `load_mna_modules(into, file; ...)`

The MNA backend provides full PDK precompilation support through `load_mna_modules()`:

```julia
module MyPDK
    using CedarSim

    # Load PDK at module definition time - enables precompilation
    const corners = CedarSim.load_mna_modules(@__MODULE__,
        joinpath(@__DIR__, "pdk.spice"))

    # corners is a NamedTuple: (typical=Module, fast=Module, slow=Module, ...)
    # Submodules are now defined: MyPDK.typical, MyPDK.fast, MyPDK.slow

    using .typical: nfet_mna_builder, pfet_mna_builder
    export nfet_mna_builder, pfet_mna_builder
end
```

### Function Signatures

```julia
# Preferred: eval into target module (for precompilation)
load_mna_modules(into::Module, file::String;
    names=nothing,                    # Filter to specific sections
    pdk_include_paths=String[],       # Include search paths
    preload=Module[]                  # Extra using statements
) -> NamedTuple{Symbol, Module}

# Alternative: return expression for manual eval
load_mna_modules(file::String; ...) -> Expr

# Single section variants
load_mna_pdk(into::Module, file; section::String, ...) -> Module
load_mna_pdk(file; section::String, ...) -> Expr
```

### Generated Builder Functions

For each subcircuit in the PDK, a builder function is generated:

```julia
# Generated for .SUBCKT nfet_03v3 d g s b W=1e-6 L=180e-9
function nfet_03v3_mna_builder(lens, spec::MNASpec, ctx::MNAContext,
                                port_1, port_2, port_3, port_4, parent_params;
                                w=nothing, l=nothing)
    # Parameter resolution with lens override support
    # Device stamping
    # Return nothing
end
```

### Usage in Circuit Construction

```julia
using MyPDK

function build_circuit(params, spec::MNASpec)
    ctx = MNAContext()
    lens = ParamLens(params)

    # Node allocation
    vdd = get_node!(ctx, :vdd)
    out = get_node!(ctx, :out)
    gnd = get_node!(ctx, :gnd)

    # Use PDK builder
    MyPDK.typical.nfet_03v3_mna_builder(lens, spec, ctx,
        out, inp, gnd, gnd, (;);  # ports + parent_params
        w=1e-6, l=180e-9)          # instance params as kwargs

    return ctx
end
```

## Implementation Details

### Key Functions in `src/spc/codegen.jl`

| Function | Purpose |
|----------|---------|
| `load_mna_modules(into, file)` | Parse PDK, eval modules into target, return NamedTuple |
| `load_mna_modules(file)` | Parse PDK, return expression for manual eval |
| `load_mna_pdk(...)` | Single-section convenience wrapper |
| `make_mna_pdk_module(ast; name)` | Generate module expression from AST |
| `collect_lib_statements(node)` | Find `.LIB` sections in parsed AST |
| `codegen_mna_subcircuit(sema, name)` | Generate builder function for subcircuit |

### Library Section Handling

PDK files typically have multiple library sections for different corners:

```spice
.LIB typical
.SUBCKT nfet_03v3 d g s b ...
...
.ENDS
.ENDL typical

.LIB fast
.SUBCKT nfet_03v3 d g s b ...
...
.ENDS
.ENDL fast
```

Each section becomes a separate Julia module with its own builder functions.

## GF180MCUPDK.jl Migration Path

### Current Structure (DAECompiler)

```julia
module sm141064
    using CedarSim, BSIM4
    eval(CedarSim.load_spice_modules(path; names=["typical"]))
    # Exports @ckt_nfet_03v3, @ckt_pfet_03v3 macros
end
```

### Future Structure (MNA)

```julia
module sm141064
    using CedarSim, BSIM4
    const corners = CedarSim.load_mna_modules(@__MODULE__, path;
        names=["typical", "statistical"])
    using .typical: nfet_03v3_mna_builder, pfet_03v3_mna_builder
    export nfet_03v3_mna_builder, pfet_03v3_mna_builder
end
```

## Test PDK

A test PDK is provided in `test/testpdk/`:

- `testpdk.spice`: Simple PDK with typical/fast/slow corners using resistor-based models
- `pdk_test.jl`: Tests for module generation, builder usage, and corner comparison

Example from test:
```julia
const corners = CedarSim.load_mna_modules(@__MODULE__, testpdk_path)

# Use in circuit
typical.inv_x1_mna_builder(lens, spec, ctx, inp, out, vdd, vss, (;);
                           wn=360e-9, wp=720e-9, l=180e-9)
```

## Comparison: DAECompiler vs MNA PDK APIs

| Aspect | DAECompiler | MNA |
|--------|-------------|-----|
| Device API | `@ckt_*` macros | `*_mna_builder` functions |
| Precompilation | `load_spice_modules()` + eval | `load_mna_modules(@__MODULE__, ...)` |
| Parameter passing | Macro kwargs | Function kwargs + lens |
| Return type | Nothing (macro side-effects) | NamedTuple of modules |
| Subcircuit nesting | Handled by macro expansion | Builder calls builder |

## VA Device Package Precompilation

For pure Verilog-A device packages like BSIM4.jl, there are two equivalent approaches:

### Existing Pattern (VAFile + Base.include)

This is what BSIM4.jl currently uses:

```julia
module BSIM4
    using RelocatableFolders
    using CedarSim

    const bsim4_va = @path joinpath(@__DIR__, "bsim4.va")
    Base.include(@__MODULE__, VAFile(bsim4_va))

    export bsim4
end
```

### New Pattern (load_mna_va_module)

The new convenience function returns the created module:

```julia
module BSIM4
    using CedarSim

    const bsim4_mod = CedarSim.load_mna_va_module(@__MODULE__,
        joinpath(@__DIR__, "bsim4.va"))
    using .bsim4_mod: bsim4

    export bsim4
end
```

Both patterns enable precompilation. The existing `VAFile` pattern is already used by
device packages and works correctly.

## Conclusion

The MNA backend now supports full PDK precompilation through `load_mna_modules()`. PDK packages can:

1. Call `load_mna_modules(@__MODULE__, path)` at top-level
2. Get precompiled builder functions for each subcircuit
3. Export builders for downstream use
4. Use `PrecompileTools` workloads to compile specific instantiations

For VA device packages, the existing `Base.include(@__MODULE__, VAFile(path))` pattern
already works. The new `load_mna_va_module()` function provides a convenience wrapper
that returns the created module.

The key difference from DAECompiler is that MNA uses explicit builder functions instead of macros, providing clearer semantics and easier debugging.
