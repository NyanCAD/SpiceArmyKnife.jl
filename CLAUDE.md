# Claude Development Notes

## Environment

- **Julia is NOT pre-installed** - install juliaup first:
  - Run: `curl -fsSL https://install.julialang.org | sh -s -- -y`
  - Then source the profile: `. ~/.bashrc`
  - Set Julia 1.11 as default: `~/.juliaup/bin/juliaup default 1.11`
  - Use `~/.juliaup/bin/julia` to run Julia
- **Use Julia 1.11** - this is what CI uses and what the Manifest.toml is locked to
  - Julia 1.12 has threading bugs that cause segfaults during artifact downloads
  - Don't add compatibility hacks for older Julia versions

## Development Guidelines

### MNA Backend Migration

- **DO NOT maintain backward compatibility with DAECompiler**
- Update existing APIs to use the new MNA backend directly
- When modifying sweep/simulation code, replace DAECompiler patterns with MNA equivalents
- Do not create duplicate types (e.g., `MNACircuitSweep`) - modify existing types instead

### Key MNA Components

- `MNASim`: Parameterized circuit simulation wrapper
- `MNAContext`: Circuit builder context for stamping components
- `alter()`: Create new simulation with modified parameters
- `dc!()` / `tran!()`: DC and transient analysis
- `CircuitSweep`: Parameter sweep over MNA circuits

## MNA Documentation

Read these files in `doc/` for detailed design information:

| File | Description |
|------|-------------|
| `doc/mna_design.md` | **Start here.** Core design decisions, key principles, sign conventions, and why we're migrating from DAECompiler |
| `doc/mna_architecture.md` | Detailed architecture: out-of-place evaluation, explicit parameter passing, GPU compatibility design |
| `doc/mna_ad_stamping.md` | Advanced: how to use ForwardDiff to extract Jacobians and handle Verilog-A contributions without AST analysis |
| `doc/mna_changelog.md` | Migration progress tracker - shows which phases are complete and what's remaining |

## Testing

Run MNA tests directly:
```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["mna"])'
```

Or run specific test files directly:
```bash
~/.juliaup/bin/julia --project=. test/mna/core.jl
~/.juliaup/bin/julia --project=. test/sweep.jl
```

## Gotchas and Patterns

### ParamLens Pattern
SPICE-generated circuits use `ParamLens` for hierarchical parameter access:
```julia
function build_circuit(params, spec)
    lens = ParamLens(params)
    # lens.subcircuit(; R=default) merges defaults with overrides
    p = lens.inner(; R1=1000.0, R2=1000.0)
    # Use p.R1, p.R2 in stamps...
end
```

The params structure must use `(subcircuit=(params=(...),))` for ParamLens to merge correctly.

### Builder Function Signature
All MNA builder functions take `(params, spec)`:
- `params`: NamedTuple of circuit parameters (or convert to ParamLens)
- `spec`: MNASpec with temp and mode

### Nested Parameter Sweeps
Use var-strings for nested paths in sweeps:
```julia
sweep = ProductSweep(var"inner.params.R1" = 100.0:200.0)
cs = CircuitSweep(builder, sweep; inner=(params=(R1=100.0,),))
```

### Phase 4 Status
SPICE codegen now emits MNA `stamp!` calls. Key files:
- `src/spc/codegen.jl` - Main codegen with `make_mna_circuit()`, `cg_mna_instance!()`
- `src/spc/interface.jl` - High-level `sp_str` macro generates MNA builders
- `src/mna/devices.jl` - Device types including time-dependent sources (PWL, SIN)

Remaining work:
- Current-controlled sources (CCVS, CCCS) - require tracking voltage source currents
