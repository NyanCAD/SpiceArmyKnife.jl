# Claude Development Notes

## Environment

- **Julia is NOT pre-installed** - install juliaup first:
  - Run: `curl -fsSL https://install.julialang.org | sh -s -- -y`
  - Then source the profile: `. ~/.bashrc`
  - Add Julia 1.11: `~/.juliaup/bin/juliaup add 1.11`
  - Set as default: `~/.juliaup/bin/juliaup default 1.11`
  - Use `~/.juliaup/bin/julia` to run Julia
- **Use Julia 1.11** - this is what CI uses and what the Manifest.toml is locked to
  - Julia 1.12 has threading bugs that cause segfaults during artifact downloads
  - Don't add compatibility hacks for older Julia versions

## Development Guidelines

### Code Modification Philosophy

- **ALWAYS update existing code** - refactor and modify in place
- **NEVER add compatibility layers** - no deprecation wrappers, no duplicate APIs
- **NEVER create parallel implementations** - one clean API, not old + new
- We are at early stage development where breaking changes are expected
- If you need to change behavior, change it directly - don't preserve the old way

### MNA Backend Migration

- **DO NOT maintain backward compatibility with DAECompiler**
- Update existing APIs to use the new MNA backend directly
- When modifying sweep/simulation code, replace DAECompiler patterns with MNA equivalents
- Do not create duplicate types (e.g., `MNACircuitSweep`) - modify existing types instead

### Key MNA Components

- `MNACircuit`: Parameterized circuit simulation wrapper
- `MNAContext`: Circuit builder context for stamping (structure discovery)
- `DirectStampContext`: Zero-allocation context for fast restamping during solve
- `alter()`: Create new simulation with modified parameters
- `dc!()` / `tran!()`: DC and transient analysis
- `CircuitSweep`: Parameter sweep over MNA circuits

See `doc/` for design documents. Check `git log --oneline -20 --name-only` for recently changed files relevant to current work.

## CI and Testing

### Workflow

1. **Sanity check** - run the specific test file for what you changed
2. **Commit and push** - CI runs `test-core` + `test-integration` in parallel
3. **Run full tests locally** - `all` tests + benchmarks while CI runs

### Commands

```bash
# Specific test file (sanity check)
~/.juliaup/bin/julia --project=test test/mna/core.jl

# All tests (core + integration)
~/.juliaup/bin/julia --project=test test/runtests.jl all

# Benchmarks
~/.juliaup/bin/julia --project=. benchmarks/vacask/run_benchmarks.jl

# Parser tests
~/.juliaup/bin/julia --project=SpectreNetlistParser.jl -e 'using Pkg; Pkg.test()'
~/.juliaup/bin/julia --project=VerilogAParser.jl -e 'using Pkg; Pkg.test()'
```

### Test Files

| File | What it tests |
|------|---------------|
| `test/mna/core.jl` | MNA stamping, matrix assembly, DC/AC |
| `test/mna/va.jl` | VA contribution stamping |
| `test/basic.jl` | SPICE codegen, simple circuits |
| `test/transients.jl` | PWL/SIN sources |
| `test/sweep.jl` | Parameter sweeps |
| `test/mna/vadistiller.jl` | VADistiller models |
| `test/mna/vadistiller_integration.jl` | Large VA models (BSIM4) |
| `test/mna/audio_integration.jl` | BJT circuits |

## Gotchas and Patterns

### Builder Function Signature
MNA builder functions have signature:
```julia
function circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
```
- `params`: NamedTuple of circuit parameters
- `spec`: MNASpec with temp and mode
- `t`: Current time for time-dependent sources
- `x`: Solution vector for nonlinear devices
- `ctx`: MNAContext or DirectStampContext (reused across rebuilds)

### ParamLens Pattern
SPICE-generated circuits use `ParamLens` for hierarchical parameter access:
```julia
lens = ParamLens(params)
p = lens.inner(; R1=1000.0, R2=1000.0)  # merges defaults with overrides
```

### Verilog-A Gotcha
Disciplines (electrical, V(), I()) are IMPLICIT in VerilogAParser.
Do NOT use `include "disciplines.vams"` - causes parser bugs.

```julia
va"""
module VAResistor(p, n);
    parameter real R = 1000.0;
    inout p, n;
    electrical p, n;
    analog I(p,n) <+ V(p,n)/R;
endmodule
"""
```
