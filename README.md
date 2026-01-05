# Cadnip.jl

**C**ircuit **A**nalysis & **D**ifferentiable **N**umerical **I**ntegration **P**rogram

Cadnip is an MNA-based analog circuit simulator written in Julia, focused on simplicity, maintainability, and robustness. It is a fork of CedarSim that replaces the DAECompiler backend with a straightforward Modified Nodal Analysis (MNA) implementation.

## Features

- Import of multi-dialect SPICE/Spectre netlists
- Import of Verilog-A models
- DC and transient analyses
- Full differentiability via ForwardDiff (for sensitivities, optimization, ML, etc.)
- Parameter sweeps with `CircuitSweep`
- Works with standard Julia releases (1.11+)

## Installation

Cadnip works with release Julia and can be installed directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/NyanCAD/Cadnip.jl")
```

Or clone and develop locally:

```bash
git clone https://github.com/NyanCAD/Cadnip.jl
cd Cadnip.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

```julia
using CedarSim
using CedarSim.MNA: MNACircuit, MNASpec, voltage

# Define a circuit using SPICE syntax
builder = mna_sp"""
V1 vcc 0 DC 5
R1 vcc out 1k
R2 out 0 1k
"""

# Create circuit and run DC analysis
circuit = MNACircuit(builder)
sol = dc!(circuit)

# Access results
println("Output voltage: ", voltage(sol, :out))  # 2.5V (voltage divider)
```

## Testing

Run the test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or run specific test groups:

```bash
julia --project=. -e 'using Pkg; Pkg.test(test_args=["mna"])'
```

## License

This package is available under the MIT license (see LICENSE.MIT). You may also use it under CERN-OHL-S v2 if that better suits your project.

Contributions are welcome! Please open an issue or pull request on GitHub.

## Related Projects

- [SpiceArmyKnife.jl](SpiceArmyKnife.jl/) - Tool for parsing and converting between netlist languages
- [VerilogAParser.jl](VerilogAParser.jl/) - Verilog-A parser
- [SpectreNetlistParser.jl](SpectreNetlistParser.jl/) - Spectre netlist parser
