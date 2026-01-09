"""
    CMCModels

Pre-parsed and precompiled CMC (Compact Model Coalition) device models for circuit simulation.

Provides foundry-standard compact models from the CMC organization, including
BSIM-CMG for multi-gate (FinFET) transistors.

# Usage
```julia
using CMCModels
using CedarSim.MNA: MNAContext, stamp!, get_node!

ctx = MNAContext()
d = get_node!(ctx, :d)
g = get_node!(ctx, :g)
e = get_node!(ctx, :e)
stamp!(sp_bsimcmg(), ctx, d, g, 0, e; _mna_spec_=spec, _mna_x_=Float64[])
```

# Exported Models
- `sp_bsimcmg`: BSIM-CMG 107 multi-gate MOSFET (4-terminal: D, G, S, E)

# Model Details
BSIM-CMG (Berkeley Short-channel IGFET Model - Common Multi-Gate) is the
industry-standard compact model for FinFET and other multi-gate transistors.
Developed at UC Berkeley under Prof. Chenming Hu, it is used by major foundries
for advanced process nodes.

Version: 107.0.0
"""
module CMCModels

using CedarSim
using CedarSim: VAFile
using VerilogAParser

# Model directory
const VA_DIR = joinpath(@__DIR__, "..", "va")

# Export model file paths for external use
const bsimcmg_va = joinpath(VA_DIR, "bsimcmg.va")
export bsimcmg_va

# Parse and evaluate the BSIM-CMG model at module load time
let
    filepath = joinpath(VA_DIR, "bsimcmg.va")
    va = VerilogAParser.parsefile(filepath)
    Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
end

# Export device type
export sp_bsimcmg

# Export module reference for SPICE integration
export sp_bsimcmg_module

end # module
