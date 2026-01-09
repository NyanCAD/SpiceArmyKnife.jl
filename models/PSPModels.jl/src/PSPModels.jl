"""
    PSPModels

Pre-parsed and precompiled PSP 103.4 MOSFET models for circuit simulation.

Provides the PSP (Penn State Philips) MOSFET model family from NXP Semiconductors,
including standard, self-heating, and non-quasi-static (NQS) variants.

# Usage
```julia
using PSPModels
using CedarSim.MNA: MNAContext, stamp!, get_node!

ctx = MNAContext()
d = get_node!(ctx, :d)
g = get_node!(ctx, :g)
stamp!(sp_psp103va(), ctx, d, g, 0, 0; _mna_spec_=spec, _mna_x_=Float64[])
```

# Exported Models
- `sp_juncap200`: JUNCAP 200 junction diode model (2-terminal: A, K)
- `sp_psp103va`: PSP 103 MOSFET (4-terminal: D, G, S, B)
- `sp_psp103tva`: PSP 103 MOSFET with self-heating (5-terminal: D, G, S, B, DT)
- `sp_pspnqs103va`: PSP 103 MOSFET non-quasi-static (4-terminal: D, G, S, B)
"""
module PSPModels

using CedarSim
using CedarSim: VAFile
using VerilogAParser

# Model directory
const VA_DIR = joinpath(@__DIR__, "..", "va")

# List of model files (without .va extension)
# Note: Order matters - juncap200 should be loaded first as it's simpler
const MODEL_NAMES = [
    "juncap200",
    "psp103",
    "psp103t",
    "psp103_nqs",
]

# Export model file paths for external use
const juncap200_va = joinpath(VA_DIR, "juncap200.va")
const psp103_va = joinpath(VA_DIR, "psp103.va")
const psp103t_va = joinpath(VA_DIR, "psp103t.va")
const psp103_nqs_va = joinpath(VA_DIR, "psp103_nqs.va")

export juncap200_va, psp103_va, psp103t_va, psp103_nqs_va

# Parse and evaluate all models at module load time
for name in MODEL_NAMES
    filepath = joinpath(VA_DIR, name * ".va")
    va = VerilogAParser.parsefile(filepath)
    Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
end

# Export device types
# Note: Module names in VA files determine the sp_ prefix:
# - JUNCAP200 -> sp_juncap200
# - PSP103VA -> sp_psp103va
# - PSP103TVA -> sp_psp103tva
# - PSPNQS103VA -> sp_pspnqs103va
export sp_juncap200, sp_psp103va, sp_psp103tva, sp_pspnqs103va

# Export module references for SPICE integration
export sp_juncap200_module, sp_psp103va_module, sp_psp103tva_module, sp_pspnqs103va_module

end # module
