"""
    VADistillerModels

Pre-parsed and precompiled VADistiller device models for circuit simulation.

Provides basic analog device models (resistor, capacitor, inductor, diode,
BJT, JFETs, MOSFETs) as well as advanced models (BSIM3v3, BSIM4v8, VDMOS).

# Usage
```julia
using VADistillerModels
using CedarSim.MNA: MNAContext, stamp!, get_node!

ctx = MNAContext()
vcc = get_node!(ctx, :vcc)
stamp!(sp_resistor(resistance=1000.0), ctx, vcc, 0; _mna_spec_=spec, _mna_x_=Float64[])
```

# Exported Models
- `sp_resistor`: Basic resistor
- `sp_capacitor`: Basic capacitor
- `sp_inductor`: Basic inductor
- `sp_diode`: PN junction diode
- `sp_bjt`: Bipolar junction transistor
- `sp_jfet1`, `sp_jfet2`: JFET models
- `sp_mes1`: MESFET model
- `sp_mos1` - `sp_mos9`: Level 1-9 MOSFET models
- `sp_vdmos`: Power MOSFET (5-terminal)
- `sp_bsim3v3`: Berkeley BSIM3v3 MOSFET
- `sp_bsim4v8`: Berkeley BSIM4v8 MOSFET
"""
module VADistillerModels

using CedarSim
using CedarSim: VAFile
using CedarSim.MNA: MNAContext, MNASpec, stamp!, get_node!,
                    compile_structure, create_workspace, fast_rebuild!, reset_direct_stamp!
using VerilogAParser
using PrecompileTools: @compile_workload
# Model directory
const VA_DIR = joinpath(@__DIR__, "..", "va")

# List of all model files (without .va extension)
const MODEL_NAMES = [
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "bjt",
    "jfet1",
    "jfet2",
    "mes1",
    "mos1",
    "mos2",
    "mos3",
    "mos6",
    "mos9",
    "vdmos",
    "bsim3v3",
    "bsim4v8",
]

# Export model file paths for external use
for name in MODEL_NAMES
    path_const = Symbol(name, "_va")
    @eval const $path_const = joinpath(VA_DIR, $name * ".va")
    @eval export $path_const
end

# Parse and evaluate all models at module load time
for name in MODEL_NAMES
    filepath = joinpath(VA_DIR, name * ".va")
    va = VerilogAParser.parsefile(filepath)
    Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
end

# Export all sp_ types
export sp_resistor, sp_capacitor, sp_inductor, sp_diode, sp_bjt
export sp_jfet1, sp_jfet2, sp_mes1
export sp_mos1, sp_mos2, sp_mos3, sp_mos6, sp_mos9
export sp_vdmos, sp_bsim3v3, sp_bsim4v8

# Export module references for SPICE integration
export sp_resistor_module, sp_capacitor_module, sp_inductor_module
export sp_diode_module, sp_bjt_module
export sp_jfet1_module, sp_jfet2_module, sp_mes1_module
export sp_mos1_module, sp_mos2_module, sp_mos3_module, sp_mos6_module, sp_mos9_module
export sp_vdmos_module, sp_bsim3v3_module, sp_bsim4v8_module

# Precompile stamp! methods for both MNAContext and DirectStampContext
# Use module-qualified names since types are created via Core.eval
# Note: Complex models (BSIM3v3, BSIM4v8, VDMOS) skipped due to compile-time issues
@compile_workload begin
    spec = MNASpec()

    # Helper to precompile both context types for a device
    function precompile_device(builder, params)
        # Phase 1: MNAContext (structure discovery)
        ctx = builder(params, spec, 0.0)

        # Phase 2: Compile structure and create DirectStampContext workspace
        cs = compile_structure(builder, params, spec; ctx=ctx)
        ws = create_workspace(cs; ctx=ctx)

        # Phase 3: DirectStampContext (value-only restamping)
        reset_direct_stamp!(ws.dctx)
        fast_rebuild!(ws, zeros(cs.n), 0.0)
    end

    # 2-terminal device builder template
    function resistor_builder(params, spec, t=0.0; x=Float64[], ctx=nothing)
        ctx = ctx === nothing ? MNAContext() : ctx
        n1 = get_node!(ctx, :n1)
        stamp!(sp_resistor_module.sp_resistor(), ctx, n1, 0; _mna_spec_=spec)
        return ctx
    end
    precompile_device(resistor_builder, NamedTuple())

    # 4-terminal MOSFET builder (most common usage)
    function mos1_builder(params, spec, t=0.0; x=Float64[], ctx=nothing)
        ctx = ctx === nothing ? MNAContext() : ctx
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)
        s = get_node!(ctx, :s)
        stamp!(sp_mos1_module.sp_mos1(), ctx, d, g, s, 0; _mna_spec_=spec)
        return ctx
    end
    precompile_device(mos1_builder, NamedTuple())
end

end # module
