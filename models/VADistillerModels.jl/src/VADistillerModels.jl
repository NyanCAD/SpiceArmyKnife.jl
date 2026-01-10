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

# Precompile stamp! methods for all three method variants:
# 1. MNAContext + ZeroVector (default when _mna_x_ not passed)
# 2. MNAContext + Vector{Float64} (when tests pass _mna_x_=x with x=Float64[])
# 3. DirectStampContext + Vector{Float64} (fast_rebuild! runtime path)
@compile_workload begin
    using CedarSim.MNA: reset_for_restamping!, ZERO_VECTOR
    spec = MNASpec()

    # Helper to precompile all three method variants for a device
    function precompile_device(builder, params)
        # Phase 1a: MNAContext + ZeroVector (stamp! called without _mna_x_)
        ctx1 = builder(params, spec, 0.0; use_zero_vector=true)

        # Phase 1b: MNAContext + Vector{Float64} (stamp! called with _mna_x_=Float64[])
        ctx2 = builder(params, spec, 0.0; use_zero_vector=false)

        # Phase 2: Compile structure and create DirectStampContext workspace
        cs = compile_structure(builder, params, spec; ctx=ctx2)
        ws = create_workspace(cs; ctx=ctx2)

        # Phase 3: DirectStampContext + Vector{Float64} (runtime restamping)
        reset_direct_stamp!(ws.dctx)
        fast_rebuild!(ws, zeros(cs.n), 0.0)
    end

    # Helper to build stamp call with optional _mna_x_
    function stamp_with_x!(dev, ctx, nodes...; spec, x, use_zero_vector)
        if use_zero_vector
            stamp!(dev, ctx, nodes...; _mna_spec_=spec)
        else
            stamp!(dev, ctx, nodes...; _mna_spec_=spec, _mna_x_=x)
        end
    end

    # Generic builder factory for 2-terminal devices
    function make_2term_builder(device_fn)
        function builder(params, spec, t=0.0; x=Float64[], ctx=nothing, use_zero_vector=false)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            n1 = get_node!(ctx, :n1)
            stamp_with_x!(device_fn(), ctx, n1, 0; spec=spec, x=x, use_zero_vector=use_zero_vector)
            return ctx
        end
    end

    # Generic builder factory for 3-terminal devices (jfet, mes)
    function make_3term_builder(device_fn)
        function builder(params, spec, t=0.0; x=Float64[], ctx=nothing, use_zero_vector=false)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)
            stamp_with_x!(device_fn(), ctx, d, g, 0; spec=spec, x=x, use_zero_vector=use_zero_vector)
            return ctx
        end
    end

    # Generic builder factory for 4-terminal devices (mos, bjt)
    function make_4term_builder(device_fn)
        function builder(params, spec, t=0.0; x=Float64[], ctx=nothing, use_zero_vector=false)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)
            s = get_node!(ctx, :s)
            stamp_with_x!(device_fn(), ctx, d, g, s, 0; spec=spec, x=x, use_zero_vector=use_zero_vector)
            return ctx
        end
    end

    # Generic builder factory for 5-terminal devices (vdmos)
    function make_5term_builder(device_fn)
        function builder(params, spec, t=0.0; x=Float64[], ctx=nothing, use_zero_vector=false)
            if ctx === nothing
                ctx = MNAContext()
            else
                reset_for_restamping!(ctx)
            end
            d = get_node!(ctx, :d)
            g = get_node!(ctx, :g)
            s = get_node!(ctx, :s)
            tj = get_node!(ctx, :tj)
            stamp_with_x!(device_fn(), ctx, d, g, s, 0, tj; spec=spec, x=x, use_zero_vector=use_zero_vector)
            return ctx
        end
    end

    # 2-terminal devices
    precompile_device(make_2term_builder(() -> sp_resistor_module.sp_resistor()), NamedTuple())
    precompile_device(make_2term_builder(() -> sp_capacitor_module.sp_capacitor()), NamedTuple())
    precompile_device(make_2term_builder(() -> sp_inductor_module.sp_inductor()), NamedTuple())
    precompile_device(make_2term_builder(() -> sp_diode_module.sp_diode()), NamedTuple())

    # 3-terminal devices (JFETs, MESFET)
    precompile_device(make_3term_builder(() -> sp_jfet1_module.sp_jfet1()), NamedTuple())
    precompile_device(make_3term_builder(() -> sp_jfet2_module.sp_jfet2()), NamedTuple())
    precompile_device(make_3term_builder(() -> sp_mes1_module.sp_mes1()), NamedTuple())

    # 4-terminal devices (BJT uses c,b,e,s but mapped to d,g,s,b positions)
    precompile_device(make_4term_builder(() -> sp_bjt_module.sp_bjt()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos1_module.sp_mos1()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos2_module.sp_mos2()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos3_module.sp_mos3()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos6_module.sp_mos6()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_mos9_module.sp_mos9()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_bsim3v3_module.sp_bsim3v3()), NamedTuple())
    precompile_device(make_4term_builder(() -> sp_bsim4v8_module.sp_bsim4v8()), NamedTuple())

    # 5-terminal devices (VDMOS with thermal node)
    precompile_device(make_5term_builder(() -> sp_vdmos_module.sp_vdmos()), NamedTuple())
end

end # module
