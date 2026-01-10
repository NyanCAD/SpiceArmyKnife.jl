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
stamp!(PSP103VA(), ctx, d, g, 0, 0; _mna_spec_=spec, _mna_x_=Float64[])
```

# Exported Models
- `JUNCAP200`: JUNCAP 200 junction diode model (2-terminal: A, K)
- `PSP103VA`: PSP 103 MOSFET (4-terminal: D, G, S, B)
- `PSP103TVA`: PSP 103 MOSFET with self-heating (5-terminal: D, G, S, B, DT)
- `PSPNQS103VA`: PSP 103 MOSFET non-quasi-static (4-terminal: D, G, S, B)
"""
module PSPModels

using CedarSim
using CedarSim: VAFile
using CedarSim.MNA: MNAContext, MNASpec, stamp!, get_node!,
                    compile_structure, create_workspace, fast_rebuild!, reset_direct_stamp!
using VerilogAParser
using PrecompileTools: @compile_workload

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

# Export device types (names match VA module declarations)
export JUNCAP200, PSP103VA, PSP103TVA, PSPNQS103VA

# Export module references for SPICE integration
export JUNCAP200_module, PSP103VA_module, PSP103TVA_module, PSPNQS103VA_module

# Precompile stamp! methods for all three method variants:
# 1. MNAContext + ZeroVector (default when _mna_x_ not passed)
# 2. MNAContext + Vector{Float64} (when tests pass _mna_x_=x with x=Float64[])
# 3. DirectStampContext + Vector{Float64} (fast_rebuild! runtime path)
# Note: PSPNQS103VA skipped - requires idt() function not yet supported
# Note: PSP103TVA skipped - requires ln_1p_d() function not yet supported
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

    # JUNCAP200 (2-terminal diode)
    function juncap_builder(params, spec, t=0.0; x=Float64[], ctx=nothing, use_zero_vector=false)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        a = get_node!(ctx, :a)
        if use_zero_vector
            stamp!(JUNCAP200_module.JUNCAP200(), ctx, a, 0; _mna_spec_=spec)
        else
            stamp!(JUNCAP200_module.JUNCAP200(), ctx, a, 0; _mna_spec_=spec, _mna_x_=x)
        end
        return ctx
    end
    precompile_device(juncap_builder, NamedTuple())

    # PSP103VA (4-terminal MOSFET)
    function psp_builder(params, spec, t=0.0; x=Float64[], ctx=nothing, use_zero_vector=false)
        if ctx === nothing
            ctx = MNAContext()
        else
            reset_for_restamping!(ctx)
        end
        d = get_node!(ctx, :d)
        g = get_node!(ctx, :g)
        s = get_node!(ctx, :s)
        if use_zero_vector
            stamp!(PSP103VA_module.PSP103VA(), ctx, d, g, s, 0; _mna_spec_=spec)
        else
            stamp!(PSP103VA_module.PSP103VA(), ctx, d, g, s, 0; _mna_spec_=spec, _mna_x_=x)
        end
        return ctx
    end
    precompile_device(psp_builder, NamedTuple())
end

end # module
