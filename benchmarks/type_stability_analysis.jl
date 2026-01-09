#!/usr/bin/env julia
#==============================================================================#
# Type Stability Investigation for MNA Residual Functions
#
# This script analyzes the type inference of key functions in the MNA
# transient simulation path to identify sources of allocations.
#==============================================================================#

using Pkg
Pkg.activate(".")

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: fast_residual!, fast_rebuild!, compile_structure, create_workspace
using CedarSim.MNA: EvalWorkspace, CompiledStructure, MNAContext, get_node!, alloc_current!
using CedarSim.MNA: stamp_G!, stamp_C!, stamp_b!, MNASpec, MNACircuit, compile
using InteractiveUtils

println("=" ^ 60)
println("MNA Residual Type Stability Investigation")
println("=" ^ 60)
println()

#==============================================================================#
# Test Circuit: Simple RC
#==============================================================================#

println("Setting up simple RC circuit...")

# Simple RC circuit builder
function build_rc(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
    if ctx === nothing
        ctx = MNAContext()
    else
        CedarSim.MNA.reset_for_restamping!(ctx)
    end
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    # Get voltages from x
    Vvcc = isempty(x) ? 0.0 : (vcc == 0 ? 0.0 : x[vcc])
    Vout = isempty(x) ? 0.0 : (out == 0 ? 0.0 : x[out])

    # Voltage source: 1V
    I_V1 = alloc_current!(ctx, :I_V1)
    stamp_G!(ctx, vcc, I_V1, 1.0)
    stamp_G!(ctx, I_V1, vcc, 1.0)
    stamp_b!(ctx, I_V1, 1.0)

    # Resistor: 1kΩ
    R = params.R
    G = 1.0 / R
    stamp_G!(ctx, vcc, vcc, G)
    stamp_G!(ctx, vcc, out, -G)
    stamp_G!(ctx, out, vcc, -G)
    stamp_G!(ctx, out, out, G)

    # Capacitor: 1µF
    C = params.C
    stamp_C!(ctx, out, out, C)

    return ctx
end

# Set up circuit and compile
params = (R=1000.0, C=1e-6)
spec = MNASpec(temp=27.0, mode=:tran)

# Compile the circuit using new EvalWorkspace API
println("Compiling circuit...")
circuit = MNACircuit(build_rc; R=params.R, C=params.C, spec=spec)
ws = compile(circuit)
cs = ws.structure

# Create test vectors
n = CedarSim.MNA.system_size(ws)
u = zeros(n)
du = zeros(n)
resid = zeros(n)
t = 0.0

println("System size: $n")
println()

#==============================================================================#
# Analysis 1: fast_residual! type stability
#==============================================================================#

println("=" ^ 60)
println("Analysis 1: fast_residual! type stability")
println("=" ^ 60)
println()

println("@code_warntype for fast_residual!:")
println("-" ^ 40)
@code_warntype fast_residual!(resid, du, u, ws, t)
println()

println("@allocated for fast_residual! (10 calls, warmed up):")
println("-" ^ 40)
# Warm up
for _ in 1:5
    fast_residual!(resid, du, u, ws, t)
end
# Measure allocations
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated fast_residual!(resid, du, u, ws, t))
end
println("Allocations: $(allocs) bytes")
println("Average: $(sum(allocs)/length(allocs)) bytes")
println()

#==============================================================================#
# Analysis 2: fast_rebuild! type stability
#==============================================================================#

println("=" ^ 60)
println("Analysis 2: fast_rebuild! type stability")
println("=" ^ 60)
println()

println("@code_warntype for fast_rebuild!:")
println("-" ^ 40)
@code_warntype fast_rebuild!(ws, u, t)
println()

println("@allocated for fast_rebuild! (10 calls, warmed up):")
println("-" ^ 40)
for _ in 1:5
    fast_rebuild!(ws, u, t)
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated fast_rebuild!(ws, u, t))
end
println("Allocations: $(allocs) bytes")
println("Average: $(sum(allocs)/length(allocs)) bytes")
println()

#==============================================================================#
# Analysis 3: Builder function type stability
#==============================================================================#

println("=" ^ 60)
println("Analysis 3: Builder function (build_rc) type stability")
println("=" ^ 60)
println()

# Create a spec with time
spec_t = MNASpec(temp=27.0, mode=:tran, time=0.0)

println("@code_warntype for build_rc:")
println("-" ^ 40)
@code_warntype build_rc(params, spec_t, 0.0; x=u)
println()

println("@allocated for build_rc (10 calls, warmed up):")
println("-" ^ 40)
for _ in 1:5
    build_rc(params, spec_t, 0.0; x=u)
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated build_rc(params, spec_t, 0.0; x=u))
end
println("Allocations: $(allocs) bytes")
println("Average: $(sum(allocs)/length(allocs)) bytes")
println()

#==============================================================================#
# Analysis 4: Check EvalWorkspace/CompiledStructure field types
#==============================================================================#

println("=" ^ 60)
println("Analysis 4: EvalWorkspace field types")
println("=" ^ 60)
println()

println("EvalWorkspace fields:")
for field in fieldnames(typeof(ws))
    val = getfield(ws, field)
    println("  $field: $(typeof(val))")
end
println()

println("CompiledStructure fields:")
for field in fieldnames(typeof(cs))
    val = getfield(cs, field)
    println("  $field: $(typeof(val))")
end
println()

#==============================================================================#
# Analysis 5: MNAContext constructor allocations
#==============================================================================#

println("=" ^ 60)
println("Analysis 5: MNAContext constructor")
println("=" ^ 60)
println()

println("@allocated for MNAContext() (10 calls):")
for _ in 1:5
    MNAContext()
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated MNAContext())
end
println("Allocations: $(allocs) bytes")
println("Average: $(sum(allocs)/length(allocs)) bytes")
println()

#==============================================================================#
# Analysis 6: Breakdown of fast_rebuild! allocations
#==============================================================================#

println("=" ^ 60)
println("Analysis 6: Breakdown of fast_rebuild! allocations")
println("=" ^ 60)
println()

# 1. MNASpec creation (should be tiny)
println("Creating MNASpec with time:")
for _ in 1:5
    MNASpec(temp=27.0, mode=:tran, time=0.0)
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated MNASpec(temp=27.0, mode=:tran, time=0.0))
end
println("  MNASpec creation: $(allocs) bytes")

# 2. Builder call
println("\nBuilder call (cs.builder):")
for _ in 1:5
    cs.builder(cs.params, MNASpec(temp=27.0, mode=:tran, time=0.0), 0.0; x=u, ctx=ws.dctx)
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated cs.builder(cs.params, MNASpec(temp=27.0, mode=:tran, time=0.0), 0.0; x=u, ctx=ws.dctx))
end
println("  Builder call: $(allocs) bytes")
println()

println("=" ^ 60)
println("Type stability analysis complete!")
println("=" ^ 60)
