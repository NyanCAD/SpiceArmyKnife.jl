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
using CedarSim.MNA: fast_residual!, fast_rebuild!, update_sparse_from_coo!, compile_circuit
using CedarSim.MNA: PrecompiledCircuit, MNAContext, get_node!, alloc_current!
using CedarSim.MNA: stamp_G!, stamp_C!, stamp_b!, MNASpec
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
function build_rc(params, spec; x=Float64[])
    ctx = MNAContext()
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

# Set up circuit
params = (R=1000.0, C=1e-6)
spec = MNASpec(temp=27.0, mode=:tran)

# Compile the circuit
println("Compiling circuit...")
pc = compile_circuit(build_rc, params, spec)

# Create test vectors
n = CedarSim.MNA.system_size(pc)
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
@code_warntype fast_residual!(resid, du, u, pc, t)
println()

println("@allocated for fast_residual! (10 calls, warmed up):")
println("-" ^ 40)
# Warm up
for _ in 1:5
    fast_residual!(resid, du, u, pc, t)
end
# Measure allocations
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated fast_residual!(resid, du, u, pc, t))
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
@code_warntype fast_rebuild!(pc, u, t)
println()

println("@allocated for fast_rebuild! (10 calls, warmed up):")
println("-" ^ 40)
for _ in 1:5
    fast_rebuild!(pc, u, t)
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated fast_rebuild!(pc, u, t))
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
@code_warntype build_rc(params, spec_t; x=u)
println()

println("@allocated for build_rc (10 calls, warmed up):")
println("-" ^ 40)
for _ in 1:5
    build_rc(params, spec_t; x=u)
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated build_rc(params, spec_t; x=u))
end
println("Allocations: $(allocs) bytes")
println("Average: $(sum(allocs)/length(allocs)) bytes")
println()

#==============================================================================#
# Analysis 4: update_sparse_from_coo! type stability
#==============================================================================#

println("=" ^ 60)
println("Analysis 4: update_sparse_from_coo! type stability")
println("=" ^ 60)
println()

println("@code_warntype for update_sparse_from_coo!:")
println("-" ^ 40)
@code_warntype update_sparse_from_coo!(pc.G, pc.G_V, pc.G_coo_to_nz, pc.G_n_coo)
println()

println("@allocated for update_sparse_from_coo! (10 calls, warmed up):")
println("-" ^ 40)
for _ in 1:5
    update_sparse_from_coo!(pc.G, pc.G_V, pc.G_coo_to_nz, pc.G_n_coo)
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated update_sparse_from_coo!(pc.G, pc.G_V, pc.G_coo_to_nz, pc.G_n_coo))
end
println("Allocations: $(allocs) bytes")
println("Average: $(sum(allocs)/length(allocs)) bytes")
println()

#==============================================================================#
# Analysis 5: Check PrecompiledCircuit field types
#==============================================================================#

println("=" ^ 60)
println("Analysis 5: PrecompiledCircuit field types")
println("=" ^ 60)
println()

for field in fieldnames(typeof(pc))
    val = getfield(pc, field)
    println("  $field: $(typeof(val))")
end
println()

#==============================================================================#
# Analysis 6: MNAContext constructor allocations
#==============================================================================#

println("=" ^ 60)
println("Analysis 6: MNAContext constructor")
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
# Analysis 7: Investigate what's inside fast_rebuild!
#==============================================================================#

println("=" ^ 60)
println("Analysis 7: Breakdown of fast_rebuild! allocations")
println("=" ^ 60)
println()

# The key insight: fast_rebuild! calls pc.builder, which creates a new MNAContext
# Let's measure each part

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
println("\nBuilder call (pc.builder):")
for _ in 1:5
    pc.builder(pc.params, MNASpec(temp=27.0, mode=:tran, time=0.0); x=u)
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated pc.builder(pc.params, MNASpec(temp=27.0, mode=:tran, time=0.0); x=u))
end
println("  Builder call: $(allocs) bytes")
println()

#==============================================================================#
# Analysis 8: Test the VACASK benchmark circuit
#==============================================================================#

println("=" ^ 60)
println("Analysis 8: VACASK RC Circuit from benchmark")
println("=" ^ 60)
println()

# Parse the actual VACASK benchmark SPICE file
rc_spice = read("/home/user/SpiceArmyKnife.jl/benchmarks/vacask/rc/cedarsim/runme.sp", String)
println("SPICE file content:")
println("-" ^ 40)
println(rc_spice)
println("-" ^ 40)
println()

# Parse to MNA code
circuit_code = parse_spice_to_mna(rc_spice; circuit_name=:vacask_rc_circuit)
eval(circuit_code)

# Create sim
sim = MNASim(vacask_rc_circuit)
MNA.assemble!(sim)

# Get the precompiled circuit
vacask_pc = sim.precompiled

println("VACASK circuit system size: $(CedarSim.MNA.system_size(vacask_pc))")

# Test allocations on VACASK circuit
println("\n@allocated for fast_residual! on VACASK circuit:")
vacask_n = CedarSim.MNA.system_size(vacask_pc)
vacask_u = zeros(vacask_n)
vacask_du = zeros(vacask_n)
vacask_resid = zeros(vacask_n)

for _ in 1:5
    fast_residual!(vacask_resid, vacask_du, vacask_u, vacask_pc, 0.0)
end
allocs = Int[]
for _ in 1:10
    push!(allocs, @allocated fast_residual!(vacask_resid, vacask_du, vacask_u, vacask_pc, 0.0))
end
println("Allocations: $(allocs) bytes")
println("Average: $(sum(allocs)/length(allocs)) bytes")
println()

#==============================================================================#
# Summary
#==============================================================================#

println("=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
println()
println("Key finding: fast_rebuild! allocates because it:")
println("  1. Creates a new MNASpec each call (~0 bytes)")
println("  2. Calls pc.builder, which creates a NEW MNAContext!")
println()
println("MNAContext allocates vectors for:")
println("  - node_names::Vector{Symbol}")
println("  - node_to_idx::Dict{Symbol,Int}")
println("  - current_names::Vector{Symbol}")
println("  - G_I, G_J, G_V::Vector")
println("  - C_I, C_J, C_V::Vector")
println("  - b::Vector{Float64}")
println("  - b_I, b_V::Vector")
println("  - internal_node_flags::BitVector")
println()
println("SOLUTION: Instead of calling builder each iteration, we should")
println("cache the stamp values and only update them via closure evaluation.")
println()
