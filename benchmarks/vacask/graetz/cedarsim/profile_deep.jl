#!/usr/bin/env julia
#==============================================================================#
# Deep Memory Allocation Profiling for Graetz Bridge
#==============================================================================#

using CedarSim
using CedarSim.MNA
using VerilogAParser
using Profile

# Load the vadistiller diode model
const diode_va_path = joinpath(@__DIR__, "..", "..", "..", "..", "test", "vadistiller", "models", "diode.va")

if isfile(diode_va_path)
    va = VerilogAParser.parsefile(diode_va_path)
    if !va.ps.errored
        Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
    else
        error("Failed to parse diode VA model")
    end
else
    error("Diode VA model not found at $diode_va_path")
end

# Load and parse the SPICE netlist
const spice_file = joinpath(@__DIR__, "runme.sp")
const spice_code = read(spice_file, String)

# Parse SPICE to code
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:graetz_circuit,
                                         imported_hdl_modules=[sp_diode_module])
eval(circuit_code)

# Compile the circuit structure and create workspace
spec = MNASpec()
params = NamedTuple()
cs = MNA.compile_structure(graetz_circuit, params, spec)
ws = MNA.create_workspace(cs)

# Create test state
n = MNA.system_size(cs)
u = zeros(n)
t = 0.0

# Warmup
for _ in 1:10
    MNA.fast_rebuild!(ws, u, t)
end

# Profile allocations
println("=== Allocation Profile ===")

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 begin
    for _ in 1:100
        MNA.fast_rebuild!(ws, u, t)
    end
end

println("\nAllocation results (top 20):")
results = Profile.Allocs.fetch()
allocs = results.allocs

# Group by call site
alloc_groups = Dict{String,@NamedTuple{count::Int,bytes::Int}}()
for a in allocs
    key = sprint(show, a.stacktrace[1:min(5, length(a.stacktrace))])
    prev = get(alloc_groups, key, (count=0, bytes=0))
    alloc_groups[key] = (count=prev.count + 1, bytes=prev.bytes + a.size)
end

# Sort by bytes
sorted = sort(collect(alloc_groups), by=x->x[2].bytes, rev=true)

for (i, (key, stats)) in enumerate(sorted[1:min(20, length(sorted))])
    println("\n[$i] $(stats.count) allocs, $(stats.bytes) bytes:")
    println("  ", replace(key, "\n" => "\n  "))
end
