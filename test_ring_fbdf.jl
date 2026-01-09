#!/usr/bin/env julia
# Test ring oscillator with new CedarRobustNLSolve and FBDF

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: MNAContext, MNASpec, MNACircuit, CedarDCOp
using VerilogAParser
using OrdinaryDiffEq: FBDF

println("="^60)
println("Ring Oscillator Test with CedarRobustNLSolve + FBDF")
println("="^60)

# Load PSP103
const psp103_path = joinpath(@__DIR__, "test", "vadistiller", "models", "psp103v4", "psp103.va")
println("\n1. Loading PSP103...")
t0 = time()
va = VerilogAParser.parsefile(psp103_path)
Core.eval(@__MODULE__, CedarSim.make_mna_module(va))
println("   Loaded in $(round(time() - t0, digits=1))s")

# Precompile both paths
device = PSP103VA(TYPE=1, W=1e-6, L=0.2e-6)
println("\n2. Precompiling MNAContext path...")
t1 = time()
precompile(MNA.stamp!, (typeof(device), MNAContext, Int, Int, Int, Int))
println("   Done in $(round(time() - t1, digits=1))s")

println("   Precompiling DirectStampContext path...")
t1b = time()
precompile(MNA.stamp!, (typeof(device), MNA.DirectStampContext, Int, Int, Int, Int))
println("   Done in $(round(time() - t1b, digits=1))s")

# Parse ring circuit
println("\n3. Parsing ring circuit...")
const spice_file = joinpath(@__DIR__, "benchmarks", "vacask", "ring", "cedarsim", "runme.sp")
const spice_code = read(spice_file, String)
const circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit,
                                         imported_hdl_modules=[PSP103VA_module])
eval(circuit_code)
println("   Parsed")

# Create circuit
println("\n4. Creating MNA circuit...")
circuit = MNACircuit(ring_circuit)

# Run transient with FBDF and new CedarDCOp
println("\n5. Running transient with FBDF + CedarDCOp (10ps)...")
println("   (This may take a while - PMOS path JIT + FBDF JIT)")
flush(stdout)
tspan = (0.0, 10e-12)  # Just 10ps
t2 = time()

# Use a progress callback
last_print = Ref(time())
progress_cb = function(u, t, integrator)
    if time() - last_print[] > 30  # Print every 30s
        println("   Progress: t=$(t), elapsed=$(round(time() - t2, digits=1))s")
        flush(stdout)
        last_print[] = time()
    end
    return false
end

try
    sol = tran!(circuit, tspan;
                solver=FBDF(),
                initializealg=CedarDCOp(),  # Uses CedarRobustNLSolve() by default
                dtmax=1e-12,
                maxiters=1000,
                force_dtmin=true,
                dense=false,
                callback=DiscreteCallback((u,t,i)->true, progress_cb))
    println("   Transient done in $(round(time() - t2, digits=1))s!")
    println("   Final time: $(sol.t[end])")
    println("   Timepoints: $(length(sol.t))")
    println("   Retcode: $(sol.retcode)")
    if length(sol.u) > 0
        println("   Max voltage at end: $(maximum(abs.(sol.u[end])))")
    end
catch e
    println("   Error after $(round(time() - t2, digits=1))s: $(typeof(e))")
    println("   $(sprint(showerror, e)[1:min(1000,end)])")
end

println("\n" * "="^60)
println("Done!")
println("="^60)
