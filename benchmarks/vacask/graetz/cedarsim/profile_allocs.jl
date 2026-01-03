#!/usr/bin/env julia
#==============================================================================#
# Memory Allocation Profiling for Graetz Bridge
#==============================================================================#

using CedarSim
using CedarSim.MNA
using VerilogAParser
using BenchmarkTools

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

# Test value-only mode status
println("=== Value-Only Mode Status ===")
println("supports_ctx_reuse: ", ws.supports_ctx_reuse)
println("supports_value_only_mode: ", ws.supports_value_only_mode)
println()

# Create test state
n = MNA.system_size(cs)
u = zeros(n)
t = 0.0

# Warmup
MNA.fast_rebuild!(ws, u, t)
MNA.fast_rebuild!(ws, u, t)

# Benchmark allocations
println("=== Allocation Benchmark ===")
println("fast_rebuild! allocations:")
b = @benchmark MNA.fast_rebuild!($ws, $u, $t) samples=1000 evals=1
display(b)
println()

# Test with different time values
println("\nfast_rebuild! with varying time:")
t_vals = [0.0, 1e-6, 1e-3, 0.5]
for t_val in t_vals
    b = @benchmark MNA.fast_rebuild!($ws, $u, $t_val) samples=100 evals=1
    println("t=$t_val: median=$(median(b).memory) bytes, min=$(minimum(b).memory) bytes")
end

# Test the builder directly to see where allocations come from
println("\n=== Builder Direct Test ===")
spec = cs.spec
params = cs.params
builder = cs.builder

# Test with MNAContext (ctx reuse path)
ctx = ws.ctx
MNA.reset_for_restamping!(ctx)
println("Builder with MNAContext reuse:")
b = @benchmark begin
    MNA.reset_for_restamping!($ctx)
    $builder($params, $spec, 0.0; x=$u, ctx=$ctx)
end samples=100 evals=1
display(b)

# Test with ValueOnlyContext if supported
if ws.supports_value_only_mode
    vctx = ws.vctx
    MNA.reset_value_only!(vctx)
    println("\nBuilder with ValueOnlyContext:")
    b = @benchmark begin
        MNA.reset_value_only!($vctx)
        $builder($params, $spec, 0.0; x=$u, ctx=$vctx)
    end samples=100 evals=1
    display(b)
else
    println("\nValueOnlyContext NOT supported - this is the issue we need to fix!")
end
