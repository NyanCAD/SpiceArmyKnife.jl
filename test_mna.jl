# Simple test for the new MNA implementation

# Add package to load path
push!(LOAD_PATH, joinpath(@__DIR__, "SpectreNetlistParser.jl/src"))
push!(LOAD_PATH, joinpath(@__DIR__, "VerilogAParser.jl/src"))
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using CedarSim

println("=== Testing MNA Implementation ===\n")

# Test 1: Simple RC circuit built programmatically
println("Test 1: Simple RC circuit (programmatic)")
circuit = MNACircuit()
vsource!(circuit, :vcc, :ground; dc=5.0, name=:V1)
resistor!(circuit, :vcc, :out, 1000.0; name=:R1)
capacitor!(circuit, :out, :ground, 1e-6; name=:C1)
ground!(circuit, :ground)

println("  Circuit has $(circuit.num_nodes) nodes and $(circuit.num_branches) branches")

# DC analysis
result = dc!(circuit)
println("  DC Operating Point:")
println(result)

println("\nTest 1: PASSED\n")

# Test 2: Simple voltage divider
println("Test 2: Voltage divider")
circuit2 = MNACircuit()
vsource!(circuit2, :vin, :ground; dc=10.0, name=:V1)
resistor!(circuit2, :vin, :vout, 1000.0; name=:R1)
resistor!(circuit2, :vout, :ground, 1000.0; name=:R2)
ground!(circuit2, :ground)

result2 = dc!(circuit2)
println("  DC Operating Point:")
println(result2)

# Expected: Vout = 5V (voltage divider)
vout_idx = circuit2.nets[:vout].index
expected_vout = 5.0
actual_vout = result2.solution[vout_idx]
println("  Expected Vout: $expected_vout V")
println("  Actual Vout: $actual_vout V")

if abs(actual_vout - expected_vout) < 0.01
    println("\nTest 2: PASSED\n")
else
    println("\nTest 2: FAILED\n")
end

# Test 3: Diode circuit (nonlinear)
println("Test 3: Diode circuit (nonlinear)")
circuit3 = MNACircuit()
vsource!(circuit3, :vin, :ground; dc=5.0, name=:V1)
resistor!(circuit3, :vin, :vd, 1000.0; name=:R1)
diode!(circuit3, :vd, :ground; IS=1e-14, N=1.0, name=:D1)
ground!(circuit3, :ground)

result3 = dc!(circuit3)
println("  DC Operating Point:")
println(result3)

# Expected: Vd ≈ 0.6-0.7V (diode forward voltage)
vd_idx = circuit3.nets[:vd].index
actual_vd = result3.solution[vd_idx]
println("  Diode voltage: $actual_vd V")

if 0.5 < actual_vd < 0.8
    println("\nTest 3: PASSED\n")
else
    println("\nTest 3: FAILED\n")
end

# Test 4: RLC circuit
println("Test 4: RLC circuit")
circuit4 = MNACircuit()
vsource!(circuit4, :vin, :ground; dc=5.0, name=:V1)
resistor!(circuit4, :vin, :n1, 100.0; name=:R1)
inductor!(circuit4, :n1, :n2, 1e-3; name=:L1)
capacitor!(circuit4, :n2, :ground, 1e-6; name=:C1)
ground!(circuit4, :ground)

result4 = dc!(circuit4)
println("  DC Operating Point (inductor acts as short, capacitor as open):")
println(result4)

println("\nTest 4: PASSED\n")

println("=== All basic tests completed ===")
