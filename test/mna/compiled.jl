#==============================================================================#
# Tests for Compiled Circuit Zero-Allocation Evaluation
#
# These tests verify that StaticCircuit and CPUCircuit achieve zero
# allocations during residual evaluation, which is critical for:
# - GPU ensemble simulation (parameter sweeps, Monte Carlo)
# - High-performance transient simulation
#==============================================================================#

using Test
using StaticArrays
using SparseArrays
using LinearAlgebra

# Import the module
using CedarSim.MNA

@testset "Compiled Circuit" begin

    #==========================================================================#
    # Test Fixtures
    #==========================================================================#

    """Build a simple RC circuit for testing."""
    function build_rc_test(params, spec; x=Float64[])
        ctx = MNAContext()

        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        R = get(params, :R, 1000.0)
        C = get(params, :C, 1e-6)

        # Voltage source at vcc
        stamp!(VoltageSource(1.0), ctx, vcc, 0)

        # Resistor from vcc to out
        stamp!(Resistor(R), ctx, vcc, out)

        # Capacitor from out to ground
        stamp!(Capacitor(C), ctx, out, 0)

        return ctx
    end

    """Build a voltage divider (pure resistive)."""
    function build_divider(params, spec; x=Float64[])
        ctx = MNAContext()

        vcc = get_node!(ctx, :vcc)
        out = get_node!(ctx, :out)

        R1 = get(params, :R1, 1000.0)
        R2 = get(params, :R2, 1000.0)
        V = get(params, :V, 5.0)

        # Voltage source
        stamp!(VoltageSource(V), ctx, vcc, 0)

        # R1: vcc to out
        stamp!(Resistor(R1), ctx, vcc, out)

        # R2: out to ground
        stamp!(Resistor(R2), ctx, out, 0)

        return ctx
    end

    """Build a 3-node ladder network."""
    function build_ladder(params, spec; x=Float64[])
        ctx = MNAContext()

        n1 = get_node!(ctx, :n1)
        n2 = get_node!(ctx, :n2)
        n3 = get_node!(ctx, :n3)

        R = get(params, :R, 1000.0)
        V = get(params, :V, 1.0)

        # Voltage source at n1
        stamp!(VoltageSource(V), ctx, n1, 0)

        # Resistors in series
        stamp!(Resistor(R), ctx, n1, n2)
        stamp!(Resistor(R), ctx, n2, n3)
        stamp!(Resistor(R), ctx, n3, 0)

        return ctx
    end

    #==========================================================================#
    # CPUCircuit Tests
    #==========================================================================#

    @testset "CPUCircuit compilation" begin
        params = (R=1000.0, C=1e-6)
        spec = MNASpec()
        ctx = build_rc_test(params, spec)

        circuit = compile_cpu_circuit(ctx)

        @test circuit.n == 3  # vcc, out, I_V1
        @test size(circuit.G) == (3, 3)
        @test size(circuit.C) == (3, 3)
        @test length(circuit.b) == 3
    end

    @testset "CPUCircuit DC solve" begin
        params = (R1=1000.0, R2=1000.0, V=5.0)
        spec = MNASpec()
        ctx = build_divider(params, spec)

        circuit = compile_cpu_circuit(ctx)
        x = solve_dc_cpu(circuit)

        # Voltage divider: V_out = V * R2/(R1+R2) = 5 * 0.5 = 2.5
        @test x[2] ≈ 2.5 atol=1e-10
    end

    @testset "CPUCircuit residual" begin
        params = (R=1000.0, C=1e-6)
        spec = MNASpec()
        ctx = build_rc_test(params, spec)

        circuit = compile_cpu_circuit(ctx)
        n = circuit.n

        u = zeros(n)
        du = zeros(n)
        resid = zeros(n)

        # Should not error
        cpu_residual!(resid, du, u, circuit, 0.0)

        # At u=0, du=0: F = -b
        @test resid ≈ -circuit.b
    end

    #==========================================================================#
    # StaticCircuit Tests
    #==========================================================================#

    @testset "StaticCircuit compilation" begin
        params = (R1=1000.0, R2=1000.0, V=5.0)
        spec = MNASpec()
        ctx = build_divider(params, spec)

        # System has 3 unknowns: vcc, out, I_V1
        circuit = compile_static_circuit(ctx, Val(3))

        @test circuit.G isa SMatrix{3,3,Float64}
        @test circuit.C isa SMatrix{3,3,Float64}
        @test circuit.b isa SVector{3,Float64}
    end

    @testset "StaticCircuit DC solve" begin
        params = (R1=1000.0, R2=1000.0, V=5.0)
        spec = MNASpec()
        ctx = build_divider(params, spec)

        circuit = compile_static_circuit(ctx, Val(3))
        x = solve_dc_static(circuit)

        @test x isa SVector{3,Float64}
        # V_out = V * R2/(R1+R2) = 2.5
        @test x[2] ≈ 2.5 atol=1e-10
    end

    @testset "StaticCircuit residual - zero allocation" begin
        params = (R1=1000.0, R2=1000.0, V=5.0)
        spec = MNASpec()
        ctx = build_divider(params, spec)

        circuit = compile_static_circuit(ctx, Val(3))

        u = @SVector [5.0, 2.5, -0.0025]
        du = @SVector zeros(3)

        # Warmup
        resid = static_residual(du, u, circuit, 0.0)

        @test resid isa SVector{3,Float64}

        # Test zero allocation with explicit type to avoid closure capture issues
        function test_alloc_3(c::StaticCircuit{3,Float64,9}, n::Int)
            total = 0
            for _ in 1:n
                u = @SVector [5.0, 2.5, -0.0025]
                du = @SVector zeros(3)
                r = static_residual(du, u, c, 0.0)
                total += Int(round(r[1] * 1000))
            end
            return total
        end
        test_alloc_3(circuit, 10)  # warmup
        test_alloc_3(circuit, 10)
        # Run 100 iterations, should be 0 bytes total (amortized 0 per call)
        allocs = @allocated test_alloc_3(circuit, 100)
        @test allocs == 0
    end

    @testset "StaticCircuit ladder - zero allocation" begin
        params = (R=1000.0, V=1.0)
        spec = MNASpec()
        ctx = build_ladder(params, spec)

        # 4 unknowns: n1, n2, n3, I_V1
        circuit = compile_static_circuit(ctx, Val(4))

        u = @SVector [1.0, 0.667, 0.333, 0.0]
        du = @SVector zeros(4)

        # Warmup
        resid = static_residual(du, u, circuit, 0.0)

        @test resid isa SVector{4,Float64}

        # Test zero allocation with explicit type
        function test_alloc_4(c::StaticCircuit{4,Float64,16}, n::Int)
            total = 0
            for _ in 1:n
                u = @SVector [1.0, 0.667, 0.333, 0.0]
                du = @SVector zeros(4)
                r = static_residual(du, u, c, 0.0)
                total += Int(round(r[1] * 1000))
            end
            return total
        end
        test_alloc_4(circuit, 10)  # warmup
        test_alloc_4(circuit, 10)
        allocs = @allocated test_alloc_4(circuit, 100)
        @test allocs == 0
    end

    @testset "StaticCircuit with capacitor - zero allocation" begin
        params = (R=1000.0, C=1e-6)
        spec = MNASpec()
        ctx = build_rc_test(params, spec)

        circuit = compile_static_circuit(ctx, Val(3))

        u = @SVector [1.0, 0.5, -0.0005]
        du = @SVector [0.0, 1000.0, 0.0]  # Capacitor charging

        # Warmup
        resid = static_residual(du, u, circuit, 0.0)

        @test resid isa SVector{3,Float64}

        # Test zero allocation with explicit type
        function test_alloc_rc(c::StaticCircuit{3,Float64,9}, n::Int)
            total = 0
            for _ in 1:n
                u = @SVector [1.0, 0.5, -0.0005]
                du = @SVector [0.0, 1000.0, 0.0]
                r = static_residual(du, u, c, 0.0)
                total += Int(round(r[1] * 1000))
            end
            return total
        end
        test_alloc_rc(circuit, 10)  # warmup
        test_alloc_rc(circuit, 10)
        allocs = @allocated test_alloc_rc(circuit, 100)
        @test allocs == 0
    end

    #==========================================================================#
    # Conversion Tests
    #==========================================================================#

    @testset "CPUCircuit to StaticCircuit conversion" begin
        params = (R1=1000.0, R2=1000.0, V=5.0)
        spec = MNASpec()
        ctx = build_divider(params, spec)

        cpu = compile_cpu_circuit(ctx)
        static = to_static(cpu, Val(3))

        @test static isa StaticCircuit{3,Float64}

        # Results should match
        x_cpu = solve_dc_cpu(cpu)
        x_static = solve_dc_static(static)

        @test x_cpu ≈ Vector(x_static)
    end

    #==========================================================================#
    # Float32 Support
    #==========================================================================#

    @testset "Float32 StaticCircuit" begin
        params = (R1=1000.0, R2=1000.0, V=5.0)
        spec = MNASpec()
        ctx = build_divider(params, spec)

        circuit = compile_static_circuit(ctx, Val(3), Float32)

        @test circuit.G isa SMatrix{3,3,Float32}
        @test circuit.b isa SVector{3,Float32}

        u = @SVector Float32[5.0, 2.5, -0.0025]
        du = @SVector zeros(Float32, 3)

        resid = static_residual(du, u, circuit, 0.0f0)

        @test resid isa SVector{3,Float32}

        # Zero allocation with explicit type
        function test_alloc_f32(c::StaticCircuit{3,Float32,9}, n::Int)
            total = 0
            for _ in 1:n
                u = @SVector Float32[5.0, 2.5, -0.0025]
                du = @SVector zeros(Float32, 3)
                r = static_residual(du, u, c, 0.0f0)
                total += Int(round(r[1] * 1000))
            end
            return total
        end
        test_alloc_f32(circuit, 10)  # warmup
        test_alloc_f32(circuit, 10)
        allocs = @allocated test_alloc_f32(circuit, 100)
        @test allocs == 0
    end

    #==========================================================================#
    # In-Place MVector Variant
    #==========================================================================#

    @testset "StaticCircuit in-place (MVector)" begin
        params = (R1=1000.0, R2=1000.0, V=5.0)
        spec = MNASpec()
        ctx = build_divider(params, spec)

        circuit = compile_static_circuit(ctx, Val(3))

        u = @SVector [5.0, 2.5, -0.0025]
        du = @SVector zeros(3)
        resid = @MVector zeros(3)

        static_residual!(resid, du, u, circuit, 0.0)

        # Result should match out-of-place
        resid_oop = static_residual(du, u, circuit, 0.0)
        @test SVector(resid) ≈ resid_oop
    end

    #==========================================================================#
    # Parameter Variations (Ensemble-style)
    #==========================================================================#

    @testset "Parameter sweep simulation" begin
        spec = MNASpec()

        # Simulate parameter sweep by compiling multiple circuits
        circuits = map([500.0, 1000.0, 2000.0]) do R1
            params = (R1=R1, R2=1000.0, V=5.0)
            ctx = build_divider(params, spec)
            compile_static_circuit(ctx, Val(3))
        end

        # Solve each
        solutions = [solve_dc_static(c) for c in circuits]

        # Verify expected voltages
        @test solutions[1][2] ≈ 5.0 * 1000/(500+1000) atol=1e-10   # 3.33
        @test solutions[2][2] ≈ 5.0 * 1000/(1000+1000) atol=1e-10  # 2.50
        @test solutions[3][2] ≈ 5.0 * 1000/(2000+1000) atol=1e-10  # 1.67
    end

    @testset "Ensemble residual - all zero allocation" begin
        spec = MNASpec()

        # Create multiple circuits (as would be done for GPU ensemble)
        circuits = map(1:10) do i
            params = (R1=1000.0*i, R2=1000.0, V=5.0)
            ctx = build_divider(params, spec)
            compile_static_circuit(ctx, Val(3))
        end

        # Test zero allocation with explicit type to avoid closure capture
        function test_ensemble_alloc(circuits::Vector{StaticCircuit{3,Float64,9}})
            total = 0
            for c in circuits
                u = @SVector [5.0, 2.5, -0.0025]
                du = @SVector zeros(3)
                r = static_residual(du, u, c, 0.0)
                # Use r to prevent dead code elimination
                total += Int(round(r[1] * 1000))
            end
            return total
        end

        # Warmup
        test_ensemble_alloc(circuits)
        test_ensemble_alloc(circuits)

        # Measure allocations
        allocs = @allocated test_ensemble_alloc(circuits)
        @test allocs == 0
    end

end # testset "Compiled Circuit"

println("Compiled circuit tests passed!")
