#==============================================================================#
# Tests for Compiled Circuit Optimization (CompiledStructure + EvalWorkspace)
#==============================================================================#

using Test
using SparseArrays
using LinearAlgebra

# Import MNA module - use CedarSim.MNA
using CedarSim.MNA: MNAContext, get_node!, resolve_index, get_rhs, reset_for_restamping!
using CedarSim.MNA: stamp!, Resistor, Capacitor, VoltageSource, Diode
using CedarSim.MNA: MNASpec, DCSolution, solve_dc, voltage, current
using CedarSim.MNA: compile_structure, create_workspace, EvalWorkspace, CompiledStructure
using CedarSim.MNA: fast_residual!, fast_rebuild!, system_size
using CedarSim.MNA: MNACircuit, compile
import CedarSim.MNA as MNA

@testset "COO to CSC Mapping" begin
    # Simple 3x3 sparse matrix
    I = [1, 2, 1, 3, 2]
    J = [1, 1, 2, 2, 3]
    V = [1.0, 2.0, 3.0, 4.0, 5.0]

    S = sparse(I, J, V, 3, 3)

    mapping = MNA.compute_coo_to_nz_mapping(I, J, S)

    # Verify mapping by updating values
    nz = nonzeros(S)
    fill!(nz, 0.0)

    for k in 1:length(V)
        if mapping[k] > 0
            nz[mapping[k]] += V[k]
        end
    end

    # Check that values match original
    S2 = sparse(I, J, V, 3, 3)
    @test S ≈ S2
end

@testset "COO to CSC with duplicates" begin
    # Matrix with duplicate entries (same (i,j))
    I = [1, 1, 2]  # Two entries at (1,1)
    J = [1, 1, 2]
    V = [1.0, 2.0, 3.0]  # Should sum to 3.0 at (1,1)

    S = sparse(I, J, V, 2, 2)

    mapping = MNA.compute_coo_to_nz_mapping(I, J, S)

    # Both entries at (1,1) should map to same position
    @test mapping[1] == mapping[2]
    @test mapping[3] > 0

    # Update test
    nz = nonzeros(S)
    fill!(nz, 0.0)
    for k in 1:length(V)
        if mapping[k] > 0
            nz[mapping[k]] += V[k]
        end
    end

    @test S[1, 1] ≈ 3.0
    @test S[2, 2] ≈ 3.0
end

@testset "Compile simple RC circuit" begin
    function build_rc(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        vin = get_node!(ctx, :vin)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(params.V; name=:V1), ctx, vin, 0)
        stamp!(Resistor(params.R), ctx, vin, out)
        stamp!(Capacitor(params.C), ctx, out, 0)

        return ctx
    end

    # Use MNACircuit + compile() which returns EvalWorkspace
    circuit = MNACircuit(build_rc; V=5.0, R=1000.0, C=1e-6)
    ws = compile(circuit)
    cs = ws.structure

    @test system_size(ws) == 3  # vin, out, I_V1
    @test cs.n_nodes == 2
    @test cs.n_currents == 1
    @test :vin in cs.node_names
    @test :out in cs.node_names

    # Check G matrix structure
    @test nnz(cs.G) > 0

    # Test fast rebuild
    u = zeros(3)
    fast_rebuild!(ws, u, 0.0)

    # G matrix should have entries for resistor and voltage source
    @test nnz(cs.G) >= 4  # At least resistor pattern
end

@testset "EvalWorkspace DC solution" begin
    function build_divider(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        vin = get_node!(ctx, :vin)
        mid = get_node!(ctx, :mid)

        stamp!(VoltageSource(params.V; name=:V1), ctx, vin, 0)
        stamp!(Resistor(params.R1), ctx, vin, mid)
        stamp!(Resistor(params.R2), ctx, mid, 0)

        return ctx
    end

    # Use MNACircuit + compile()
    circuit = MNACircuit(build_divider; V=10.0, R1=1000.0, R2=1000.0, spec=MNASpec(mode=:dcop))
    ws = compile(circuit)
    cs = ws.structure

    # DC solve using builder
    sol = solve_dc(cs.builder, cs.params, MNASpec(mode=:dcop))

    # Check voltage divider: V_mid = V * R2 / (R1 + R2) = 10 * 0.5 = 5.0
    @test voltage(sol, :mid) ≈ 5.0 atol=1e-10
    @test voltage(sol, :vin) ≈ 10.0 atol=1e-10
end

@testset "EvalWorkspace with DC solve" begin
    function build_rc(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        vin = get_node!(ctx, :vin)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(params.V; name=:V1), ctx, vin, 0)
        stamp!(Resistor(params.R), ctx, vin, out)
        stamp!(Capacitor(params.C), ctx, out, 0)

        return ctx
    end

    # Compile circuit
    circuit = MNACircuit(build_rc; V=5.0, R=1000.0, C=1e-6)
    ws = compile(circuit)
    cs = ws.structure

    @test system_size(ws) == 3

    # DC solve using builder
    sol = solve_dc(cs.builder, cs.params, MNASpec(mode=:dcop))
    @test voltage(sol, :vin) ≈ 5.0 atol=1e-10
    @test voltage(sol, :out) ≈ 5.0 atol=1e-10  # No load, so V_out = V_in
end

@testset "fast_residual! computation" begin
    function build_rc(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        vin = get_node!(ctx, :vin)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(params.V; name=:V1), ctx, vin, 0)
        stamp!(Resistor(params.R), ctx, vin, out)
        stamp!(Capacitor(params.C), ctx, out, 0)

        return ctx
    end

    circuit = MNACircuit(build_rc; V=5.0, R=1000.0, C=1e-6)
    ws = compile(circuit)
    cs = ws.structure

    n = system_size(ws)
    u = zeros(n)
    du = zeros(n)
    resid = zeros(n)

    # At DC operating point (u from DC solve), residual should be ~0
    dc_sol = solve_dc(cs.builder, cs.params, MNASpec(mode=:dcop))
    u .= dc_sol.x

    fast_residual!(resid, du, u, ws, 0.0)

    # Residual at DC operating point should be near zero (du=0)
    @test norm(resid) < 1e-10
end

@testset "Nonlinear circuit compilation" begin
    function build_diode_circuit(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        vin = get_node!(ctx, :vin)
        out = get_node!(ctx, :out)

        stamp!(VoltageSource(params.V; name=:V1), ctx, vin, 0)
        stamp!(Resistor(params.R), ctx, vin, out)
        stamp!(Diode(Is=1e-14, Vt=0.026), ctx, out, 0; x=x)

        return ctx
    end

    circuit = MNACircuit(build_diode_circuit; V=5.0, R=1000.0, spec=MNASpec(mode=:dcop))
    ws = compile(circuit)
    cs = ws.structure

    @test system_size(ws) == 3
    @test cs.n_nodes == 2

    # Test that structure matches at different operating points
    u0 = zeros(3)
    fast_rebuild!(ws, u0, 0.0)
    G_nnz_0 = nnz(cs.G)

    u1 = [5.0, 0.6, 0.0]  # Diode forward biased
    fast_rebuild!(ws, u1, 0.0)
    G_nnz_1 = nnz(cs.G)

    # Structure should be same (nonlinear just changes values)
    @test G_nnz_0 == G_nnz_1
end

@testset "Compiling different parameter sets" begin
    function build_r(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        vin = get_node!(ctx, :vin)

        stamp!(VoltageSource(params.V; name=:V1), ctx, vin, 0)
        stamp!(Resistor(params.R), ctx, vin, 0)

        return ctx
    end

    # Test that compiling different parameter sets works
    circuit1 = MNACircuit(build_r; V=10.0, R=1000.0, spec=MNASpec(mode=:dcop))
    circuit2 = MNACircuit(build_r; V=10.0, R=500.0, spec=MNASpec(mode=:dcop))

    ws1 = compile(circuit1)
    ws2 = compile(circuit2)

    sol1 = solve_dc(ws1.structure.builder, ws1.structure.params, MNASpec(mode=:dcop))
    sol2 = solve_dc(ws2.structure.builder, ws2.structure.params, MNASpec(mode=:dcop))

    # Current I = V/R
    I1 = current(sol1, :I_V1)
    I2 = current(sol2, :I_V1)

    @test I1 ≈ -10.0/1000.0 atol=1e-10  # Negative due to sign convention
    @test I2 ≈ -10.0/500.0 atol=1e-10
end

@testset "Performance comparison" begin
    function build_large_ladder(params, spec, t::Real=0.0; x=Float64[], ctx=nothing)
        if ctx === nothing
            ctx = MNAContext()
        else
            MNA.reset_for_restamping!(ctx)
        end
        N = params.N

        vin = get_node!(ctx, :vin)
        stamp!(VoltageSource(params.V; name=:V1), ctx, vin, 0)

        prev = vin
        for i in 1:N
            next = get_node!(ctx, Symbol(:n, i))
            stamp!(Resistor(params.R), ctx, prev, next)
            stamp!(Capacitor(params.C), ctx, next, 0)
            prev = next
        end

        return ctx
    end

    N = 50
    circuit = MNACircuit(build_large_ladder; V=5.0, R=1000.0, C=1e-12, N=N)
    ws = compile(circuit)

    @test system_size(ws) == N + 2  # N nodes + vin + current variable

    # Time multiple evaluations
    n = system_size(ws)
    u = rand(n)
    du = zeros(n)
    resid = zeros(n)

    # Warmup
    fast_residual!(resid, du, u, ws, 0.0)

    # Multiple iterations (simulating Newton steps)
    for i in 1:10
        fast_residual!(resid, du, u, ws, Float64(i) * 1e-9)
    end

    # Verify residual is finite (no NaN/Inf)
    @test all(isfinite, resid)
end

println("All precompile tests passed!")
