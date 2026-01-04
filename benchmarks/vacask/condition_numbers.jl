#!/usr/bin/env julia
#==============================================================================#
# VACASK Jacobian Condition Number Analysis
#
# Computes condition numbers for the Jacobian matrices (G + γC) of each
# VACASK benchmark circuit. This measures system stiffness.
#
# Usage:
#   julia --project=. benchmarks/vacask/condition_numbers.jl
#==============================================================================#

using Pkg
Pkg.instantiate()

using CedarSim
using CedarSim.MNA
using CedarSim.MNA: system_size
using VerilogAParser
using LinearAlgebra
using SparseArrays
using Printf

const BENCHMARK_DIR = @__DIR__

#==============================================================================#
# Helper functions
#==============================================================================#

"""
    compute_dae_condition(G, C, γ=1.0)

Compute condition number of DAE Jacobian J = G + γC.
Uses 2-norm for accuracy.
"""
function compute_dae_condition(G::AbstractMatrix, C::AbstractMatrix, γ::Real=1.0)
    J = Matrix(G + γ * C)
    n = size(J, 1)

    if n == 0
        return (cond=1.0, rank_deficiency=0, n=0)
    end

    # Check for singular matrix
    try
        cnd = cond(J, 2)
        rank_J = rank(J)
        return (cond=cnd, rank_deficiency=n - rank_J, n=n)
    catch e
        return (cond=Inf, rank_deficiency=n, n=n)
    end
end

"""
    analyze_matrix_structure(G, C)

Analyze the structure of G and C matrices.
"""
function analyze_matrix_structure(G::AbstractMatrix, C::AbstractMatrix)
    n = size(G, 1)
    n == 0 && return (n=0, G_nnz=0, G_density=0.0, G_diag_dominance=0.0,
                       G_diag_min=0.0, G_diag_max=0.0, G_diag_ratio=0.0,
                       C_nnz=0, C_density=0.0, C_diag_min=0.0, C_diag_max=0.0, C_diag_ratio=0.0,
                       n_differential=0, n_algebraic=0)

    # G matrix analysis
    G_nnz = nnz(G)
    G_density = G_nnz / (n * n)
    G_diag_dominance = sum(abs.(diag(G)) .>= sum(abs.(G), dims=2)[:] .- abs.(diag(G))) / n

    # C matrix analysis
    C_nnz = nnz(C)
    C_density = C_nnz / (n * n)

    # Check for differential vs algebraic variables
    C_row_sums = sum(abs.(Matrix(C)), dims=2)[:]
    n_differential = count(x -> x > 1e-30, C_row_sums)
    n_algebraic = n - n_differential

    # Scale analysis - look at min/max of diagonal elements
    G_diag = abs.(diag(G))
    G_diag_nonzero = filter(x -> x > 1e-30, G_diag)
    if !isempty(G_diag_nonzero)
        G_diag_min = minimum(G_diag_nonzero)
        G_diag_max = maximum(G_diag)
        G_diag_ratio = G_diag_max / G_diag_min
    else
        G_diag_min = G_diag_max = G_diag_ratio = 0.0
    end

    C_diag = abs.(diag(C))
    C_diag_nonzero = filter(x -> x > 1e-30, C_diag)
    if !isempty(C_diag_nonzero)
        C_diag_min = minimum(C_diag_nonzero)
        C_diag_max = maximum(C_diag)
        C_diag_ratio = C_diag_max / C_diag_min
    else
        C_diag_min = C_diag_max = C_diag_ratio = 0.0
    end

    return (
        n = n,
        G_nnz = G_nnz,
        G_density = G_density,
        G_diag_dominance = G_diag_dominance,
        G_diag_min = G_diag_min,
        G_diag_max = G_diag_max,
        G_diag_ratio = G_diag_ratio,
        C_nnz = C_nnz,
        C_density = C_density,
        C_diag_min = C_diag_min,
        C_diag_max = C_diag_max,
        C_diag_ratio = C_diag_ratio,
        n_differential = n_differential,
        n_algebraic = n_algebraic,
    )
end

"""
    detect_charge_states(sys::MNASystem)

Detect which equations correspond to charge state variables.
Returns indices of charge variables.
"""
function detect_charge_states(sys::MNASystem)
    n = system_size(sys)
    charge_indices = Int[]

    # Charge variables have the pattern:
    # G[i,i] = 1 (constraint equation)
    # C[i,i] = 0 (algebraic, not differential)
    # C[j,i] = ±1 for some j (KCL coupling)

    for i in 1:n
        G_ii = sys.G[i,i]
        C_ii = abs(sys.C[i,i])

        # Check if this looks like a charge constraint
        if abs(G_ii - 1.0) < 1e-10 && C_ii < 1e-20
            # Check for KCL coupling (C[j,i] = ±1)
            has_kcl = false
            for j in 1:n
                if j != i && abs(abs(sys.C[j,i]) - 1.0) < 1e-10
                    has_kcl = true
                    break
                end
            end
            if has_kcl
                push!(charge_indices, i)
            end
        end
    end

    return charge_indices
end

#==============================================================================#
# Benchmark circuit loaders
#==============================================================================#

function load_rc_circuit()
    println("\n=== RC Circuit ===")
    spice_file = joinpath(BENCHMARK_DIR, "rc", "cedarsim", "runme.sp")
    spice_code = read(spice_file, String)
    circuit_code = parse_spice_to_mna(spice_code; circuit_name=:rc_circuit_cond)

    eval(circuit_code)
    builder = getfield(Main, :rc_circuit_cond)

    # Just assemble at t=0 without DC solve to avoid world age issues
    spec = MNASpec(mode=:dcop)
    ctx = Base.invokelatest(builder, (;), spec)
    sys = MNA.assemble!(ctx)

    # Simple DC solve
    u0 = sys.G \ sys.b

    return sys, u0
end

function load_graetz_circuit()
    println("\n=== Graetz Bridge ===")

    # Load diode model
    diode_va_path = joinpath(BENCHMARK_DIR, "..", "..", "test", "vadistiller", "models", "diode.va")
    va = VerilogAParser.parsefile(diode_va_path)
    if !va.ps.errored
        Core.eval(Main, CedarSim.make_mna_module(va))
    else
        error("Failed to parse diode VA model")
    end

    spice_file = joinpath(BENCHMARK_DIR, "graetz", "cedarsim", "runme.sp")
    spice_code = read(spice_file, String)
    circuit_code = parse_spice_to_mna(spice_code; circuit_name=:graetz_circuit_cond,
                                       imported_hdl_modules=[Main.sp_diode_module])

    eval(circuit_code)
    builder = getfield(Main, :graetz_circuit_cond)

    spec = MNASpec(mode=:dcop)
    ctx = Base.invokelatest(builder, (;), spec)
    sys = MNA.assemble!(ctx)
    u0 = sys.G \ sys.b

    return sys, u0
end

function load_mul_circuit()
    println("\n=== Voltage Multiplier ===")

    # Load diode model (may already be loaded)
    diode_va_path = joinpath(BENCHMARK_DIR, "..", "..", "test", "vadistiller", "models", "diode.va")
    if !isdefined(Main, :sp_diode_module)
        va = VerilogAParser.parsefile(diode_va_path)
        if !va.ps.errored
            Core.eval(Main, CedarSim.make_mna_module(va))
        else
            error("Failed to parse diode VA model")
        end
    end

    spice_file = joinpath(BENCHMARK_DIR, "mul", "cedarsim", "runme.sp")
    spice_code = read(spice_file, String)
    circuit_code = parse_spice_to_mna(spice_code; circuit_name=:mul_circuit_cond,
                                       imported_hdl_modules=[Main.sp_diode_module])

    eval(circuit_code)
    builder = getfield(Main, :mul_circuit_cond)

    spec = MNASpec(mode=:dcop)
    ctx = Base.invokelatest(builder, (;), spec)
    sys = MNA.assemble!(ctx)
    u0 = sys.G \ sys.b

    return sys, u0
end

function load_ring_circuit()
    println("\n=== Ring Oscillator ===")

    # Load PSP103 model
    psp103_path = joinpath(BENCHMARK_DIR, "..", "..", "test", "vadistiller", "models", "psp103v4", "psp103.va")
    if !isdefined(Main, :PSP103VA_module)
        va = VerilogAParser.parsefile(psp103_path)
        if !va.ps.errored
            Core.eval(Main, CedarSim.make_mna_module(va))
        else
            error("Failed to parse PSP103 VA model")
        end
    end

    spice_file = joinpath(BENCHMARK_DIR, "ring", "cedarsim", "runme.sp")
    spice_code = read(spice_file, String)
    circuit_code = parse_spice_to_mna(spice_code; circuit_name=:ring_circuit_cond,
                                       imported_hdl_modules=[Main.PSP103VA_module])

    eval(circuit_code)
    builder = getfield(Main, :ring_circuit_cond)

    spec = MNASpec(mode=:dcop)
    ctx = Base.invokelatest(builder, (;), spec)
    sys = MNA.assemble!(ctx)
    u0 = sys.G \ sys.b

    return sys, u0
end

#==============================================================================#
# Main analysis
#==============================================================================#

function analyze_circuit(name::String, sys::MNASystem, x::Vector{Float64})
    println("\n" * "="^60)
    println("Circuit: $name")
    println("="^60)

    G, C = sys.G, sys.C

    # Matrix structure analysis
    structure = analyze_matrix_structure(G, C)

    println("\nMatrix Structure:")
    println("  System size:       $(structure.n)")
    println("  G nnz:             $(structure.G_nnz) (density: $(round(structure.G_density * 100, digits=2))%)")
    println("  C nnz:             $(structure.C_nnz) (density: $(round(structure.C_density * 100, digits=2))%)")
    println("  Differential vars: $(structure.n_differential)")
    println("  Algebraic vars:    $(structure.n_algebraic)")

    println("\nDiagonal Element Range:")
    @printf("  G: [%.2e, %.2e] ratio=%.2e\n", structure.G_diag_min, structure.G_diag_max, structure.G_diag_ratio)
    @printf("  C: [%.2e, %.2e] ratio=%.2e\n", structure.C_diag_min, structure.C_diag_max, structure.C_diag_ratio)

    # Condition numbers for different γ values (representing different time steps)
    println("\nCondition Numbers (J = G + γC) - UNSCALED:")

    # γ = 1/dt, so:
    # dt = 1ns  → γ = 1e9
    # dt = 1μs  → γ = 1e6
    # dt = 1ms  → γ = 1e3
    # dt = 1s   → γ = 1

    γ_values = [0.0, 1e3, 1e6, 1e9, 1e12]
    γ_labels = ["DC (γ=0)", "γ=1e3 (dt=1ms)", "γ=1e6 (dt=1μs)", "γ=1e9 (dt=1ns)", "γ=1e12 (dt=1ps)"]

    cond_results = Dict{Float64, NamedTuple}()
    for (γ, label) in zip(γ_values, γ_labels)
        result = compute_dae_condition(G, C, γ)
        cond_results[γ] = result
        if result.rank_deficiency > 0
            @printf("  %-20s  cond=%.2e (rank deficient by %d)\n", label, result.cond, result.rank_deficiency)
        else
            @printf("  %-20s  cond=%.2e\n", label, result.cond)
        end
    end

    # Detect charge states
    charge_idx = detect_charge_states(sys)
    if !isempty(charge_idx)
        println("\nCharge State Variables: $(length(charge_idx)) detected")
    end

    # ========== CHARGE SCALING ANALYSIS ==========
    println("\n--- Charge Scaling Analysis ---")

    # Method 1: Row equilibration (normalize row norms)
    D_row = MNA.compute_row_scaling(sys)
    scaled_sys_row = MNA.apply_row_scaling(sys, D_row)

    println("\nMethod 1: Row Equilibration")
    @printf("  Scaling range: [%.2e, %.2e]\n", minimum(D_row), maximum(D_row))

    # Method 2: Charge-aware row scaling (normalize Jacobian diagonal)
    γ_repr = 1e9  # Typical timestep of 1ns
    D_charge = MNA.compute_charge_row_scaling(sys; γ=γ_repr)
    scaled_sys_charge = MNA.apply_row_scaling(sys, D_charge)

    println("\nMethod 2: Charge Row Scaling (for γ=1e9)")
    @printf("  Scaling range: [%.2e, %.2e]\n", minimum(D_charge), maximum(D_charge))

    # Compare condition numbers
    println("\nCondition Numbers - SCALED vs UNSCALED:")
    println("  γ                    Unscaled      Row-Equil      Charge-Row     Improvement")
    println("  " * "-"^80)

    scaled_cond_results = Dict{Float64, NamedTuple}()
    for (γ, label) in zip(γ_values, γ_labels)
        cond_unscaled = cond_results[γ].cond
        cond_row = MNA.dae_condition_number(scaled_sys_row, γ)
        cond_charge = MNA.dae_condition_number(scaled_sys_charge, γ)

        # Best improvement
        best_scaled = min(cond_row, cond_charge)
        improvement = cond_unscaled / best_scaled

        scaled_cond_results[γ] = (unscaled=cond_unscaled, row=cond_row, charge=cond_charge, improvement=improvement)

        @printf("  %-20s  %.2e      %.2e       %.2e       %.1fx\n",
                label, cond_unscaled, cond_row, cond_charge, improvement)
    end

    return (name=name, structure=structure, charge_indices=charge_idx,
            conditions=cond_results, scaled_conditions=scaled_cond_results)
end

#==============================================================================#
# Run analysis
#==============================================================================#

function main()
    println("="^60)
    println("VACASK Jacobian Condition Number Analysis")
    println("="^60)

    results = []

    # RC Circuit (simple, linear)
    try
        sys, x = load_rc_circuit()
        push!(results, analyze_circuit("RC Circuit", sys, x))
    catch e
        @error "RC Circuit: Failed" exception=(e, catch_backtrace())
    end

    # Graetz Bridge (4 diodes)
    try
        sys, x = load_graetz_circuit()
        push!(results, analyze_circuit("Graetz Bridge", sys, x))
    catch e
        @error "Graetz Bridge: Failed" exception=(e, catch_backtrace())
    end

    # Voltage Multiplier (4 diodes, 4 caps)
    try
        sys, x = load_mul_circuit()
        push!(results, analyze_circuit("Voltage Multiplier", sys, x))
    catch e
        @error "Voltage Multiplier: Failed" exception=(e, catch_backtrace())
    end

    # Ring Oscillator (PSP103 MOSFETs) - large circuit, skip for now
    # try
    #     sys, x = load_ring_circuit()
    #     push!(results, analyze_circuit("Ring Oscillator", sys, x))
    # catch e
    #     @error "Ring Oscillator: Failed" exception=(e, catch_backtrace())
    # end

    println("\n" * "="^60)
    println("Summary")
    println("="^60)

    println("\nBaseline (unscaled) condition numbers:")
    for r in results
        @printf("  %-20s  n=%-4d  cond(G)=%.2e  cond(G+γC)=%.2e (γ=1e9)\n",
                r.name, r.structure.n,
                r.conditions[0.0].cond, r.conditions[1e9].cond)
    end

    println("\nImprovement from row scaling (at γ=1e9):")
    for r in results
        sc = r.scaled_conditions[1e9]
        @printf("  %-20s  Unscaled: %.2e → Best Scaled: %.2e  (%.1fx improvement)\n",
                r.name, sc.unscaled, min(sc.row, sc.charge), sc.improvement)
    end

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
