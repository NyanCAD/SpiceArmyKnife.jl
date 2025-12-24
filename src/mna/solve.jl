#==============================================================================#
# MNA Phase 1: Analysis Solvers
#
# This module provides analysis functions for the assembled MNA system:
# - DC Analysis: Steady-state solution (G*x = b)
# - AC Analysis: Small-signal frequency response ((G + jωC)*x = b)
# - Transient: ODEProblem formulation for DifferentialEquations.jl
#
# The solvers work with the MNASystem assembled from MNAContext.
#==============================================================================#

using LinearAlgebra
using SparseArrays
using Accessors
using ForwardDiff: Dual, value

# Extract real value from ForwardDiff.Dual (for tgrad compatibility)
# When ODE solvers compute time gradients, t is passed as a Dual number.
# We need to extract the real value to build circuits with Float64 vectors.
real_time(t::Real) = Float64(t)
real_time(t::Dual) = Float64(value(t))

export DCSolution, ACSolution
export solve_dc, solve_dc!, solve_ac
export make_ode_problem, make_dae_problem  # For static MNASystem
export voltage, current, magnitude_db, phase_deg

#==============================================================================#
# Simulation Specification
#==============================================================================#

"""
    MNASpec{T}

Simulation specification for MNA analysis.

Contains simulation-level parameters that are separate from circuit parameters.
Passed explicitly to circuit builders (not via ScopedValue) to enable JIT optimization.

# Fields
- `temp::Float64`: Temperature in Celsius (default: 27.0)
- `mode::Symbol`: Analysis mode - `:dcop`, `:tran`, `:tranop`, `:ac` (default: :tran)
- `time::T`: Current simulation time for transient sources (default: 0.0)

The time field is parameterized to support ForwardDiff automatic differentiation
during transient analysis with Rosenbrock ODE solvers.

# Example
```julia
spec = MNASpec(temp=50.0, mode=:dcop)
spec_at_1ms = MNASpec(temp=27.0, mode=:tran, time=1e-3)
```

# Design Rationale
Unlike CedarSim's ScopedValue-based SimSpec, MNASpec is passed explicitly.
This enables full JIT optimization since Julia's closure boxing issue
prevents optimization of captured ScopedValue accesses.
"""
Base.@kwdef struct MNASpec{T<:Real}
    temp::Float64 = 27.0
    mode::Symbol = :tran
    time::T = 0.0
end

export MNASpec

"""
    with_temp(spec::MNASpec, temp::Real) -> MNASpec

Create new spec with different temperature.
"""
with_temp(spec::MNASpec, temp::Real) = MNASpec(temp=Float64(temp), mode=spec.mode, time=spec.time)

"""
    with_mode(spec::MNASpec, mode::Symbol) -> MNASpec

Create new spec with different mode.
"""
with_mode(spec::MNASpec, mode::Symbol) = MNASpec(temp=spec.temp, mode=mode, time=spec.time)

"""
    with_time(spec::MNASpec, t::Real) -> MNASpec

Create new spec with different time.
Note: time type is preserved to support ForwardDiff Dual numbers.
"""
with_time(spec::MNASpec, t::T) where {T<:Real} = MNASpec(temp=spec.temp, mode=spec.mode, time=t)

export with_temp, with_mode, with_time

#==============================================================================#
# Solution Types
#==============================================================================#

"""
    DCSolution

Result of DC operating point analysis.

# Fields
- `x::Vector{Float64}`: Solution vector [V₁, V₂, ..., I₁, I₂, ...]
- `node_names::Vector{Symbol}`: Node names for interpretation
- `current_names::Vector{Symbol}`: Current variable names
- `n_nodes::Int`: Number of voltage nodes
"""
struct DCSolution
    x::Vector{Float64}
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    n_nodes::Int
end

"""
    DCSolution(sys::MNASystem, x::Vector{Float64})

Create a DC solution from a system and solution vector.
"""
DCSolution(sys::MNASystem, x::Vector{Float64}) =
    DCSolution(x, sys.node_names, sys.current_names, sys.n_nodes)

# Accessors
Base.getindex(sol::DCSolution, i::Int) = sol.x[i]
Base.length(sol::DCSolution) = length(sol.x)

"""
    voltage(sol::DCSolution, name::Symbol) -> Float64

Get the voltage at a node by name.
"""
function voltage(sol::DCSolution, name::Symbol)
    (name === :gnd || name === Symbol("0")) && return 0.0
    idx = findfirst(==(name), sol.node_names)
    idx === nothing && error("Unknown node: $name")
    return sol.x[idx]
end

"""
    voltage(sol::DCSolution, idx::Int) -> Float64

Get the voltage at a node by index (0 = ground).
"""
function voltage(sol::DCSolution, idx::Int)
    idx == 0 && return 0.0
    return sol.x[idx]
end

"""
    current(sol::DCSolution, name::Symbol) -> Float64

Get a current variable by name.
"""
function current(sol::DCSolution, name::Symbol)
    idx = findfirst(==(name), sol.current_names)
    idx === nothing && error("Unknown current: $name")
    return sol.x[sol.n_nodes + idx]
end

function Base.show(io::IO, sol::DCSolution)
    print(io, "DCSolution(")
    for (i, name) in enumerate(sol.node_names)
        i > 1 && print(io, ", ")
        @printf(io, "%s=%.4g", name, sol.x[i])
    end
    for (i, name) in enumerate(sol.current_names)
        print(io, ", ")
        @printf(io, "%s=%.4g", name, sol.x[sol.n_nodes + i])
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", sol::DCSolution)
    println(io, "DC Solution:")
    println(io, "  Node Voltages:")
    for (i, name) in enumerate(sol.node_names)
        @printf(io, "    V(%s) = %.6g V\n", name, sol.x[i])
    end
    if !isempty(sol.current_names)
        println(io, "  Branch Currents:")
        for (i, name) in enumerate(sol.current_names)
            @printf(io, "    %s = %.6g A\n", name, sol.x[sol.n_nodes + i])
        end
    end
end

#==============================================================================#
# AC Solution
#==============================================================================#

"""
    ACSolution

Result of AC small-signal analysis.

# Fields
- `freqs::Vector{Float64}`: Frequency points (Hz)
- `x::Vector{Vector{ComplexF64}}`: Solution at each frequency
- `node_names::Vector{Symbol}`: Node names
- `current_names::Vector{Symbol}`: Current variable names
- `n_nodes::Int`: Number of voltage nodes
"""
struct ACSolution
    freqs::Vector{Float64}
    x::Vector{Vector{ComplexF64}}
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
    n_nodes::Int
end

"""
    voltage(sol::ACSolution, name::Symbol) -> Vector{ComplexF64}

Get the complex voltage at a node across all frequencies.
"""
function voltage(sol::ACSolution, name::Symbol)
    (name === :gnd || name === Symbol("0")) && return zeros(ComplexF64, length(sol.freqs))
    idx = findfirst(==(name), sol.node_names)
    idx === nothing && error("Unknown node: $name")
    return [x[idx] for x in sol.x]
end

"""
    voltage(sol::ACSolution, name::Symbol, freq_idx::Int) -> ComplexF64

Get the complex voltage at a specific frequency index.
"""
function voltage(sol::ACSolution, name::Symbol, freq_idx::Int)
    (name === :gnd || name === Symbol("0")) && return 0.0 + 0.0im
    idx = findfirst(==(name), sol.node_names)
    idx === nothing && error("Unknown node: $name")
    return sol.x[freq_idx][idx]
end

"""
    magnitude_db(sol::ACSolution, name::Symbol) -> Vector{Float64}

Get the voltage magnitude in dB at a node.
"""
magnitude_db(sol::ACSolution, name::Symbol) = 20 .* log10.(abs.(voltage(sol, name)))

"""
    phase_deg(sol::ACSolution, name::Symbol) -> Vector{Float64}

Get the voltage phase in degrees at a node.
"""
phase_deg(sol::ACSolution, name::Symbol) = rad2deg.(angle.(voltage(sol, name)))

function Base.show(io::IO, sol::ACSolution)
    print(io, "ACSolution($(length(sol.freqs)) frequencies, ")
    print(io, "$(length(sol.node_names)) nodes)")
end

#==============================================================================#
# DC Analysis
#==============================================================================#

"""
    solve_dc(sys::MNASystem) -> DCSolution

Solve for DC operating point: G*x = b

For circuits with capacitors/inductors, this finds the steady-state
where all derivatives are zero (C*dx/dt = 0).
"""
function solve_dc(sys::MNASystem)
    n = system_size(sys)
    n == 0 && return DCSolution(Float64[], Symbol[], Symbol[], 0)

    # Solve G*x = b
    # Use \ which automatically selects appropriate solver
    x = sys.G \ sys.b

    return DCSolution(sys, x)
end

"""
    solve_dc!(x::Vector{Float64}, sys::MNASystem)

Solve DC operating point into pre-allocated vector x.
"""
function solve_dc!(x::Vector{Float64}, sys::MNASystem)
    n = system_size(sys)
    n == 0 && return x
    length(x) >= n || resize!(x, n)

    # Solve in-place using ldiv! if possible
    copyto!(view(x, 1:n), sys.b)
    F = lu(sys.G)
    ldiv!(F, view(x, 1:n))

    return x
end

"""
    solve_dc(ctx::MNAContext) -> DCSolution

Convenience function: assemble and solve in one step.
"""
function solve_dc(ctx::MNAContext)
    sys = assemble!(ctx)
    return solve_dc(sys)
end

#==============================================================================#
# Builder-Based DC Analysis (with Newton Iteration)
#==============================================================================#

using NonlinearSolve
using SciMLBase
using ADTypes

"""
    solve_dc(builder, params, spec::MNASpec;
             abstol=1e-10, maxiters=100) -> DCSolution

Solve DC operating point using a circuit builder function.

This is the recommended API for DC analysis. It supports both linear and
nonlinear devices. For linear circuits, it converges in one iteration.
For nonlinear devices (diodes, MOSFETs, VA devices with V*V terms), it uses
Newton iteration via NonlinearSolve.jl.

# Arguments
- `builder`: Circuit builder function `(params, spec; x=Float64[]) -> MNAContext`
- `params`: Circuit parameters (NamedTuple)
- `spec`: Simulation specification (MNASpec with mode=:dcop recommended)
- `abstol`: Convergence tolerance (default: 1e-10)
- `maxiters`: Maximum Newton iterations (default: 100)

# How It Works
1. Builds circuit at initial guess to get system size and structure
2. Solves linear system G*x = b as initial guess
3. Checks residual - if converged, returns immediately (linear case)
4. Otherwise, creates NonlinearProblem and iterates until ||F(u)|| < abstol

# Example
```julia
function build_circuit(params, spec; x=Float64[])
    ctx = MNAContext()
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(5.0; name=:V1), ctx, vcc, 0)
    stamp!(Resistor(1000.0), ctx, vcc, out)
    # For nonlinear devices, pass x to stamp:
    stamp!(MyNonlinearDevice(), ctx, out, 0; x=x)

    return ctx
end

sol = solve_dc(build_circuit, (;), MNASpec(mode=:dcop))
```

# See Also
- `solve_dc(sys::MNASystem)`: Direct linear solve on pre-assembled system
- `solve_dc(ctx::MNAContext)`: Assemble and solve in one step
"""
function solve_dc(builder::F, params::P, spec::MNASpec;
                  abstol::Real=1e-10, maxiters::Int=100) where {F,P}
    # Build at x=0 to get system size and initial structure
    ctx0 = builder(params, spec; x=Float64[])
    sys0 = assemble!(ctx0)
    n = system_size(sys0)

    if n == 0
        return DCSolution(Float64[], Symbol[], Symbol[], 0)
    end

    # Try linear solve first - if it satisfies tolerance, we're done
    x0 = sys0.G \ sys0.b

    # Check if linear solution is good enough
    ctx_check = builder(params, spec; x=x0)
    sys_check = assemble!(ctx_check)
    resid0 = sys_check.G * x0 - sys_check.b
    if norm(resid0) < abstol
        return DCSolution(sys_check, x0)
    end

    # Need Newton iteration - create NonlinearProblem
    # Residual function: F(u) = G(u)*u - b(u)
    function residual!(F, u, p)
        # Rebuild circuit at current operating point
        ctx = builder(params, spec; x=u)
        sys = assemble!(ctx)

        # F(u) = G(u)*u - b(u)
        mul!(F, sys.G, u)
        F .-= sys.b
        return nothing
    end

    # Create and solve NonlinearProblem
    nlprob = NonlinearProblem(residual!, x0)

    # Use RobustMultiNewton with finite differences (like dcop.jl)
    nlsolver = RobustMultiNewton(autodiff=AutoFiniteDiff())

    sol = solve(nlprob, nlsolver; abstol=abstol, maxiters=maxiters)

    if sol.retcode != SciMLBase.ReturnCode.Success
        @warn "Nonlinear DC solve did not converge: $(sol.retcode)"
    end

    # Get final system for node names
    ctx_final = builder(params, spec; x=sol.u)
    sys_final = assemble!(ctx_final)

    return DCSolution(sys_final, sol.u)
end

#==============================================================================#
# AC Analysis
#==============================================================================#

"""
    solve_ac(sys::MNASystem, freqs::AbstractVector{<:Real}) -> ACSolution

Solve AC small-signal analysis at given frequencies.

For each frequency f, solves: (G + j*2π*f*C) * x = b

This linearizes around the DC operating point, so DC analysis
should be performed first if the circuit contains nonlinear elements.
"""
function solve_ac(sys::MNASystem, freqs::AbstractVector{<:Real})
    n = system_size(sys)
    nf = length(freqs)

    results = Vector{Vector{ComplexF64}}(undef, nf)
    b_complex = complex.(sys.b)

    for (i, f) in enumerate(freqs)
        omega = 2π * f
        # Form Y = G + jωC
        Y = sys.G + (im * omega) * sys.C
        # Solve Y*x = b
        results[i] = Y \ b_complex
    end

    return ACSolution(collect(Float64, freqs), results,
                      sys.node_names, sys.current_names, sys.n_nodes)
end

"""
    solve_ac(sys::MNASystem; fstart, fstop, points_per_decade) -> ACSolution

Solve AC analysis with logarithmically spaced frequencies.
"""
function solve_ac(sys::MNASystem; fstart::Real, fstop::Real, points_per_decade::Int=10)
    # Generate log-spaced frequencies
    decades = log10(fstop / fstart)
    n_points = max(2, round(Int, decades * points_per_decade) + 1)
    freqs = 10 .^ range(log10(fstart), log10(fstop), length=n_points)
    return solve_ac(sys, freqs)
end

#==============================================================================#
# Transient Analysis: ODEProblem Formulation
#==============================================================================#

"""
    make_ode_function(sys::MNASystem) -> ODEFunction

Create an ODEFunction for use with DifferentialEquations.jl.

The MNA system G*x + C*dx/dt = b is converted to the form:
    C * dx/dt = b - G*x

This returns an ODEFunction with mass_matrix = C.

# Notes
- For singular C (algebraic constraints), use a DAE solver
- For constant C with nonzero diagonal, use implicit ODE solver
- For time-dependent sources, use `make_ode_function_timed` instead
"""
function make_ode_function(sys::MNASystem)
    G = sys.G
    C = sys.C
    b = sys.b
    n = system_size(sys)

    # Check if C has any structure
    has_dynamics = nnz(C) > 0

    if !has_dynamics
        # Pure algebraic system - warn and return constant function
        @warn "No capacitors/inductors - system is purely algebraic. Consider using solve_dc instead."
    end

    # RHS function: C * du/dt = b - G*u
    # We provide f!(du, u, p, t) where du represents C * dx/dt
    function rhs!(du, u, p, t)
        # du = b - G*u
        mul!(du, G, u)
        du .*= -1
        du .+= b
        return nothing
    end

    # Jacobian: d(rhs)/du = -G
    function jac!(J, u, p, t)
        copyto!(J, -G)
        return nothing
    end

    # Create ODEFunction with mass matrix
    # Using SciMLBase/DiffEqBase types
    return (
        rhs! = rhs!,
        jac! = jac!,
        mass_matrix = C,
        jac_prototype = -G  # Sparsity pattern
    )
end

"""
    make_ode_problem(sys::MNASystem, tspan::Tuple{Real,Real};
                     u0::Union{Nothing,Vector{Float64}}=nothing) -> NamedTuple

Create an ODEProblem-like structure for transient analysis.

Returns a NamedTuple with fields needed to construct an ODEProblem:
- `f`: RHS function
- `u0`: Initial condition
- `tspan`: Time span
- `mass_matrix`: Mass matrix C
- `jac`: Jacobian function
- `jac_prototype`: Sparsity pattern for Jacobian

# Usage with OrdinaryDiffEq
```julia
using OrdinaryDiffEq

prob_data = make_ode_problem(sys, (0.0, 1e-3))
f = ODEFunction(prob_data.f;
                mass_matrix = prob_data.mass_matrix,
                jac = prob_data.jac,
                jac_prototype = prob_data.jac_prototype)
prob = ODEProblem(f, prob_data.u0, prob_data.tspan)
sol = solve(prob, Rodas5())
```

# Arguments
- `sys::MNASystem`: The assembled MNA system
- `tspan`: Time span (tstart, tstop)
- `u0`: Initial condition (default: DC solution)
"""
function make_ode_problem(sys::MNASystem, tspan::Tuple{Real,Real};
                          u0::Union{Nothing,Vector{Float64}}=nothing)
    n = system_size(sys)

    # Default initial condition: DC solution
    if u0 === nothing
        dc_sol = solve_dc(sys)
        u0 = dc_sol.x
    end

    # Get ODE function components
    ode_funcs = make_ode_function(sys)

    return (
        f = ode_funcs.rhs!,
        u0 = u0,
        tspan = Float64.(tspan),
        mass_matrix = ode_funcs.mass_matrix,
        jac = ode_funcs.jac!,
        jac_prototype = ode_funcs.jac_prototype,
        sys = sys  # Keep reference for solution interpretation
    )
end

# NOTE: Builder-based make_ode_problem removed - use MNACircuit + ODEProblem instead

"""
    make_dae_function(sys::MNASystem) -> NamedTuple

Create a DAE function for use with DAE solvers (e.g., Sundials IDA).

The MNA system G*x + C*dx/dt = b is converted to implicit DAE form:
    F(du, u, p, t) = C*du + G*u - b = 0

This is useful when C is singular (has zero rows for algebraic equations).

# Returns
NamedTuple with:
- `f!`: Residual function F!(resid, du, u, p, t)
- `jac_du!`: Jacobian w.r.t. du (= C)
- `jac_u!`: Jacobian w.r.t. u (= G)
- `differential_vars`: Boolean vector indicating differential variables

# Usage with Sundials
```julia
using Sundials

dae_data = make_dae_function(sys)
prob = DAEProblem(dae_data.f!, dae_data.du0, dae_data.u0, tspan;
                  differential_vars = dae_data.differential_vars)
sol = solve(prob, IDA())
```
"""
function make_dae_function(sys::MNASystem)
    G = sys.G
    C = sys.C
    b = sys.b
    n = system_size(sys)

    # DAE residual: F = C*du + G*u - b = 0
    function dae_residual!(resid, du, u, p, t)
        # resid = C*du + G*u - b
        mul!(resid, C, du)        # resid = C*du
        mul!(resid, G, u, 1.0, 1.0)  # resid += G*u
        resid .-= b               # resid -= b
        return nothing
    end

    # Jacobian w.r.t. du: dF/d(du) = C
    function jac_du!(J, du, u, p, gamma, t)
        copyto!(J, C)
        return nothing
    end

    # Jacobian w.r.t. u: dF/du = G
    function jac_u!(J, du, u, p, gamma, t)
        copyto!(J, G)
        return nothing
    end

    # Determine which variables are differential (have nonzero C row)
    # A variable is differential if its corresponding row in C has nonzeros
    differential_vars = zeros(Bool, n)
    for j in 1:n
        for k in nzrange(C, j)
            i = rowvals(C)[k]
            differential_vars[i] = true
        end
    end

    return (
        f! = dae_residual!,
        jac_du! = jac_du!,
        jac_u! = jac_u!,
        differential_vars = differential_vars,
        C = C,
        G = G,
        b = b
    )
end

"""
    make_dae_problem(sys::MNASystem, tspan::Tuple{Real,Real};
                     u0::Union{Nothing,Vector{Float64}}=nothing) -> NamedTuple

Create a DAEProblem-like structure for transient analysis with DAE solvers.

# Arguments
- `sys::MNASystem`: The assembled MNA system
- `tspan`: Time span (tstart, tstop)
- `u0`: Initial condition (default: DC solution)

# Returns
NamedTuple with fields for DAEProblem construction.
"""
function make_dae_problem(sys::MNASystem, tspan::Tuple{Real,Real};
                          u0::Union{Nothing,Vector{Float64}}=nothing)
    n = system_size(sys)

    # Default initial condition: DC solution
    if u0 === nothing
        dc_sol = solve_dc(sys)
        u0 = dc_sol.x
    end

    # Get DAE function components
    dae_funcs = make_dae_function(sys)

    # Initial du from the DAE: C*du = b - G*u => du = C \ (b - G*u)
    # For consistent initialization, du should satisfy F(du, u, 0) = 0
    rhs = sys.b - sys.G * u0

    # Compute initial du (only for differential variables)
    # C is often singular (zero rows for algebraic equations)
    du0 = zeros(n)
    C = sys.C
    diff_vars = dae_funcs.differential_vars

    if any(diff_vars)
        # For each differential variable, compute its initial derivative
        # from the corresponding row of C*du = rhs
        C_dense = Matrix(C)
        for i in 1:n
            if diff_vars[i]
                # Find the diagonal element of C for this variable
                c_ii = C_dense[i, i]
                if abs(c_ii) > 1e-15
                    # Simple case: diagonal C element, du[i] = rhs[i] / c_ii
                    du0[i] = rhs[i] / c_ii
                else
                    # Off-diagonal case: solve row by row (simplified)
                    # For MNA, C is usually diagonal or block-diagonal
                    row_sum = sum(abs.(C_dense[i, :]))
                    if row_sum > 1e-15
                        # Use pseudoinverse for this row
                        du0[i] = rhs[i] / row_sum
                    end
                end
            end
            # Algebraic variables (diff_vars[i] == false) keep du0[i] = 0
        end
    end

    return (
        f! = dae_funcs.f!,
        u0 = u0,
        du0 = du0,
        tspan = Float64.(tspan),
        differential_vars = dae_funcs.differential_vars,
        jac_du! = dae_funcs.jac_du!,
        jac_u! = dae_funcs.jac_u!,
        sys = sys
    )
end

#==============================================================================#
# MNASim: Parameterized MNA Circuit Wrapper
#==============================================================================#

"""
    MNASim{F,P}

Parameterized MNA circuit simulation wrapper, similar to CedarSim's ParamSim.

# Fields
- `builder::F`: Function that builds the circuit given parameters
- `mode::Symbol`: Simulation mode (:tran, :dcop, :tranop)
- `params::P`: Named tuple of circuit parameters

# Modes
- `:tran` - Full transient simulation with time-dependent sources
- `:dcop` - DC operating point (time-dependent sources at t=0)
- `:tranop` - Transient operating point (steady state for tran)

# Example
```julia
function build_circuit(p)
    ctx = MNAContext()
    vcc = get_node!(ctx, :vcc)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(p.Vcc), ctx, vcc, 0)
    stamp!(Resistor(p.R), ctx, vcc, out)
    stamp!(Capacitor(p.C), ctx, out, 0)

    return ctx
end

sim = MNASim(build_circuit; Vcc=5.0, R=1000.0, C=1e-6)
sys = assemble!(sim)
sol = solve_dc(sys)
```

# New API with explicit spec:
```julia
function build_circuit(params, spec)
    ctx = MNAContext()
    # params: circuit parameters (R, C, Vcc, etc.)
    # spec: MNASpec with temp, mode

    R_temp = params.R * (1 + 0.004 * (spec.temp - 27))
    stamp!(Resistor(R_temp), ctx, ...)

    v = spec.mode == :dcop ? params.Vdc : params.Vss
    stamp!(VoltageSource(v), ctx, ...)

    return ctx
end

sim = MNASim(build_circuit; spec=MNASpec(temp=50.0), Vcc=5.0, R=1000.0)
```
"""
struct MNASim{F,S,P}
    builder::F
    spec::S        # MNASpec or compatible
    params::P      # Circuit parameters (NamedTuple)

    function MNASim(builder::F; spec=MNASpec(), mode::Symbol=:tran, temp::Real=27.0, kwargs...) where {F}
        # Support both new API (spec=...) and legacy (mode=...)
        if spec isa MNASpec
            actual_spec = spec
        else
            actual_spec = MNASpec(temp=Float64(temp), mode=mode)
        end
        params = NamedTuple(kwargs)
        new{F,typeof(actual_spec),typeof(params)}(builder, actual_spec, params)
    end
end

export MNASim, alter

# Callable interface - invokes builder with (params, spec)
function (sim::MNASim)()
    return sim.builder(sim.params, sim.spec)
end

# Build and assemble in one step
function assemble!(sim::MNASim)
    ctx = sim()
    return assemble!(ctx)
end

# Create a new sim with different spec
function with_spec(sim::MNASim, spec::MNASpec)
    return MNASim(sim.builder; spec=spec, sim.params...)
end

# Create a new sim with different mode (convenience)
function with_mode(sim::MNASim, mode::Symbol)
    new_spec = MNASpec(temp=sim.spec.temp, mode=mode)
    return with_spec(sim, new_spec)
end

# Create a new sim with different temperature (convenience)
function with_temp(sim::MNASim, temp::Real)
    new_spec = MNASpec(temp=Float64(temp), mode=sim.spec.mode)
    return with_spec(sim, new_spec)
end

# Create a new sim with altered circuit parameters
# Supports nested paths via var-strings: alter(sim; var"inner.R1" = 100.0)
function alter(sim::MNASim; kwargs...)
    # Helper to convert symbol to lens (handles nested paths like "a.b.c")
    function to_lens(selector::Symbol)
        parts = Symbol.(split(string(selector), "."))
        return Accessors.opticcompose(PropertyLens.(parts)...)
    end

    new_params = sim.params
    for (selector, value) in pairs(kwargs)
        if value === nothing
            continue  # Skip nothing values (sentinel for "use default")
        end
        lens = to_lens(selector)
        # Auto-convert numeric types to Float64 if the target is Float64
        if isa(value, Number) && isa(lens(new_params), Float64)
            value = Float64(value)
        end
        new_params = Accessors.set(new_params, lens, value)
    end

    return MNASim(sim.builder; spec=sim.spec, new_params...)
end

export with_spec

#==============================================================================#
# Out-of-Place Evaluation (GPU-Compatible)
#==============================================================================#

"""
    eval_circuit(builder, params, spec; t=0.0, u=nothing) -> MNASystem

Out-of-place circuit evaluation that returns a new MNASystem.

This is the GPU-compatible API that builds fresh matrices each call.
For ensemble GPU solving and parameter sweeps, this avoids mutation
and enables `ODEProblem{false}` formulation.

# Arguments
- `builder`: Circuit builder function `(params, spec) -> MNAContext`
- `params`: Circuit parameters (NamedTuple)
- `spec`: Simulation specification (MNASpec)
- `t`: Current time (for time-dependent sources)
- `u`: Current solution (for non-linear devices, Newton iteration)

# Example
```julia
function build_rc(params, spec)
    ctx = MNAContext()
    v = spec.mode == :dcop ? 0.0 : params.Vcc
    stamp!(VoltageSource(v), ctx, get_node!(ctx, :vcc), 0)
    stamp!(Resistor(params.R), ctx, get_node!(ctx, :vcc), get_node!(ctx, :out))
    stamp!(Capacitor(params.C), ctx, get_node!(ctx, :out), 0)
    return ctx
end

params = (Vcc=5.0, R=1000.0, C=1e-6)
spec = MNASpec(mode=:tran)
sys = eval_circuit(build_rc, params, spec)
```

# Design Notes
- Returns fresh matrices, no mutation
- Enables GPU ensemble solving with `EnsembleGPUKernel`
- JIT optimizes constant stamps to simple assignments
"""
function eval_circuit(builder::F, params::P, spec::MNASpec;
                      t::Real=0.0, u=nothing) where {F,P}
    # For now, t and u are passed through spec extension
    # Future: could pass to builder for time-dependent/nonlinear devices
    ctx = builder(params, spec)
    return assemble!(ctx)
end

export eval_circuit

"""
    eval_circuit(sim::MNASim; t=0.0, u=nothing) -> MNASystem

Out-of-place evaluation from MNASim wrapper.
"""
function eval_circuit(sim::MNASim; t::Real=0.0, u=nothing)
    return eval_circuit(sim.builder, sim.params, sim.spec; t=t, u=u)
end

"""
    solve_dc(sim::MNASim) -> DCSolution

Build circuit from sim and solve DC operating point.
"""
function solve_dc(sim::MNASim)
    sys = assemble!(sim)
    return solve_dc(sys)
end

"""
    solve_ac(sim::MNASim, freqs; kwargs...) -> ACSolution

Build circuit from sim and perform AC analysis.
"""
function solve_ac(sim::MNASim, freqs::AbstractVector{<:Real}; kwargs...)
    sys = assemble!(sim)
    return solve_ac(sys, freqs; kwargs...)
end

# NOTE: make_dc_initialized_* removed - use MNACircuit + DAEProblem/ODEProblem instead
# The new API automatically performs DC initialization.

#==============================================================================#
# Utility Functions
#==============================================================================#

"""
    check_singular(sys::MNASystem) -> Bool

Check if the G matrix is singular (no DC solution possible).
Returns true if singular.
"""
function check_singular(sys::MNASystem)
    n = system_size(sys)
    n == 0 && return false
    try
        F = lu(sys.G; check=false)
        return !issuccess(F)
    catch
        return true
    end
end

"""
    condition_number(sys::MNASystem) -> Float64

Compute the condition number of the G matrix.
Large values indicate ill-conditioning.
"""
function condition_number(sys::MNASystem)
    n = system_size(sys)
    n == 0 && return 1.0
    # For sparse matrices, compute via SVD of small systems or estimate
    if n <= 100
        return cond(Matrix(sys.G))
    else
        # For large systems, estimate using iterative methods
        # (simplified: just return norm ratio)
        return norm(sys.G, 1) * norm(inv(Matrix(sys.G)), 1)
    end
end

# Note: dc! and tran! for MNASim are defined in sweeps.jl to integrate
# with CedarSim's existing sweep API (CircuitSweep, ProductSweep, etc.)

#==============================================================================#
# Symbolic Solution Access
#==============================================================================#

"""
    MNASolutionAccessor

Provides symbolic access to ODE solution via node names.
Wraps an ODESolution with the MNASystem for name resolution.

# Example
```julia
sim = MNASim(build_circuit; Vcc=5.0, R=1000.0)
sol = tran!(sim, (0.0, 1e-3))
acc = MNASolutionAccessor(sol, assemble!(sim))
acc[:out]  # Voltage trajectory at node :out
```
"""
struct MNASolutionAccessor{S}
    sol::S
    sys::MNASystem
end

export MNASolutionAccessor

# Symbolic indexing: acc[:node_name]
function Base.getindex(acc::MNASolutionAccessor, name::Symbol)
    (name === :gnd || name === Symbol("0")) && return zeros(length(acc.sol.t))
    idx = findfirst(==(name), acc.sys.node_names)
    if idx !== nothing
        return [acc.sol(t)[idx] for t in acc.sol.t]
    end
    idx = findfirst(==(name), acc.sys.current_names)
    if idx !== nothing
        curr_idx = acc.sys.n_nodes + idx
        return [acc.sol(t)[curr_idx] for t in acc.sol.t]
    end
    error("Unknown variable: $name")
end

# Time access
Base.getproperty(acc::MNASolutionAccessor, s::Symbol) =
    s === :t ? acc.sol.t : getfield(acc, s)

# Interpolation
(acc::MNASolutionAccessor)(t::Real) = acc.sol(t)

"""
    voltage(acc::MNASolutionAccessor, name::Symbol, t::Real)

Get voltage at node at specific time.
"""
function voltage(acc::MNASolutionAccessor, name::Symbol, t::Real)
    (name === :gnd || name === Symbol("0")) && return 0.0
    idx = findfirst(==(name), acc.sys.node_names)
    idx === nothing && error("Unknown node: $name")
    return acc.sol(t)[idx]
end

"""
    voltage(acc::MNASolutionAccessor, name::Symbol)

Get voltage trajectory at node.
"""
voltage(acc::MNASolutionAccessor, name::Symbol) = acc[name]

#==============================================================================#
# Hierarchical Scope Access (for sol[sys.x1.r1] style)
#==============================================================================#

"""
    NodeRef

Reference to a node in the circuit hierarchy.
Enables `sol[sys.subcircuit.node]` style access.
"""
struct NodeRef
    path::Vector{Symbol}  # Hierarchical path
    name::Symbol          # Final node name
end

NodeRef(name::Symbol) = NodeRef(Symbol[], name)

export NodeRef

"""
    ScopedSystem

Wrapper that enables hierarchical access like `sys.x1.r1.p`.
Returns NodeRef objects that can be used to index solutions.

# Example
```julia
sys = assemble!(sim)
s = scope(sys)

# Access nodes hierarchically (assuming node :x1_out exists)
sol[s.x1.out]  # Accesses node :x1_out
sol[s.vcc]     # Accesses node :vcc
```
"""
struct ScopedSystem
    sys::MNASystem
    path::Vector{Symbol}
end

ScopedSystem(sys::MNASystem) = ScopedSystem(sys, Symbol[])

function Base.getproperty(s::ScopedSystem, name::Symbol)
    if name === :sys || name === :path
        return getfield(s, name)
    end
    # Check if this is a terminal node
    full_name = isempty(s.path) ? name : Symbol(join([s.path..., name], "_"))
    if full_name in s.sys.node_names || full_name in s.sys.current_names
        return NodeRef(s.path, name)
    end
    # Otherwise, extend the path for hierarchical access
    return ScopedSystem(s.sys, [s.path..., name])
end

export ScopedSystem

# Allow acc[NodeRef] access
function Base.getindex(acc::MNASolutionAccessor, ref::NodeRef)
    full_name = isempty(ref.path) ? ref.name : Symbol(join([ref.path..., ref.name], "_"))
    return acc[full_name]
end

function voltage(acc::MNASolutionAccessor, ref::NodeRef, t::Real)
    full_name = isempty(ref.path) ? ref.name : Symbol(join([ref.path..., ref.name], "_"))
    return voltage(acc, full_name, t)
end

"""
    scope(sys::MNASystem) -> ScopedSystem

Create a scoped view of the system for hierarchical node access.
"""
scope(sys::MNASystem) = ScopedSystem(sys)

export scope

#==============================================================================#
# MNACircuit: Phase 6 SciML DAE Integration
#==============================================================================#

"""
    MNACircuit{F,P,S}

Circuit definition for SciML DAE integration, following the System → Problem → Solution pattern.

This is the Phase 6 replacement for `CircuitIRODESystem`, providing
the same interface while using the MNA backend instead of DAECompiler.

# SciML Pattern
    MNACircuit (System) → DAEProblem (Problem) → solve() → Solution

# Architecture
- `builder`: Function `(params, spec; x=Float64[]) -> MNAContext`
- `params`: Circuit parameters (NamedTuple)
- `spec`: Base simulation spec (MNASpec)
- `tspan`: Time span for transient

# DAE Formulation
The MNA system G*x + C*dx/dt = b is converted to implicit DAE form:
    F(du, u, p, t) = C*du + G*u - b = 0

For nonlinear devices, G, C, and b are recomputed at each evaluation
by calling the builder with the current operating point.

# Usage
```julia
# Define circuit builder
function build_inverter(params, spec; x=Float64[])
    ctx = MNAContext()
    # ... stamp devices, passing x for nonlinear devices ...
    return ctx
end

# Create circuit system
circuit = MNACircuit(build_inverter, (Vdd=1.0,), MNASpec(), (0.0, 1e-6))

# Convert to SciML DAEProblem
using DifferentialEquations
prob = DAEProblem(circuit)
sol = solve(prob, IDA())
```

# See Also
- `make_dae_problem`: Lower-level DAE problem creation
- `CircuitIRODESystem`: DAECompiler equivalent (legacy)
"""
struct MNACircuit{F,P,S}
    builder::F
    params::P
    spec::S
    tspan::Tuple{Float64,Float64}
end

export MNACircuit

# Keep deprecated alias for backwards compatibility
const MNACircuitProblem = MNACircuit
export MNACircuitProblem

"""
    MNACircuit(builder, params, spec, tspan)

Create an MNA circuit system for SciML DAE solvers.

# Arguments
- `builder`: Circuit builder `(params, spec; x=Float64[]) -> MNAContext`
- `params`: Circuit parameters
- `spec`: Base simulation spec
- `tspan`: Time span `(t0, tf)`
"""
function MNACircuit(builder::F, params::P, spec::S, tspan::Tuple{<:Real,<:Real}) where {F,P,S}
    MNACircuit{F,P,S}(builder, params, spec, Float64.(tspan))
end

"""
    system_size(circuit::MNACircuit) -> Int

Get the system size (number of unknowns) for the circuit.
"""
function system_size(circuit::MNACircuit)
    ctx0 = circuit.builder(circuit.params, circuit.spec; x=Float64[])
    sys0 = assemble!(ctx0)
    return system_size(sys0)
end

"""
    make_dae_residual(circuit::MNACircuit) -> Function

Create the DAE residual function for the circuit.

For nonlinear circuits, this rebuilds G, C, b at each evaluation.
For linear circuits, the matrices are constant but b(t) may vary.

Returns `(resid, du, u, p, t) -> nothing` suitable for DAEProblem.
"""
function make_dae_residual(circuit::MNACircuit)
    builder = circuit.builder
    params = circuit.params
    base_spec = circuit.spec

    function dae_residual!(resid, du, u, p, t)
        # Build circuit at current operating point and time
        spec_t = MNASpec(temp=base_spec.temp, mode=:tran, time=real_time(t))
        ctx = builder(params, spec_t; x=u)
        sys = assemble!(ctx)

        # F(du, u) = C*du + G*u - b = 0
        mul!(resid, sys.C, du)
        mul!(resid, sys.G, u, 1.0, 1.0)
        resid .-= sys.b

        return nothing
    end

    return dae_residual!
end

"""
    make_dae_jacobian(circuit::MNACircuit) -> Function

Create the combined Jacobian function J = γ*C + G for DAE solvers.

For Sundials IDA, the Jacobian is ∂F/∂u + γ*∂F/∂(du) = G + γ*C.
"""
function make_dae_jacobian(circuit::MNACircuit)
    builder = circuit.builder
    params = circuit.params
    base_spec = circuit.spec

    function dae_jac!(J, du, u, p, gamma, t)
        # Build circuit at current operating point
        spec_t = MNASpec(temp=base_spec.temp, mode=:tran, time=real_time(t))
        ctx = builder(params, spec_t; x=u)
        sys = assemble!(ctx)

        # J = G + γ*C
        copyto!(J, sys.G)
        J .+= gamma .* sys.C

        return nothing
    end

    return dae_jac!
end

"""
    detect_differential_vars(circuit::MNACircuit) -> BitVector

Determine which variables are differential (have du terms in C*du).

Variables with nonzero rows in the C matrix are differential.
Variables with zero rows are algebraic (no time derivatives).
"""
function detect_differential_vars(circuit::MNACircuit)
    ctx0 = circuit.builder(circuit.params, circuit.spec; x=Float64[])
    sys0 = assemble!(ctx0)
    return detect_differential_vars(sys0)
end

"""
    detect_differential_vars(sys::MNASystem) -> BitVector

Determine which variables are differential from the C matrix structure.
"""
function detect_differential_vars(sys::MNASystem)
    n = system_size(sys)
    C = sys.C
    diff_vars = falses(n)

    for j in 1:n
        for k in nzrange(C, j)
            i = rowvals(C)[k]
            diff_vars[i] = true
        end
    end

    return diff_vars
end

"""
    compute_initial_conditions(circuit::MNACircuit) -> (u0, du0)

Compute consistent initial conditions via DC operating point.

1. Solves DC problem (du=0) to get u0
2. Computes du0 to satisfy F(du0, u0, 0) = 0

This is equivalent to CedarDCOp initialization.
"""
function compute_initial_conditions(circuit::MNACircuit)
    # DC solve for u0
    dc_spec = MNASpec(temp=circuit.spec.temp, mode=:dcop, time=0.0)
    u0 = solve_dc(circuit.builder, circuit.params, dc_spec).x

    n = length(u0)
    du0 = zeros(n)

    # At t=0, need F(du0, u0) = C*du0 + G*u0 - b = 0
    # So: C*du0 = b - G*u0
    # For singular C (algebraic vars), du0 components are 0

    ctx0 = circuit.builder(circuit.params, circuit.spec; x=u0)
    sys0 = assemble!(ctx0)

    rhs = sys0.b - sys0.G * u0
    diff_vars = detect_differential_vars(sys0)

    # Compute du0 for differential variables
    # Simple diagonal approximation (works for typical MNA)
    C_dense = Matrix(sys0.C)
    for i in 1:n
        if diff_vars[i]
            c_ii = C_dense[i, i]
            if abs(c_ii) > 1e-15
                du0[i] = rhs[i] / c_ii
            end
        end
    end

    return u0, du0
end

"""
    SciMLBase.DAEProblem(circuit::MNACircuit; kwargs...)

Convert MNACircuit to SciML DAEProblem.

# Keyword Arguments
- `u0`: Initial state (default: DC solution)
- `du0`: Initial derivatives (default: computed for consistency)
- `abstol`: Solver tolerance

# Example
```julia
circuit = MNACircuit(builder, params, spec, tspan)
prob = DAEProblem(circuit)
sol = solve(prob, IDA())
```
"""
function SciMLBase.DAEProblem(circuit::MNACircuit;
                               u0=nothing, du0=nothing, kwargs...)
    # Get initial conditions
    if u0 === nothing || du0 === nothing
        u0_computed, du0_computed = compute_initial_conditions(circuit)
        u0 = u0 === nothing ? u0_computed : u0
        du0 = du0 === nothing ? du0_computed : du0
    end

    # Create residual function
    residual! = make_dae_residual(circuit)

    # Detect differential variables
    diff_vars = detect_differential_vars(circuit)

    return SciMLBase.DAEProblem(
        residual!,
        du0,
        u0,
        circuit.tspan;
        differential_vars = diff_vars,
        kwargs...
    )
end

"""
    SciMLBase.ODEProblem(circuit::MNACircuit; kwargs...)

Convert MNACircuit to SciML ODEProblem with mass matrix formulation.

Uses mass matrix form: C * du/dt = b(t) - G*u
where C is the capacitance matrix (mass matrix) and the RHS is b - G*u.

# Keyword Arguments
- `u0`: Initial state (default: DC solution)

# Example
```julia
circuit = MNACircuit(builder, params, spec, tspan)
prob = ODEProblem(circuit)
sol = solve(prob, Rodas5P())  # Rosenbrock handles singular mass matrix
```

# Notes
- For nonlinear circuits with voltage-dependent capacitance, use DAEProblem instead
- ODEProblem with mass matrix works well for linear circuits with Rodas5P
- DAEProblem with IDA is recommended for large circuits and better algebraic handling
"""
function SciMLBase.ODEProblem(circuit::MNACircuit; u0=nothing, kwargs...)
    builder = circuit.builder
    params = circuit.params
    base_spec = circuit.spec

    # Get initial conditions via DC solve
    if u0 === nothing
        dc_spec = MNASpec(temp=base_spec.temp, mode=:dcop, time=0.0)
        u0 = solve_dc(builder, params, dc_spec).x
    end

    # Build initial system for structure and mass matrix
    ctx0 = builder(params, base_spec; x=u0)
    sys0 = assemble!(ctx0)

    # RHS function: C * du/dt = b(t) - G*u
    # Returns f = b - G*u (the RHS without mass matrix)
    function rhs!(du, u, p, t)
        spec_t = MNASpec(temp=base_spec.temp, mode=:tran, time=real_time(t))
        ctx = builder(params, spec_t; x=u)
        sys = assemble!(ctx)

        # du = b - G*u
        mul!(du, sys.G, u)
        du .*= -1
        du .+= sys.b
        return nothing
    end

    # Jacobian: d(rhs)/du = -G
    function jac!(J, u, p, t)
        spec_t = MNASpec(temp=base_spec.temp, mode=:tran, time=real_time(t))
        ctx = builder(params, spec_t; x=u)
        sys = assemble!(ctx)
        copyto!(J, -sys.G)
        return nothing
    end

    # Create ODEFunction with constant mass matrix C
    # For nonlinear voltage-dependent capacitors, use DAEProblem instead
    f = SciMLBase.ODEFunction(
        rhs!;
        mass_matrix = sys0.C,
        jac = jac!,
        jac_prototype = -sys0.G
    )

    return SciMLBase.ODEProblem(f, u0, circuit.tspan; kwargs...)
end

# NOTE: make_nonlinear_dae_* removed - use MNACircuit + DAEProblem instead
# MNACircuit automatically handles nonlinear devices by rebuilding matrices each step.
