#==============================================================================#
# Cedar DC Operating Point Initialization
#
# CedarDCOp is the recommended initialization algorithm for MNA circuits.
# It performs proper DC operating point analysis before transient simulation.
#
# UNIFIED DC SOLVE:
# Both dc!(circuit) and CedarDCOp initialization share the same core logic:
# - _dc_newton_compiled does Newton iteration to solve F(u) = G(u)*u - b(u) = 0
# This ensures consistent DC operating point results between standalone DC
# analysis and transient initialization.
#==============================================================================#

using Accessors: @set
using OrdinaryDiffEq
using OrdinaryDiffEq.OrdinaryDiffEqCore: ODEIntegrator
using OrdinaryDiffEq: BrownFullBasicInit, CheckInit, NoInit, DImplicitEuler
using LinearAlgebra
using NonlinearSolve
using SciMLBase
using SciMLBase: ReturnCode
using DiffEqBase
using ADTypes
using Sundials

export CedarDCOp, CedarTranOp, CedarUICOp

"""
    CedarDCOp <: DiffEqBase.DAEInitializationAlgorithm

Initialization algorithm for MNA circuits that:
1. For Sundials IDA: Switches to :dcop mode, solves DC steady state with
   robust nonlinear solver, then calls DefaultInit (CheckInit) or ShampineCollocationInit
2. For OrdinaryDiffEq: Same DC solve followed by DefaultInit or ShampineCollocationInit

This is the recommended initialization algorithm for circuits with nonlinear
devices (diodes, MOSFETs, voltage-dependent capacitors).

# Arguments
- `abstol`: Tolerance for the DC solve (default: 1e-9)
- `maxiters`: Maximum Newton iterations (default: 500). VA models with internal
  nodes (like BJT excess phase) may need more iterations to converge at tight
  tolerances.
- `nlsolve`: Nonlinear solver to use (default: `CedarRobustNLSolve()`)
- `use_shampine`: Use ShampineCollocationInit after DC solve (default: false).
  This can help oscillators where DC solve gets close but doesn't fully satisfy
  the DAE constraints - Shampine takes a small step to find a consistent state.

# Example
```julia
sol = tran!(circuit, (0.0, 1e-3))  # Uses CedarDCOp by default
sol = tran!(circuit, (0.0, 1e-3); initializealg=CedarDCOp(abstol=1e-8))  # Tighter
sol = tran!(circuit, (0.0, 1e-3); initializealg=CedarDCOp(maxiters=1000))  # More iters
# For oscillators - DC solve + Shampine refinement:
sol = tran!(circuit, (0.0, 1e-3); initializealg=CedarDCOp(use_shampine=true))
```

# Solver Details
Uses `CedarRobustNLSolve()` by default which combines RobustMultiNewton algorithms
with LevenbergMarquardt and PseudoTransient for maximum robustness on difficult
circuits (oscillators, singular Jacobians).
"""
struct CedarDCOp{NLSOLVE} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    maxiters::Int
    nlsolve::NLSOLVE
    use_shampine::Bool
end
CedarDCOp(;abstol=1e-9, maxiters=500, nlsolve=CedarRobustNLSolve(), use_shampine=false) =
    CedarDCOp(abstol, maxiters, nlsolve, use_shampine)

"""
    CedarTranOp <: DiffEqBase.DAEInitializationAlgorithm

Similar to CedarDCOp but uses :tranop mode which may preserve some
operating-point-dependent behavior.

# Arguments
- `abstol`: Tolerance for the DC solve (default: 1e-9)
- `maxiters`: Maximum Newton iterations (default: 500)
- `nlsolve`: Nonlinear solver to use (default: `CedarRobustNLSolve()`)
- `use_shampine`: Use ShampineCollocationInit after DC solve (default: false)
"""
struct CedarTranOp{NLSOLVE} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    maxiters::Int
    nlsolve::NLSOLVE
    use_shampine::Bool
end
CedarTranOp(;abstol=1e-9, maxiters=500, nlsolve=CedarRobustNLSolve(), use_shampine=false) =
    CedarTranOp(abstol, maxiters, nlsolve, use_shampine)

"""
    CedarUICOp <: DiffEqBase.DAEInitializationAlgorithm

UIC (Use Initial Conditions) initialization via pseudo-transient relaxation.

Instead of solving for DC equilibrium (which may not exist for oscillators),
takes fixed implicit Euler steps to relax algebraic constraints while letting
the circuit dynamics evolve naturally.

Uses SciML's DImplicitEuler (DAE path) or ImplicitEuler (ODE path) with
`force_dtmin=true` to march through even if individual steps don't fully converge.

# Arguments
- `warmup_steps::Int`: Number of warmup steps (default: 10)
- `dt::Float64`: Fixed timestep for warmup (default: 1e-12)
- `use_shampine::Bool`: Use ShampineCollocationInit instead of CheckInit (default: false).
  Shampine takes a small step and adjusts both u and du, which can help for oscillators.

# When to Use
- Oscillator circuits with no stable DC operating point
- Circuits where CedarDCOp Newton iteration fails to converge
- When you have known initial conditions and just need constraint relaxation

# Example
```julia
sol = tran!(circuit, (0.0, 1e-3); initializealg=CedarUICOp())
sol = tran!(circuit, (0.0, 1e-3); initializealg=CedarUICOp(warmup_steps=20, dt=1e-11))
# For oscillators - use Shampine which modifies both u and du:
sol = tran!(circuit, (0.0, 1e-3); initializealg=CedarUICOp(warmup_steps=50, use_shampine=true))
```

# Algorithm
1. Create warmup integrator with DImplicitEuler/ImplicitEuler (adaptive=false)
2. Take `warmup_steps` fixed-dt steps with `force_dtmin=true`
3. Extract relaxed state (u, du) from warmup integrator
4. Apply CheckInit or ShampineCollocationInit for final consistency
"""
struct CedarUICOp <: DiffEqBase.DAEInitializationAlgorithm
    warmup_steps::Int
    dt::Float64
    use_shampine::Bool
end
CedarUICOp(; warmup_steps::Int=10, dt::Float64=1e-12, use_shampine::Bool=false) =
    CedarUICOp(warmup_steps, dt, use_shampine)

#==============================================================================#
# Sundials Integration
#
# For Sundials IDA, we implement proper dcop mode switching and Newton solve
# before calling IDADefaultInit.
#==============================================================================#

function SciMLBase.initialize_dae!(integrator::Sundials.IDAIntegrator,
                                   alg::Union{CedarDCOp, CedarTranOp})
    prob = integrator.sol.prob
    # Use the DC solve tolerance directly - don't constrain to transient tolerance
    abstol = alg.abstol

    (; p, f) = prob
    u0 = integrator.u
    du0 = integrator.du
    tmp = similar(u0)

    du0 .= 0.0
    f(tmp, du0, u0, p, 0.0)

    if norm(tmp) < abstol
        return  # Already converged
    end

    # Use the SAME Newton solver as dc_solve_with_ctx for unified DC initialization
    # The workspace p contains the CompiledStructure and can be used directly
    ws = p::EvalWorkspace
    cs = ws.structure

    # Create a CompiledStructure with dcop/tranop mode for the solve
    mode = alg isa CedarDCOp ? :dcop : :tranop
    cs_dc = @set cs.spec = with_mode(cs.spec, mode)

    # Use zeros as initial guess (same as dc_solve_with_ctx)
    u0_zeros = zeros(length(u0))

    # Call the shared Newton iteration
    u_sol, converged = MNA._dc_newton_compiled(cs_dc, ws, u0_zeros;
                                                abstol=abstol, maxiters=alg.maxiters,
                                                nlsolve=alg.nlsolve)

    integrator.u .= u_sol
    if !converged
        @warn "DC operating point analysis failed. Further failures may follow."
        integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, ReturnCode.InitialFailure)
    end

    copyto!(integrator.uprev, integrator.u)
    integrator.u_modified = true

    # Use Shampine or DefaultInit for final consistency
    if alg.use_shampine
        # ShampineCollocationInit takes a small step to find consistent state
        # Good for oscillators where DC solve gets close but doesn't fully converge
        return SciMLBase.initialize_dae!(integrator, OrdinaryDiffEq.ShampineCollocationInit())
    else
        return SciMLBase.initialize_dae!(integrator, Sundials.DefaultInit())
    end
end

#==============================================================================#
# OrdinaryDiffEq Integration (for ODE solvers with mass matrix)
#
# For ODE solvers, CedarDCOp performs the DC solve to compute u0.
# The mass matrix C is constant (voltage-dependent capacitors use charge
# formulation), so initialization only needs to solve for the DC operating point.
#==============================================================================#

function SciMLBase.initialize_dae!(integrator::ODEIntegrator,
                                   alg::Union{CedarDCOp, CedarTranOp})
    prob = integrator.sol.prob
    # Use the DC solve tolerance directly - don't constrain to transient tolerance
    # DC analysis for circuits with internal model nodes (like BJT excess phase)
    # may have larger residuals that are still acceptable for transient startup
    abstol = alg.abstol

    # For ODE problems, p is the EvalWorkspace
    ws = prob.p
    if !(ws isa EvalWorkspace)
        # Not an MNA circuit - delegate to default
        return
    end

    u0 = integrator.u
    cs = ws.structure

    # Create a CompiledStructure with dcop/tranop mode for the solve
    mode = alg isa CedarDCOp ? :dcop : :tranop
    cs_dc = @set cs.spec = with_mode(cs.spec, mode)

    # Solve DC operating point using the simulation workspace
    u_sol, converged = MNA._dc_newton_compiled(cs_dc, ws, zeros(length(u0));
                                                abstol=abstol, maxiters=alg.maxiters,
                                                nlsolve=alg.nlsolve)

    integrator.u .= u_sol
    if !converged
        @warn "DC operating point analysis failed. Further failures may follow."
        integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, ReturnCode.InitialFailure)
    end

    # Use Shampine or DefaultInit for final consistency
    if alg.use_shampine
        # ShampineCollocationInit takes a small step to find consistent state
        # Good for oscillators where DC solve gets close but doesn't fully converge
        return SciMLBase.initialize_dae!(integrator, OrdinaryDiffEq.ShampineCollocationInit())
    else
        return SciMLBase.initialize_dae!(integrator, DiffEqBase.DefaultInit())
    end
end

#==============================================================================#
# CedarUICOp: Pseudo-Transient Relaxation for Oscillators
#
# Instead of Newton iteration for DC equilibrium, take fixed implicit Euler
# steps to relax algebraic constraints. Useful for oscillators with no stable
# DC operating point.
#==============================================================================#

function SciMLBase.initialize_dae!(integrator::Sundials.IDAIntegrator, alg::CedarUICOp)
    prob = integrator.sol.prob
    ws = prob.p::EvalWorkspace

    # For DAE warmup, use ODE formulation with mass matrix
    # This is the same approach as the ODE path but allows us to warm up DAE problems
    warmup_tspan = (0.0, alg.warmup_steps * alg.dt * 2)

    # Create ODE problem from the workspace's structure (mass matrix formulation)
    # The mass matrix C is already available in the compiled structure
    cs = ws.structure

    # RHS function: du = b - G*u (same as ODEProblem formulation)
    function warmup_rhs!(du, u, p, t)
        fast_rebuild!(p, u, real_time(t))
        mul!(du, p.structure.G, u)
        du .*= -1
        du .+= p.dctx.b
        return nothing
    end

    # Create ODE function with mass matrix (no explicit Jacobian - use autodiff)
    warmup_f = SciMLBase.ODEFunction(warmup_rhs!; mass_matrix=cs.C)
    warmup_prob = SciMLBase.ODEProblem(warmup_f, prob.u0, warmup_tspan, ws)

    # Use ImplicitEuler for warmup (backward Euler for relaxation)
    # force_dtmin=true lets it march through even if Newton struggles
    # NoInit skips initialization - we start from zeros intentionally
    warmup_int = init(warmup_prob, ImplicitEuler();
                      adaptive=false, dt=alg.dt, force_dtmin=true,
                      initializealg=NoInit())

    for _ in 1:alg.warmup_steps
        step!(warmup_int)
    end

    # Copy the relaxed state back to the main integrator
    integrator.u .= warmup_int.u
    # Estimate du from the last warmup step
    integrator.du .= (warmup_int.u .- warmup_int.uprev) ./ alg.dt

    copyto!(integrator.uprev, integrator.u)
    integrator.u_modified = true

    # Apply final consistency check: CheckInit or ShampineCollocationInit
    if alg.use_shampine
        # ShampineCollocationInit takes a small step and adjusts both u and du
        # Good for oscillators where we need to refine the approximated derivatives
        return SciMLBase.initialize_dae!(integrator, OrdinaryDiffEq.ShampineCollocationInit())
    else
        # CheckInit validates without modifying - may fail for oscillators with high residual
        return SciMLBase.initialize_dae!(integrator, OrdinaryDiffEq.CheckInit())
    end
end

function SciMLBase.initialize_dae!(integrator::ODEIntegrator, alg::CedarUICOp)
    prob = integrator.sol.prob

    # Create warmup ODE problem with mass matrix
    warmup_tspan = (0.0, alg.warmup_steps * alg.dt * 2)  # Finite tspan
    warmup_prob = remake(prob; tspan=warmup_tspan)

    # ImplicitEuler with mass matrix for ODE warmup
    # Use NoInit for the warmup phase - we're intentionally starting from
    # potentially inconsistent initial conditions and letting the solver relax them
    # force_dtmin=true lets it march through even if Newton struggles
    warmup_int = init(warmup_prob, ImplicitEuler();
                      adaptive=false, dt=alg.dt, force_dtmin=true,
                      initializealg=NoInit())

    for _ in 1:alg.warmup_steps
        step!(warmup_int)
    end

    # For mass matrix ODE, we use the finite difference from the last step as du estimate
    integrator.u .= warmup_int.u

    # Apply final consistency check: CheckInit or ShampineCollocationInit
    if alg.use_shampine
        # ShampineCollocationInit takes a small step and adjusts both u and du
        return SciMLBase.initialize_dae!(integrator, OrdinaryDiffEq.ShampineCollocationInit())
    else
        # CheckInit validates without modifying
        return SciMLBase.initialize_dae!(integrator, OrdinaryDiffEq.CheckInit())
    end
end
