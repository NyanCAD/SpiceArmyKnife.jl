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
   robust nonlinear solver, then calls IDADefaultInit
2. For OrdinaryDiffEq: Delegates to ShampineCollocationInit which handles
   voltage-dependent capacitors well

This is the recommended initialization algorithm for circuits with nonlinear
devices (diodes, MOSFETs, voltage-dependent capacitors).

# Arguments
- `abstol`: Tolerance for the DC solve (default: 1e-10)
- `nlsolve`: Nonlinear solver to use (default: `CedarRobustNLSolve()`)

# Example
```julia
sol = tran!(circuit, (0.0, 1e-3))  # Uses CedarDCOp by default
```

# Solver Details
Uses `CedarRobustNLSolve()` by default which combines RobustMultiNewton algorithms
with LevenbergMarquardt and PseudoTransient for maximum robustness on difficult
circuits (oscillators, singular Jacobians).
"""
struct CedarDCOp{NLSOLVE} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    nlsolve::NLSOLVE
end
CedarDCOp(;abstol=1e-10, nlsolve=CedarRobustNLSolve()) = CedarDCOp(abstol, nlsolve)

"""
    CedarTranOp <: DiffEqBase.DAEInitializationAlgorithm

Similar to CedarDCOp but uses :tranop mode which may preserve some
operating-point-dependent behavior.
"""
struct CedarTranOp{NLSOLVE} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    nlsolve::NLSOLVE
end
CedarTranOp(;abstol=1e-10, nlsolve=CedarRobustNLSolve()) = CedarTranOp(abstol, nlsolve)

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

# When to Use
- Oscillator circuits with no stable DC operating point
- Circuits where CedarDCOp Newton iteration fails to converge
- When you have known initial conditions and just need constraint relaxation

# Example
```julia
sol = tran!(circuit, (0.0, 1e-3); initializealg=CedarUICOp())
sol = tran!(circuit, (0.0, 1e-3); initializealg=CedarUICOp(warmup_steps=20, dt=1e-11))
```

# Algorithm
1. Create warmup integrator with DImplicitEuler/ImplicitEuler (adaptive=false)
2. Take `warmup_steps` fixed-dt steps with `force_dtmin=true`
3. Extract relaxed state (u, du) from warmup integrator
4. Hand off to main solver's DefaultInit for final consistency check
"""
struct CedarUICOp <: DiffEqBase.DAEInitializationAlgorithm
    warmup_steps::Int
    dt::Float64
end
CedarUICOp(; warmup_steps::Int=10, dt::Float64=1e-12) = CedarUICOp(warmup_steps, dt)

#==============================================================================#
# Sundials Integration
#
# For Sundials IDA, we implement proper dcop mode switching and Newton solve
# before calling IDADefaultInit.
#==============================================================================#

function SciMLBase.initialize_dae!(integrator::Sundials.IDAIntegrator,
                                   alg::Union{CedarDCOp, CedarTranOp})
    prob = integrator.sol.prob
    abstol = min(integrator.opts.abstol, alg.abstol)

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
                                                abstol=abstol, maxiters=100,
                                                nlsolve=alg.nlsolve)

    integrator.u .= u_sol
    if !converged
        @warn "DC operating point analysis failed. Further failures may follow."
        integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, ReturnCode.InitialFailure)
    end

    copyto!(integrator.uprev, integrator.u)
    integrator.u_modified = true

    # Let DefaultInit handle the rest
    return SciMLBase.initialize_dae!(integrator, Sundials.DefaultInit())
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
    abstol = min(integrator.opts.abstol, alg.abstol)

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
                                                abstol=abstol, maxiters=100,
                                                nlsolve=alg.nlsolve)

    integrator.u .= u_sol
    if !converged
        @warn "DC operating point analysis failed. Further failures may follow."
        integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, ReturnCode.InitialFailure)
    end

    return SciMLBase.initialize_dae!(integrator, DiffEqBase.DefaultInit())
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

    # Let DefaultInit validate/fix consistency for the main solver
    return SciMLBase.initialize_dae!(integrator, Sundials.DefaultInit())
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

    # The mass matrix ODE doesn't directly give us du, but after warmup
    # the algebraic constraints should be satisfied.
    return SciMLBase.initialize_dae!(integrator, DiffEqBase.DefaultInit())
end
