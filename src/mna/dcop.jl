#==============================================================================#
# Cedar DC Operating Point Initialization
#
# Robust DAE initialization that mirrors the original CedarSim approach.
# CedarDCOp wraps ShampineCollocationInit (which works well for voltage-dependent
# capacitors) with additional robustness features.
#
# The Sundials integration does proper dcop mode switching and bootstrapped
# nonlinear solve before IDADefaultInit.
#
# UNIFIED DC SOLVE:
# Both dc!(circuit) and CedarDCOp initialization share the same core logic:
# - dc_residual! computes F(u) = G(u)*u - b(u) with dcop mode (defined in solve.jl)
# - dc_solve_core does Newton iteration to solve F(u) = 0 (defined in solve.jl)
# This ensures consistent DC operating point results between standalone DC
# analysis and transient initialization.
#==============================================================================#

using OrdinaryDiffEq
using LinearAlgebra
using NonlinearSolve
using SciMLBase
using SciMLBase: NLStats, ReturnCode
using DiffEqBase
using ADTypes
using Sundials

export CedarDCOp, CedarTranOp

# Note: dc_residual! and dc_solve_core are defined in solve.jl and exported from there

#==============================================================================#
# Workspace-based DC Residual (for compiled circuits)
#
# This is the zero-allocation path used by CedarDCOp during transient initialization.
# Uses DirectStampContext to rebuild values in-place.
#==============================================================================#

"""
    dc_residual_ws!(out, u, ws::EvalWorkspace, spec)

DC residual using EvalWorkspace (for compiled circuits).

This is the zero-allocation path used by CedarDCOp during transient initialization.
Uses DirectStampContext to rebuild values in-place. It shares the same residual
definition as dc_residual! (F(u) = G(u)*u - b(u)) but operates on compiled structures.
"""
function dc_residual_ws!(out::AbstractVector, u::AbstractVector,
                         ws::EvalWorkspace, spec)
    cs = ws.structure
    dctx = ws.dctx

    # Reset and rebuild with dcop mode
    reset_direct_stamp!(dctx)
    cs.builder(cs.params, spec, 0.0; x=u, ctx=dctx)

    # Apply deferred b stamps
    n_deferred = cs.n_b_deferred
    for k in 1:n_deferred
        idx = dctx.b_resolved[k]
        if idx > 0
            dctx.b[idx] += dctx.b_V[k]
        end
    end

    # F(u) = G*u - b
    mul!(out, cs.G, u)
    out .-= dctx.b
    return nothing
end

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
- `nlsolve`: Nonlinear solver to use (default: RobustMultiNewton with finite diff)

# Example
```julia
sol = tran!(circuit, (0.0, 1e-3))  # Uses CedarDCOp by default
```
"""
struct CedarDCOp{NLSOLVE} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    nlsolve::NLSOLVE
end
CedarDCOp(;abstol=1e-10, nlsolve=RobustMultiNewton(autodiff=AutoFiniteDiff())) = CedarDCOp(abstol, nlsolve)

"""
    CedarTranOp <: DiffEqBase.DAEInitializationAlgorithm

Similar to CedarDCOp but uses :tranop mode which may preserve some
operating-point-dependent behavior.
"""
struct CedarTranOp{NLSOLVE} <: DiffEqBase.DAEInitializationAlgorithm
    abstol::Float64
    nlsolve::NLSOLVE
end
CedarTranOp(;abstol=1e-10, nlsolve=RobustMultiNewton(autodiff=AutoFiniteDiff())) = CedarTranOp(abstol, nlsolve)

"""
    bootstrapped_nlsolve(nlprob, nlsolve, abstol; num_trajectories=10, maxiters=200)

Robust nonlinear solver that defeats local minima traps by starting from
multiple random initial points and iteratively solving each trajectory
until one satisfies the given tolerance.
"""
function bootstrapped_nlsolve(nlprob, nlsolve, abstol; num_trajectories=10, maxiters=200)
    stats = NLStats(0, 0, 0, 0, 0)
    last_residual = nothing
    local prob

    for idx in 1:num_trajectories
        try
            prob = remake(nlprob, u0=1e-7 * randn(size(nlprob.u0)))
            sol = solve(prob, nlsolve; abstol, maxiters)
            last_residual = sol.resid

            stats = NLStats(
                stats.nf + sol.stats.nf,
                stats.njacs + sol.stats.njacs,
                stats.nfactors + sol.stats.nfactors,
                stats.nsolve + sol.stats.nsolve,
                stats.nsteps + sol.stats.nsteps
            )

            prob.u0 .= sol.u
            if sol.retcode == ReturnCode.Success
                return SciMLBase.build_solution(prob, nlsolve, sol.u, sol.resid;
                    retcode=sol.retcode, original=sol.original, stats=stats)
            end
        catch e
            if isa(e, LinearAlgebra.SingularException)
                prob.u0 .= 1e-7 * randn(size(prob.u0))
                continue
            end
            rethrow(e)
        end
    end

    return SciMLBase.build_solution(
        prob, nlsolve, prob.u0, last_residual;
        retcode=ReturnCode.MaxIters, original=nothing, stats=stats,
    )
end

#==============================================================================#
# Sundials Integration
#
# For Sundials IDA, we implement proper dcop mode switching and bootstrapped
# nonlinear solve before calling IDADefaultInit.
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

    # Switch to dcop/tranop mode for the solve
    mode = alg isa CedarDCOp ? :dcop : :tranop
    original_spec = ws.structure.spec
    dc_spec = with_mode(original_spec, mode)

    # Create a temporary compiled structure with dcop mode
    # We need to use the same _dc_newton_compiled that dc! uses
    cs_dcop = CompiledStructure(
        ws.structure.builder,
        ws.structure.params,
        dc_spec,
        ws.structure.n_nodes,
        ws.structure.n_currents,
        ws.structure.G,
        ws.structure.C,
        ws.structure.n_b_deferred
    )

    # Use zeros as initial guess (same as dc_solve_with_ctx)
    u0_zeros = zeros(length(u0))

    # Call the shared Newton iteration
    u_sol, converged = MNA._dc_newton_compiled(cs_dcop, ws, u0_zeros;
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
    SciMLBase.initialize_dae!(integrator, Sundials.DefaultInit())
    return
end

#==============================================================================#
# Helper Functions for Mode Switching
#==============================================================================#

"""
Evaluate DAE residual with :dcop mode for DC initialization.

This is a thin wrapper around dc_residual_ws! that handles the mode switch.
"""
function _eval_dae_dcop(out, du0, u, ws::EvalWorkspace, alg)
    mode = alg isa CedarDCOp ? :dcop : :tranop
    original_spec = ws.structure.spec
    dc_spec = with_mode(original_spec, mode)

    # Use shared residual function
    dc_residual_ws!(out, u, ws, dc_spec)

    return nothing
end
