#==============================================================================#
# Cedar DC Operating Point Initialization
#
# Robust DAE initialization that mirrors the original CedarSim approach.
# CedarDCOp wraps ShampineCollocationInit (which works well for voltage-dependent
# capacitors) with additional robustness features.
#
# The Sundials integration does proper dcop mode switching and bootstrapped
# nonlinear solve before IDADefaultInit.
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

    # Create nonlinear function that evaluates with :dcop mode
    nlf! = (out, u, nl_p) -> _eval_dae_dcop(out, du0, u, nl_p, alg)

    nlprob = NonlinearProblem(nlf!, u0, p)
    sol = bootstrapped_nlsolve(nlprob, alg.nlsolve, abstol)

    integrator.u .= sol.u
    if sol.retcode != ReturnCode.Success
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
"""
function _eval_dae_dcop(out, du0, u, ws::EvalWorkspace, alg)
    mode = alg isa CedarDCOp ? :dcop : :tranop
    original_spec = ws.structure.spec
    dc_spec = with_mode(original_spec, mode)

    cs = ws.structure
    dctx = ws.dctx

    reset_direct_stamp!(dctx)
    cs.builder(cs.params, dc_spec, 0.0; x=u, ctx=dctx)

    n_deferred = cs.n_b_deferred
    for k in 1:n_deferred
        idx = dctx.b_resolved[k]
        if idx > 0
            dctx.b[idx] += dctx.b_V[k]
        end
    end

    # F(0, u) = G*u - b
    mul!(out, cs.G, u)
    out .-= dctx.b

    return nothing
end
