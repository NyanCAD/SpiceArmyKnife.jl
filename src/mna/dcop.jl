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
using LinearAlgebra
using NonlinearSolve
using SciMLBase
using SciMLBase: ReturnCode
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
    SciMLBase.initialize_dae!(integrator, Sundials.DefaultInit())
    return
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

    return
end
