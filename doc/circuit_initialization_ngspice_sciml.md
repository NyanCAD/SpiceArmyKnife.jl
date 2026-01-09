# Circuit Simulation Initialization: NGSPICE Methods and SciML Equivalents

The SciML ecosystem provides a modern, composable alternative to traditional SPICE initialization, with **NonlinearSolve.jl** replacing Newton-Raphson operating point analysis and **DifferentialEquations.jl** providing explicit DAE initialization algorithms that give users precise control over consistency enforcement—capabilities SPICE handles implicitly and often opaquely.

This report maps ngspice initialization algorithms to their Julia equivalents, providing practitioners with actionable guidance for implementing robust circuit simulation initialization in the SciML ecosystem.

---

## NGSPICE initialization fundamentals

NGSPICE uses Modified Nodal Analysis (MNA) to formulate circuit equations as a differential-algebraic system of the form **C(x)·dx/dt + G(x)·x + f(x,t) = s(t)**, where the matrix **C** is singular, making this a DAE rather than a pure ODE. The initialization challenge is finding consistent initial conditions that satisfy both the algebraic constraints (KCL/KVL) and energy storage element states.

### DC operating point (.op) and Newton-Raphson

The `.op` analysis finds the steady-state solution by treating inductors as short circuits and capacitors as open circuits, then solving the resulting nonlinear algebraic system **i(v_dc) + u_dc = 0** via Newton-Raphson iteration. The algorithm linearizes about the current estimate, computing the Jacobian **J(v^k) = ∂f/∂v**, then solves the linear system **v^(k+1) = v^k - J^(-1)·f(v^k)** iteratively until convergence criteria are met.

Convergence requires both voltage and current tolerances to be satisfied: **|Δv_n| < RELTOL × max(|v_n^k|, |v_n^(k-1)|) + VNTOL** for voltages and analogous criteria for currents using **ABSTOL**. The default iteration limit **ITL1=100** bounds the maximum attempts before declaring non-convergence.

### .IC versus .NODESET directives

The **.IC directive** sets true initial conditions as hard constraints. Without the UIC option, these values are forced during the DC bias calculation, effectively clamping specified node voltages while solving for the rest of the circuit. After DC convergence, the constraints release for transient simulation. With UIC on `.TRAN`, the values are directly "stuffed" into the solution vector without any DC solve.

The **.NODESET directive** provides initial guesses rather than constraints—a critical distinction. These values seed the Newton-Raphson iteration but do not constrain the final solution, which settles to the circuit's natural equilibrium. NODESET primarily aids convergence by providing better starting points, especially for multi-stable circuits where the final state depends on the iteration path.

### UIC (use initial conditions) in transient analysis

The **UIC keyword** on `.TRAN` commands completely bypasses DC operating point calculation. The simulator directly uses `.IC` values (or device-level `IC=` parameters) to initialize energy storage elements, then begins integration immediately. This proves essential for oscillator circuits with no stable DC point, or when the DC solver fails to converge. The tradeoff: AC analysis becomes unavailable since it requires a valid operating point, and supplies may not be properly biased.

---

## NGSPICE convergence recovery algorithms

When standard Newton-Raphson fails, ngspice employs two primary continuation algorithms before declaring non-convergence.

### GMIN stepping algorithm

GMIN stepping exploits the observation that adding conductance across semiconductor junctions linearizes their exponential characteristics. The algorithm starts with **GMIN ≈ 0.01 S** (effectively 100Ω resistors damping junction nonlinearities), finds a solution with this linearized circuit, then progressively reduces GMIN through steps like **10^(-3) → 10^(-4) → ... → 10^(-12) S**, using each solution as the initial guess for the next step. This works particularly well for CMOS circuits with many reverse-biased junctions.

### Source stepping algorithm

Source stepping is a homotopy method that scales all sources to near-zero, finds a solution at this "easy" operating point, then incrementally increases source values toward nominal. Dynamic source stepping adjusts the step size—accelerating when convergence is easy, retreating when iterations fail. This method works better than GMIN stepping for circuits with forward-biased junctions and follows the natural I-V curve trajectory.

### Pseudo-transient continuation

When both GMIN and source stepping fail, ngspice's `optran` branch implements pseudo-transient analysis: add capacitors to all nodes (or use existing reactive elements), run a transient simulation with sources ramping from zero, and let the circuit settle to steady state. The final time point becomes the DC operating point. This method is particularly effective for oscillator circuits and highly nonlinear designs.

---

## NGSPICE DAE and numerical handling

### Integration methods for time-domain analysis

NGSPICE converts the DAE to an algebraic system at each timestep using implicit integration. The **trapezoidal rule** (default) provides second-order accuracy and A-stability but can cause numerical ringing on high-Q circuits. **Gear methods** (BDF) offer better stability for stiff systems at the cost of some accuracy, with orders 2-6 available via `.option METHOD=gear MAXORD=n`.

### Singular Jacobian handling

Singular matrices arise from floating nodes (no DC path to ground), capacitor-only paths, inductor loops, or voltage source loops. NGSPICE detects these with pivot monitoring and reports problematic nodes. Remediation includes the **RSHUNT option** (adds high-value resistors from each node to ground), GMIN addition across junctions, and circuit topology corrections. The **PIVREL** and **PIVTOL** options control pivot acceptance thresholds.

### Newton-Raphson damping strategies

NGSPICE prevents Newton iteration overshoot through junction voltage limiting (restricting changes across pn-junctions to 2-3 thermal voltages per iteration to prevent exponential overflow) and global update damping with automatic factor reduction when residuals fail to decrease.

---

## SciML DAE initialization system

The SciML ecosystem handles DAE initialization explicitly through the `initializealg` keyword in `solve()`, providing users direct control over how consistency is achieved—a significant architectural difference from SPICE's implicit handling.

### Core initialization algorithms

**BrownFullBasicInit** implements Brown's algorithm for index-1 DAEs, keeping differential variables constant at user-specified values while modifying algebraic variables (and derivatives) to satisfy constraints. This is the recommended default for most circuit DAEs where initial voltages/currents on energy storage elements are known:

```julia
prob = DAEProblem(f!, du₀, u₀, tspan; differential_vars = [true, true, false])
sol = solve(prob, IDA(), initializealg = BrownFullBasicInit())
```

**ShampineCollocationInit** takes a small forward step in time, treating all initial values as guesses and potentially modifying both differential and algebraic variables. This works without requiring `differential_vars` specification and handles difficult cases where Brown's method fails.

**CheckInit** (default in Sundials.jl v5+) validates consistency without modification, erroring immediately if initial conditions don't satisfy constraints. This prevents unexpected value changes but requires users to provide already-consistent conditions.

### Mass matrix versus implicit DAE formulation

For MNA-derived circuits, two formulations are available. The **mass matrix form** `M·u' = f(u,p,t)` works with Rosenbrock methods (Rodas5P) and BDF methods (QNDF, FBDF), with algebraic equations indicated by zero rows in M. The **fully implicit form** `F(du,u,p,t) = 0` requires IDA or DFBDF solvers and explicit `differential_vars` specification. The mass matrix approach integrates more naturally with custom MNA engines.

---

## SciML equivalents for SPICE operating point analysis

### SteadyStateDiffEq.jl for DC operating point

The direct equivalent to `.op` uses `SteadyStateProblem` with either dynamic or rootfinding approaches:

```julia
using SteadyStateDiffEq, OrdinaryDiffEq

prob = SteadyStateProblem(circuit_dynamics!, u0, p)

# DynamicSS: approaches steady state via time integration (like pseudo-transient)
sol = solve(prob, DynamicSS(Rodas5P()))

# SSRootfind: direct Newton solve (like SPICE Newton-Raphson)
sol = solve(prob, SSRootfind())
```

**DynamicSS** mimics SPICE's source stepping/pseudo-transient approach—inherently more stable but slower. **SSRootfind** performs direct Newton-Raphson, faster when initial guesses are good but less robust.

### NonlinearSolve.jl algorithm mapping

The NonlinearSolve.jl package provides direct equivalents to SPICE's Newton-Raphson variants:

| SPICE Method | NonlinearSolve.jl Equivalent |
|--------------|------------------------------|
| Standard Newton | `NewtonRaphson()` |
| Newton with limiting | `NewtonRaphson(linesearch = BackTracking())` |
| Damped Newton | `TrustRegion()` |
| GMIN stepping (regularization) | `LevenbergMarquardt()` |
| Source stepping/pseudo-transient | `PseudoTransient()` |
| Automatic fallback | `RobustMultiNewton()` |

The **PseudoTransient** solver implements Switched Evolution Relaxation—starting with small pseudo-timesteps (implicit Euler-like stability) and automatically increasing them as residuals decrease, transitioning to pure Newton near the solution. This closely mimics ngspice's transient-based DC operating point finder.

---

## Practical implementation for custom MNA engines

### ngspice-like NonlinearSolvePolyAlgorithm

The following polyalgorithm mirrors ngspice's fallback sequence for DC operating point analysis:

```julia
using NonlinearSolve

ngspice_like_solver = NonlinearSolvePolyAlgorithm((
    # First try: Newton with backtracking (like SPICE's junction limiting)
    NewtonRaphson(linesearch = BackTracking(order = 3)),
    
    # Second try: Trust region (more robust than pure LM for circuits)
    TrustRegion(),
    
    # Third try: LM with strong damping (like GMIN stepping effect)
    LevenbergMarquardt(damping_initial = 1.0),
    
    # Last resort: pseudo-transient continuation
    PseudoTransient(alpha_initial = 1e-4)
))

function find_operating_point(mna_residual!, u_guess, p)
    prob = NonlinearProblem(mna_residual!, u_guess, p)
    return solve(prob, ngspice_like_solver; maxiters = 200)
end
```

For circuits with particularly sharp nonlinearities, `TrustRegion(radius_update_scheme = RadiusUpdateSchemes.Bastin)` provides more aggressive trust region shrinking.

### UIC for oscillators and circuits without stable equilibrium

The standard `PseudoTransient()` solver is inappropriate for oscillators because it still seeks steady state (dv/dt = 0). For circuits without stable DC operating points, use a **relaxation warmup** approach: take a few implicit integration steps to let algebraic constraints settle without expecting convergence to equilibrium.

```julia
using OrdinaryDiffEq

function solve_with_relaxation_warmup(circuit!, M, u0, tspan, p;
                                       warmup_steps = 10,
                                       warmup_dt = 1e-12)
    """
    Take a few backward Euler steps with fixed dt to relax 
    from inconsistent initial conditions, then hand off to adaptive solver.
    """
    
    # Mass matrix form: M * du/dt = f(u, p, t)
    f_ode = ODEFunction(circuit!; mass_matrix = M)
    
    # Phase 1: Fixed-step implicit Euler warmup
    warmup_tspan = (tspan[1], tspan[1] + warmup_steps * warmup_dt)
    warmup_prob = ODEProblem(f_ode, u0, warmup_tspan, p)
    
    warmup_sol = solve(warmup_prob, ImplicitEuler();
                       adaptive = false,
                       dt = warmup_dt,
                       force_dtmin = true,  # Don't error on convergence issues
                       save_everystep = false)
    
    # Phase 2: Continue with adaptive solver from relaxed state  
    u_relaxed = warmup_sol.u[end]
    main_prob = ODEProblem(f_ode, u_relaxed, tspan, p)
    
    return solve(main_prob, Rodas5P())  # or FBDF() for larger circuits
end
```

For particularly difficult cases (oscillators with hard switching), `TRBDF2()` provides numerical damping that absorbs high-frequency transients from inconsistent initial conditions:

```julia
function solve_oscillator_uic(circuit!, M, u0, tspan, p)
    f_ode = ODEFunction(circuit!; mass_matrix = M)
    prob = ODEProblem(f_ode, u0, tspan, p)
    
    return solve(prob, TRBDF2();
                 dtmin = 1e-18,      # Allow very small steps initially
                 maxiters = 1_000_000)
end
```

### What ngspice does with UIC (for reference)

With `.TRAN ... UIC`, ngspice:

1. Reads `.IC V(node)=value` and device-level `IC=value` parameters
2. Sets those node voltages / branch currents directly into the solution vector
3. Computes consistent derivatives by evaluating the circuit equations at t=0
4. Starts integrating immediately with the trapezoidal rule

The `BrownFullBasicInit` algorithm in DifferentialEquations.jl performs a similar operation—keeping differential variables fixed while solving for algebraic consistency—but requires explicit specification of which variables are differential via the `differential_vars` parameter.

### Complete MNA engine integration example

```julia
using NonlinearSolve, OrdinaryDiffEq, SparseArrays

# Your sparse Jacobian prototype from MNA structure
jac_prototype = create_mna_jacobian_sparsity(circuit)

ngspice_dc_solver = NonlinearSolvePolyAlgorithm((
    NewtonRaphson(linesearch = BackTracking(order = 3)),
    TrustRegion(),
    LevenbergMarquardt(damping_initial = 1.0),
    PseudoTransient(alpha_initial = 1e-4)
))

function find_operating_point(mna_residual!, u_guess, p)
    prob = NonlinearProblem(mna_residual!, u_guess, p)
    return solve(prob, ngspice_dc_solver; maxiters = 200)
end

function transient_analysis(mna_ode!, M, u0, tspan, p; 
                            use_ic = false,
                            warmup_steps = 20,
                            warmup_dt = 1e-12)
    
    f = ODEFunction(mna_ode!; mass_matrix = M, jac_prototype = jac_prototype)
    
    if use_ic
        # UIC mode: implicit Euler warmup to relax algebraic constraints
        warmup_tspan = (tspan[1], tspan[1] + warmup_steps * warmup_dt)
        warmup_prob = ODEProblem(f, u0, warmup_tspan, p)
        
        warmup_sol = solve(warmup_prob, ImplicitEuler(); 
                           adaptive = false, 
                           dt = warmup_dt, 
                           force_dtmin = true,
                           save_everystep = false)
        
        u_start = warmup_sol.u[end]
    else
        # Normal mode: find DC operating point first
        dc_sol = find_operating_point(dc_residual!, u0, p)
        u_start = dc_sol.u
    end
    
    # Main transient simulation
    prob = ODEProblem(f, u_start, tspan, p)
    return solve(prob, FBDF())  # or Rodas5P() for smaller circuits
end
```

---

## Stiff solver selection for transient analysis

### Solver recommendations by circuit size

| Circuit Size | Recommended Solver | Notes |
|--------------|-------------------|-------|
| Small (<100 nodes) | `Rodas5P()` | Rosenbrock, better than trap for stiff |
| Medium | `TRBDF2()` | Good numerical damping |
| Large sparse | `QNDF()` with sparse Jacobian | Similar to Gear |
| Very large | `FBDF()` or `IDA()` | CVODE-class solvers |

### Jacobian computation strategies

For large circuits, analytical Jacobians dramatically improve performance. For hand-coded MNA:

```julia
using SparseConnectivityTracer, ADTypes

sparsity = jacobian_sparsity(circuit!, du, u)
jac_prototype = sparse(sparsity)

f = ODEFunction(circuit!;
    jac_prototype = jac_prototype,
    colorvec = matrix_colors(jac_prototype))  # Graph coloring for efficiency
```

For very large circuits, **Jacobian-free Newton-Krylov** with preconditioning avoids explicit Jacobian storage:

```julia
sol = solve(prob, QNDF(linsolve = KrylovJL_GMRES()))
```

---

## Practical convergence strategies

### Ill-conditioned system handling

The Julia equivalents to SPICE's singular matrix remediation include:

```julia
# GMIN-like regularization
function circuit_with_gmin!(F, u, p)
    circuit!(F, u, p)
    GMIN = 1e-12
    for i in eachindex(u)
        F[i] += GMIN * u[i]
    end
end

# Levenberg-Marquardt damping (prevents singular Jacobian)
sol = solve(prob, LevenbergMarquardt(damping_initial = 1.0))

# SVD for near-singular systems
sol = solve(prob, NewtonRaphson(linsolve = SVDFactorization()))
```

### Preconditioning for large circuits

```julia
using IncompleteLU, LinearSolve

function circuit_preconditioner(W, p)
    Pl = ilu(W, τ = 50.0)
    return Pl, LinearAlgebra.I
end

sol = solve(prob, NewtonRaphson(
    linsolve = KrylovJL_GMRES(precs = circuit_preconditioner)))
```

---

## Summary mapping table

| NGSPICE Feature | SciML Equivalent | Key Package |
|-----------------|------------------|-------------|
| `.op` Newton-Raphson | `NonlinearProblem` + `NewtonRaphson()` | NonlinearSolve.jl |
| `.op` with fallbacks | `NonlinearSolvePolyAlgorithm` (Newton→TrustRegion→LM→PseudoTransient) | NonlinearSolve.jl |
| `.IC` directive | `u0` specification + `BrownFullBasicInit` | DifferentialEquations.jl |
| `.NODESET` directive | `guesses` keyword | ModelingToolkit.jl |
| `UIC` on `.TRAN` | Implicit Euler warmup + adaptive solver | OrdinaryDiffEq.jl |
| `UIC` for oscillators | Relaxation warmup (no steady-state assumption) | OrdinaryDiffEq.jl |
| GMIN stepping | `LevenbergMarquardt(damping_initial = 1.0)` | NonlinearSolve.jl |
| Source stepping | `PseudoTransient(alpha_initial = 1e-4)` | NonlinearSolve.jl |
| Pseudo-transient DC | `DynamicSS()` or `PseudoTransient()` | SteadyStateDiffEq.jl |
| Trapezoidal integration | `Trapezoid()` | OrdinaryDiffEq.jl |
| Gear/BDF integration | `FBDF()`, `QNDF()` | OrdinaryDiffEq.jl |
| Junction limiting | `BackTracking(order = 3)` linesearch | NonlinearSolve.jl |

The SciML ecosystem offers more explicit control over initialization and continuation than traditional SPICE, at the cost of requiring users to understand and specify these choices. For production circuit simulation, the combination of `NonlinearSolvePolyAlgorithm` for DC operating point analysis and relaxation warmup for UIC-style transient initialization provides a robust, ngspice-equivalent workflow.
