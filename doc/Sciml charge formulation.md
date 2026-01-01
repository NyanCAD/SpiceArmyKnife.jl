# SciML Circuit Simulation: DAE and Variable Mass Matrix Support

**State-dependent mass matrices are not officially supported in SciML**, but practical workarounds exist. For MNA circuits with voltage-dependent capacitors, the recommended approach is reformulating to a `DAEProblem` with IDA/DFBDF solvers, or using charge-based formulations that yield constant mass matrices. This report explains why SPICE's companion model approach doesn't directly map to SciML's architecture and provides concrete implementation strategies.

## Non-constant mass matrix support: technically exists but not recommended

SciML's official documentation states plainly: **"Non-constant mass matrices are not directly supported: users are advised to transform their problem through substitution to a DAE with constant mass matrices."** [sciml +3](https://docs.sciml.ai/DiffEqDocs/latest/solvers/dae_solve/) However, the situation is more nuanced than this suggests.

A hidden mechanism using `DiffEqArrayOperator` with an `update_func` parameter was introduced in DifferentialEquations.jl v6.12.0 (March 2020). This allows specifying state and time-dependent mass matrices: [sciml](https://sciml.ai/news/2020/03/23/DAE/index.html) [Sciml](https://sciml.ai/news/2020/03/23/DAE/index.html)

```julia
function update_func(M, u, p, t)
    M[1,1] = cos(t)
    M[2,2] = u[1]  # State-dependent entry
end
M = DiffEqArrayOperator(ones(3,3), update_func=update_func)
prob = ODEProblem(ODEFunction(f, mass_matrix=M), u0, tspan)
sol = solve(prob, RadauIIA5())
```

**Only `RadauIIA5`** is confirmed to properly call the update function during integration. [Sciml](https://sciml.ai/news/2020/03/23/DAE/) [Sciml](https://sciml.ai/news/2020/03/23/DAE/index.html) Rosenbrock methods (`Rodas4`, `Rodas5`, etc.) explicitly only support constant mass matrices. [Sciml](https://docs.sciml.ai/DiffEqDocs/v6.15/solvers/dae_solve/) Chris Rackauckas has noted that "state-dependent mass matrices are inherently pretty unstable" and recommends DAE formulation instead. [julialang](https://discourse.julialang.org/t/state-dependent-mass-matrix/40238)

The conflicting forum posts you've encountered reflect this gap between the feature's existence and its recommended use. Tests exist in the OrdinaryDiffEq.jl repository, but the feature remains undocumented and effectively experimental.

## Reformulating to constant mass matrix: the charge-based approach

For a voltage-dependent capacitor with C(V), the standard formulation creates problems:

```
I = d(Q)/dt = d(C(V)·V)/dt = C(V)·dV/dt + V·(dC/dV)·(dV/dt)
```

This yields a state-dependent mass matrix in MNA form `C(x)·ẋ = f(x)`. The solution is to use **charge q as a state variable** instead of deriving current from voltage:

**Step 1**: Define charge explicitly with `i = dq/dt` (constant coefficient on derivative)

**Step 2**: Add algebraic constraint `q - Q(V) = 0` where Q(V) is your charge-voltage relationship

**Step 3**: The system becomes an index-1 DAE with constant mass matrix:
```
[I  0] [q̇]   [f₁(q,V)]
[0  0] [V̇] = [q - Q(V)]
```

This introduces additional variables but guarantees charge conservation and eliminates the state-dependent mass matrix entirely. The incremental capacitance `C_inc = dQ/dV = C(V) + V·dC/dV` appears only in the algebraic Jacobian, not the mass matrix.

For implementation, you'd modify your MNA stamping to add charge variables for each nonlinear capacitor and stamp the constitutive relationship `q = Q(V)` as an algebraic constraint in rows with zero mass matrix entries.

## How SPICE reconciles DAE formulations with ODE integrators

SPICE doesn't solve DAEs in the modern numerical sense—it uses **companion models** that discretize energy-storage components into equivalent resistive circuits at each timestep, converting the DAE into a sequence of purely algebraic problems.

For backward Euler applied to a capacitor `I = C·dV/dt`: [Illinois](http://emlab.illinois.edu/ece546/Lect_16.pdf)

```
I^{n+1} = (C/h)·V^{n+1} - (C/h)·V^n = G_eq·V^{n+1} + I_eq
```

At each timestep, the capacitor becomes a **conductance G_eq = C/h in parallel with a current source I_eq = -C·V^n/h**. [SourceForge](https://qucs.sourceforge.net/tech/node26.html) The "history" from previous timesteps is baked into I_eq. The entire circuit then solves as a nonlinear resistive network using Newton-Raphson.

For **voltage-dependent capacitors**, SPICE uses charge-based evaluation within Newton iterations:

```
I^{n+1,m+1} = (1/h)·[Q(V^{n+1,m}) + C^m·(V^{n+1,m+1} - V^{n+1,m}) - Q(V^n)]
```

The indices represent: n = timestep, m = Newton iteration. The capacitance C^m = dQ/dV is evaluated at each iteration, but Q^n (charge at the previous timestep) is computed once and held constant. [SourceForge](https://qucs.sourceforge.net/tech/node26.html) This is why you "can't reference previous time steps" directly—SPICE's architecture computes history terms at timestep boundaries, then solves the algebraic system.

**The key insight**: SPICE doesn't have a state-dependent mass matrix problem because it never formulates the problem that way. The companion model transforms derivatives into algebraic relationships before any matrix inversion occurs.

## Available DAE solvers in SciML for circuit simulation

For `DAEProblem` (fully implicit form `f(du, u, p, t) = 0`):

| Solver | Best For | Notes |
|--------|----------|-------|
| **IDA** | Float64 production work | SUNDIALS wrapper, excellent for large systems, use with `:KLU` for sparse |
| **DFBDF** | Julia-native, arbitrary precision | Variable-order BDF, supports GPUs, recommended when IDA unavailable |
| **DABDF2** | General purpose | 2nd order A-L stable adaptive BDF |
| **DImplicitEuler** | Discontinuous problems | 1st order, handles non-smooth switching |

For `ODEProblem` with singular mass matrix (semi-explicit form):

- **Rosenbrock methods** (`Rodas4`, `Rodas5`, `Rodas5P`): Best for low-medium accuracy, constant mass only [Sciml](https://docs.sciml.ai/DiffEqDocs/latest/solvers/dae_solve/) [GitHub](https://github.com/SciML/DiffEqDocs.jl/blob/master/docs/src/solvers/dae_solve.md)
- **RadauIIA5**: High accuracy FIRK, only confirmed solver for non-constant mass
- **BDF methods** (`QNDF`, `FBDF`): Variable order, sparse support

**For MNA circuits**: Use `DAEProblem` with IDA and `:KLU` sparse linear solver for production work. This handles state-dependent systems naturally through the residual formulation:

```julia
function circuit_residual!(resid, du, u, p, t)
    # Compute M(u) and f(u) for your MNA system
    M = compute_mass_matrix(u)
    f = compute_rhs(u)
    resid .= M * du - f
end
prob = DAEProblem(circuit_residual!, du0, u0, tspan, 
                  differential_vars=diff_vars)
sol = solve(prob, IDA(linear_solver=:KLU))
```

## Best practical approach for voltage-dependent capacitors

**Recommended path**: Reformulate to charge-based DAEProblem with constant mass matrix.

1. **Identify all voltage-dependent capacitors** in your circuit
2. **Add charge state variables** `q_i` for each
3. **Replace** `I_C = C(V)·dV/dt` with:
   - Differential equation: `dq/dt = I_branch` (from KCL)
   - Algebraic constraint: `q - Q(V) = 0`
4. **Stamp into MNA** with charge variables and algebraic rows
5. **Solve with IDA or DFBDF**

If you must use the `ODEProblem` + mass matrix approach:

```julia
# Only use RadauIIA5, explicitly construct mass matrix update
M = DiffEqArrayOperator(M0, update_func=update_mass_matrix!)
f = ODEFunction(mna_rhs!, mass_matrix=M)
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, RadauIIA5())
```

This is experimental and may exhibit instabilities for stiff circuits.

## Why SciML differs from SPICE's simple solvers

SPICE's simplicity comes from its **companion model architecture**—it never solves a DAE directly. Each integration step produces an algebraic system (G·x = b with sources encoding history), solved by Newton-Raphson on a resistive network.

SciML takes a different approach: it formulates the continuous-time problem and applies general-purpose DAE/ODE integrators. This is more mathematically principled but means you can't simply "use Backward Euler" the way SPICE does—SciML's `ImplicitEuler` expects the problem in standard form, not pre-discretized companion model form.

**To emulate SPICE in SciML**, you'd implement the companion model discretization yourself:

```julia
function spice_step!(u_new, u_old, h, circuit)
    # Build conductance matrix G with companion stamps
    for cap in circuit.capacitors
        G_eq = cap.C / h
        I_eq = -cap.C * u_old[cap.node] / h
        # Stamp G_eq into G, I_eq into RHS
    end
    # Solve G·u_new = RHS using Newton-Raphson
end
```

This sacrifices SciML's adaptive timestepping and error control but gains the simplicity of SPICE's approach. For circuits where SPICE methods are appropriate (stiff, switching, many nonlinear elements), this manual approach may outperform general DAE solvers.

## Generating constant mass matrices from Verilog analysis

For automated flows processing Verilog-A models, the strategy is to:

1. **Parse charge expressions** `Q(V)` from the Verilog-A model definitions
2. **Generate charge state variables** for each node with reactive contributions
3. **Emit algebraic constraints** relating charges to voltages via Q(V)
4. **Construct constant mass matrix** with ones for charge derivatives, zeros elsewhere

ModelingToolkit.jl can help with symbolic manipulation—if you generate MTK symbolic equations from your Verilog parser, MTK's `structural_simplify` and `dae_index_lowering` can automatically handle index reduction and mass matrix construction.

## Conclusion

The path forward for SciML-based circuit simulation with voltage-dependent capacitors is clear: **reformulate using charge variables** to obtain constant mass matrices, then use `DAEProblem` with IDA or DFBDF. The experimental `DiffEqArrayOperator` mechanism for state-dependent mass matrices exists but is unstable and poorly documented.

SPICE's elegance comes from never confronting the mass matrix problem directly—companion models transform derivatives into algebraic source terms at each timestep. Replicating this in SciML requires either manual discretization (losing adaptivity) or the charge-based reformulation (adding algebraic constraints). For production circuit simulation, the latter is the mathematically sound choice that leverages SciML's DAE solver infrastructure.
