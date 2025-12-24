#==============================================================================#
# MNA Phase 5: VA Contribution Function Support
#
# This module provides primitives for stamping Verilog-A style contributions
# into MNA matrices using ForwardDiff automatic differentiation.
#
# Key concept: s-dual approach
#   - Use Dual number with partials=1 as Laplace variable 's'
#   - ddt(x) = s * x automatically separates resistive/reactive parts
#   - value(result) → resistive contribution (stamps into G)
#   - partials(result) → reactive contribution (stamps into C via charge)
#
# See doc/mna_ad_stamping.md and doc/phase5_implementation_plan.md
#==============================================================================#

import ForwardDiff
using ForwardDiff: Dual, value, partials, Tag

export va_ddt, stamp_contribution!, ContributionTag

#==============================================================================#
# S-Dual for ddt() (time derivative in Laplace domain)
#==============================================================================#

"""
    ContributionTag

Tag type for ForwardDiff duals used in VA contribution evaluation.
Distinguishes contribution duals from other ForwardDiff usage.

ContributionTag is defined to have lower precedence than all other tags,
so it becomes the outermost dual when nested with voltage duals.
"""
struct ContributionTag end

# Define tag ordering: ContributionTag < all other tags
# This ensures that when mixing ContributionTag duals with voltage duals,
# the ContributionTag becomes the outer wrapper
ForwardDiff.:≺(::Type{ContributionTag}, ::Type) = true
ForwardDiff.:≺(::Type, ::Type{ContributionTag}) = false
ForwardDiff.:≺(::Type{ContributionTag}, ::Type{ContributionTag}) = false

"""
    va_ddt(x)

Verilog-A ddt() function (time derivative) for MNA contribution stamping.

In the Laplace domain, ddt(x) = s*x where s is the complex frequency.
We represent s as a Dual number with value=0 and partials=1.

When evaluating a contribution like `I(p,n) <+ V/R + C*ddt(V)`:
- The value part gives the resistive contribution: V/R
- The partials part gives the charge: C*V (which stamps as C into the C matrix)

# Example
```julia
# Capacitor: I = C * dV/dt
# In Laplace domain: I = s*C*V
# With s-dual: contribution = C * va_ddt(V) = Dual(0, C*V)
# - value = 0 (no DC current through capacitor)
# - partials = C*V (charge, whose derivative w.r.t. V gives C)
```
"""
@inline function va_ddt(x::Real)
    # s * x where s = Dual(0, 1)
    # Result: Dual(0*x, 1*x) = Dual(0, x)
    return Dual{ContributionTag}(zero(x), x)
end

@inline function va_ddt(x::Dual{ContributionTag,T,N}) where {T,N}
    # s * x where x is already a dual
    # s = Dual(0, 1), so s*x = Dual(0*value(x), 1*value(x) + 0*partials(x))
    # = Dual(0, value(x))
    # But we need to preserve the nested structure for Jacobian extraction
    return Dual{ContributionTag}(zero(T), value(x))
end

# Handle nested duals (for Jacobian computation)
@inline function va_ddt(x::Dual{T,V,N}) where {T,V,N}
    # When x has a different tag (voltage dual for Jacobian)
    # We wrap in ContributionTag dual
    return Dual{ContributionTag}(zero(x), x)
end

#==============================================================================#
# Contribution Evaluation and Stamping
#==============================================================================#

"""
    stamp_contribution!(ctx::MNAContext, p::Int, n::Int, I_val, I_jac_p, I_jac_n, q_val, q_jac_p, q_jac_n)

Low-level stamping of a current contribution with pre-computed values.

# Arguments
- `ctx`: MNA context to stamp into
- `p, n`: Positive and negative node indices (0 = ground)
- `I_val`: Current value at operating point
- `I_jac_p`: ∂I/∂Vp (conductance w.r.t. positive node)
- `I_jac_n`: ∂I/∂Vn (conductance w.r.t. negative node)
- `q_val`: Charge value at operating point
- `q_jac_p`: ∂q/∂Vp (capacitance w.r.t. positive node)
- `q_jac_n`: ∂q/∂Vn (capacitance w.r.t. negative node)

# MNA Formulation
For a branch current I(p,n) flowing from p to n:
- KCL at p: current leaves p → positive contribution
- KCL at n: current enters n → negative contribution

Resistive part stamps:
- G[p,p] += ∂I/∂Vp, G[p,n] += ∂I/∂Vn
- G[n,p] -= ∂I/∂Vp, G[n,n] -= ∂I/∂Vn
- b[p] -= I - G*V (companion model residual)
- b[n] += I - G*V

Reactive part stamps:
- C[p,p] += ∂q/∂Vp, C[p,n] += ∂q/∂Vn
- C[n,p] -= ∂q/∂Vp, C[n,n] -= ∂q/∂Vn
"""
function stamp_contribution!(
    ctx::MNAContext,
    p::Int, n::Int,
    I_val::Real, I_jac_p::Real, I_jac_n::Real,
    q_val::Real, q_jac_p::Real, q_jac_n::Real;
    Vp::Real=0.0, Vn::Real=0.0
)
    # Stamp conductance (Jacobian of resistive current)
    stamp_G!(ctx, p, p,  I_jac_p)
    stamp_G!(ctx, p, n,  I_jac_n)
    stamp_G!(ctx, n, p, -I_jac_p)
    stamp_G!(ctx, n, n, -I_jac_n)

    # Stamp capacitance (Jacobian of charge)
    stamp_C!(ctx, p, p,  q_jac_p)
    stamp_C!(ctx, p, n,  q_jac_n)
    stamp_C!(ctx, n, p, -q_jac_p)
    stamp_C!(ctx, n, n, -q_jac_n)

    # Stamp RHS using Newton companion model
    # For nonlinear I = f(V), the linearization at operating point V0 is:
    #   I_linear = f(V0) + f'(V0) * (V - V0)
    #            = f'(V0) * V + (f(V0) - f'(V0) * V0)
    # In matrix form: G * V = b, where:
    #   G = f'(V0) (Jacobian, already stamped above)
    #   b = I(V0) - dI/dVp * Vp - dI/dVn * Vn
    # This is the Newton companion model for nonlinear elements
    b_companion = I_val - I_jac_p * Vp - I_jac_n * Vn
    stamp_b!(ctx, p, -b_companion)
    stamp_b!(ctx, n,  b_companion)

    return nothing
end

"""
    evaluate_contribution(contrib_fn, Vp::Real, Vn::Real) -> NamedTuple

Evaluate a contribution function and extract all components using AD.

# Arguments
- `contrib_fn`: Function that computes I(Vpn) using va_ddt() for reactive parts
- `Vp`: Voltage at positive node
- `Vn`: Voltage at negative node

# Returns
NamedTuple with:
- `I`: Resistive current value
- `dI_dVp`: ∂I/∂Vp (conductance)
- `dI_dVn`: ∂I/∂Vn (conductance)
- `q`: Charge value
- `dq_dVp`: ∂q/∂Vp (capacitance)
- `dq_dVn`: ∂q/∂Vn (capacitance)

# Example
```julia
# Parallel RC: I = V/R + C*ddt(V)
contrib(V) = V/R + C * va_ddt(V)
result = evaluate_contribution(contrib, 1.0, 0.0)
# result.I ≈ 1.0/R
# result.dI_dVp ≈ 1/R
# result.q ≈ C * 1.0
# result.dq_dVp ≈ C
```
"""
# Helper to extract scalar value from possibly nested duals
_extract_scalar(x::Real) = Float64(x)
function _extract_scalar(x::Dual)
    v = value(x)
    return v isa Dual ? _extract_scalar(v) : Float64(v)
end

function evaluate_contribution(contrib_fn, Vp::Real, Vn::Real)
    # Create duals for voltage differentiation
    # We use a different tag to distinguish from ContributionTag
    Vp_dual = Dual{Nothing}(Vp, one(Vp), zero(Vp))  # ∂/∂Vp = 1, ∂/∂Vn = 0
    Vn_dual = Dual{Nothing}(Vn, zero(Vn), one(Vn))  # ∂/∂Vp = 0, ∂/∂Vn = 1

    # Compute contribution with branch voltage
    Vpn_dual = Vp_dual - Vn_dual  # ∂Vpn/∂Vp = 1, ∂Vpn/∂Vn = -1

    result = contrib_fn(Vpn_dual)

    # The result can have different structures depending on whether the contribution
    # contains only reactive parts, only resistive parts, or both:
    #
    # 1. Pure resistive (V/R): Dual{Nothing}(I, dI/dVp, dI/dVn)
    # 2. Pure reactive (C*ddt(V)): Dual{ContributionTag}(Dual{Nothing}(0,...), Dual{Nothing}(q,...))
    # 3. Mixed (V/R + C*ddt(V)): Dual{Nothing}(Dual{ContributionTag}(...), ...)
    #
    # We need to handle all three cases.

    if result isa Dual{ContributionTag}
        # Cases 2: ContributionTag is outer (pure reactive)
        # value(result) = Dual{Nothing}(I, dI/dVp, dI/dVn) where I=0 for capacitor
        # partials(result,1) = Dual{Nothing}(q, dq/dVp, dq/dVn)
        resist_dual = value(result)    # Dual{Nothing} for I and derivatives
        react_dual = partials(result, 1)  # Dual{Nothing} for q and derivatives

        I = _extract_scalar(value(resist_dual))
        dI_dVp = _extract_scalar(partials(resist_dual, 1))
        dI_dVn = _extract_scalar(partials(resist_dual, 2))

        q = _extract_scalar(value(react_dual))
        dq_dVp = _extract_scalar(partials(react_dual, 1))
        dq_dVn = _extract_scalar(partials(react_dual, 2))

    elseif result isa Dual  # Dual{Nothing} is outer
        val = value(result)
        dVp_part = partials(result, 1)
        dVn_part = partials(result, 2)

        if val isa Dual{ContributionTag}
            # Case 3: Mixed - voltage dual outside, ContributionTag inside
            # value(result) = Dual{ContributionTag}(Dual{Nothing}(I,...), Dual{Nothing}(q,...))
            inner_resist = value(val)  # Could be nested dual
            inner_react = partials(val, 1)

            I = _extract_scalar(inner_resist)
            q = _extract_scalar(inner_react)

            # Get dI/dV from the Jacobian parts
            if dVp_part isa Dual{ContributionTag}
                dI_dVp = _extract_scalar(value(dVp_part))
            else
                dI_dVp = _extract_scalar(dVp_part)
            end

            if dVn_part isa Dual{ContributionTag}
                dI_dVn = _extract_scalar(value(dVn_part))
            else
                dI_dVn = _extract_scalar(dVn_part)
            end

            # Get dq/dV from inner_react partials
            if inner_react isa Dual
                dq_dVp = _extract_scalar(partials(inner_react, 1))
                dq_dVn = _extract_scalar(partials(inner_react, 2))
            else
                dq_dVp = 0.0
                dq_dVn = 0.0
            end
        else
            # Case 1: Pure resistive
            I = _extract_scalar(val)
            dI_dVp = _extract_scalar(dVp_part)
            dI_dVn = _extract_scalar(dVn_part)
            q = 0.0
            dq_dVp = 0.0
            dq_dVn = 0.0
        end
    else
        # Scalar result (shouldn't happen with duals, but handle it)
        I = Float64(result)
        dI_dVp = 0.0
        dI_dVn = 0.0
        q = 0.0
        dq_dVp = 0.0
        dq_dVn = 0.0
    end

    return (
        I = I,
        dI_dVp = dI_dVp,
        dI_dVn = dI_dVn,
        q = q,
        dq_dVp = dq_dVp,
        dq_dVn = dq_dVn,
    )
end

"""
    stamp_current_contribution!(ctx::MNAContext, p::Int, n::Int, contrib_fn, x::AbstractVector)

Stamp a current contribution I(p,n) <+ expr into MNA matrices.

This is the main entry point for VA-style contribution stamping.
Uses ForwardDiff to automatically compute Jacobians.

# Arguments
- `ctx`: MNA context to stamp into
- `p, n`: Node indices (0 = ground)
- `contrib_fn`: Function `V -> I` that may use `va_ddt()` for reactive parts
- `x`: Current solution vector (for node voltages)

# Example
```julia
# Resistor: I(p,n) <+ V(p,n)/R
stamp_current_contribution!(ctx, p, n, V -> V/R, x)

# Capacitor: I(p,n) <+ C*ddt(V(p,n))
stamp_current_contribution!(ctx, p, n, V -> C * va_ddt(V), x)

# Parallel RC: I(p,n) <+ V(p,n)/R + C*ddt(V(p,n))
stamp_current_contribution!(ctx, p, n, V -> V/R + C * va_ddt(V), x)

# Diode: I(p,n) <+ Is*(exp(V(p,n)/Vt) - 1)
stamp_current_contribution!(ctx, p, n, V -> Is*(exp(V/Vt) - 1), x)
```
"""
function stamp_current_contribution!(
    ctx::MNAContext,
    p::Int, n::Int,
    contrib_fn,
    x::AbstractVector
)
    # Get node voltages (ground = 0)
    Vp = p == 0 ? 0.0 : x[p]
    Vn = n == 0 ? 0.0 : x[n]

    # Evaluate contribution and extract all derivatives
    result = evaluate_contribution(contrib_fn, Vp, Vn)

    # Stamp into matrices with operating point voltages for companion model
    stamp_contribution!(ctx, p, n,
        result.I, result.dI_dVp, result.dI_dVn,
        result.q, result.dq_dVp, result.dq_dVn;
        Vp=Vp, Vn=Vn)

    return nothing
end

"""
    stamp_voltage_contribution!(ctx::MNAContext, p::Int, n::Int, v_fn, x::AbstractVector, current_name::Symbol)

Stamp a voltage contribution V(p,n) <+ expr into MNA matrices.

Voltage contributions require a current variable (like voltage sources).
The constraint equation V(p,n) = expr is enforced via an auxiliary current.

# Arguments
- `ctx`: MNA context
- `p, n`: Node indices
- `v_fn`: Function that computes voltage value
- `x`: Current solution vector
- `current_name`: Name for the auxiliary current variable

# MNA Formulation
Adds a current variable I and stamps:
- G[p, I] = 1, G[n, I] = -1 (current enters p, leaves n)
- G[I, p] = 1, G[I, n] = -1 (voltage constraint)
- b[I] = v_fn(x) (voltage value)
"""
function stamp_voltage_contribution!(
    ctx::MNAContext,
    p::Int, n::Int,
    v_fn,
    x::AbstractVector,
    current_name::Symbol
)
    # Allocate current variable
    I_idx = alloc_current!(ctx, current_name)

    # Get voltage value
    Vp = p == 0 ? 0.0 : x[p]
    Vn = n == 0 ? 0.0 : x[n]
    Vpn = Vp - Vn
    v_val = v_fn(Vpn)

    # Stamp voltage source pattern
    stamp_G!(ctx, p, I_idx, 1.0)
    stamp_G!(ctx, n, I_idx, -1.0)
    stamp_G!(ctx, I_idx, p, 1.0)
    stamp_G!(ctx, I_idx, n, -1.0)
    stamp_b!(ctx, I_idx, v_val)

    return I_idx
end

export stamp_current_contribution!, stamp_voltage_contribution!, evaluate_contribution
export stamp_charge_contribution!, evaluate_charge_contribution

#==============================================================================#
# Charge-Based Contribution Stamping (Phase 6: Nonlinear Capacitors)
#==============================================================================#

"""
    evaluate_charge_contribution(q_fn, Vp::Real, Vn::Real) -> NamedTuple

Evaluate a charge function and extract capacitance via AD.

For voltage-dependent capacitors, the charge is q(V) and capacitance is C(V) = dq/dV.
In transient analysis, the current is I = dq/dt = C(V) * dV/dt.

# Arguments
- `q_fn`: Function that computes charge q(V)
- `Vp`: Voltage at positive node
- `Vn`: Voltage at negative node

# Returns
NamedTuple with:
- `q`: Charge value at operating point
- `dq_dVp`: ∂q/∂Vp (capacitance contribution from positive node)
- `dq_dVn`: ∂q/∂Vn (capacitance contribution from negative node)
- `C`: Effective capacitance = dq/dVp - dq/dVn = dq/dVpn

# Example
```julia
# Nonlinear junction capacitance: q(V) = Cj0 * φ * (1 - (1 - V/φ)^(1-m))
q_fn(V) = Cj0 * phi * (1 - (1 - V/phi)^(1-m))
result = evaluate_charge_contribution(q_fn, 0.3, 0.0)
# result.C gives the voltage-dependent capacitance at V=0.3
```
"""
function evaluate_charge_contribution(q_fn, Vp::Real, Vn::Real)
    # Create duals for voltage differentiation
    Vp_dual = Dual{Nothing}(Vp, one(Vp), zero(Vp))  # ∂/∂Vp = 1, ∂/∂Vn = 0
    Vn_dual = Dual{Nothing}(Vn, zero(Vn), one(Vn))  # ∂/∂Vp = 0, ∂/∂Vn = 1

    # Compute q with branch voltage
    Vpn_dual = Vp_dual - Vn_dual  # ∂Vpn/∂Vp = 1, ∂Vpn/∂Vn = -1

    result = q_fn(Vpn_dual)

    # Extract values
    q = _extract_scalar(value(result))
    dq_dVp = _extract_scalar(partials(result, 1))
    dq_dVn = _extract_scalar(partials(result, 2))

    # Effective capacitance for branch:
    # For q(Vpn) where Vpn = Vp - Vn, via chain rule:
    #   dq/dVp = dq/dVpn * dVpn/dVp = dq/dVpn * 1 = dq/dVpn
    #   dq/dVn = dq/dVpn * dVpn/dVn = dq/dVpn * (-1) = -dq/dVpn
    # So the effective capacitance C = dq/dVpn = dq_dVp
    C = dq_dVp

    return (
        q = q,
        dq_dVp = dq_dVp,
        dq_dVn = dq_dVn,
        C = C
    )
end

"""
    stamp_charge_contribution!(ctx::MNAContext, p::Int, n::Int, q_fn, x::AbstractVector)

Stamp a voltage-dependent charge contribution into MNA matrices.

For nonlinear capacitors with charge q(V), this stamps:
- C matrix: capacitance C(V) = dq/dV at the operating point
- b vector: charge residual for Newton companion model

The DAE formulation uses:
    I = dq/dt = C(V) * dV/dt

In the MNA system C*dx/dt + G*x = b, this adds C(V) to the C matrix.

# Arguments
- `ctx`: MNA context to stamp into
- `p, n`: Node indices (0 = ground)
- `q_fn`: Function `V -> q` that computes charge as function of branch voltage
- `x`: Current solution vector (for node voltages)

# Example
```julia
# Linear capacitor: q = C0 * V
stamp_charge_contribution!(ctx, p, n, V -> C0 * V, x)

# Nonlinear junction cap: q = Cj0 * φ * (1 - (1 - V/φ)^(1-m))
stamp_charge_contribution!(ctx, p, n, V -> Cj0 * phi * (1 - (1 - V/phi)^(1-m)), x)

# MOSFET gate charge (multi-terminal): q = f(Vgs, Vds)
# For multi-terminal, use stamp_multiport_charge! instead
```

# MNA Formulation
For charge q(V) at branch (p,n):
- C[p,p] += ∂q/∂Vp, C[p,n] += ∂q/∂Vn
- C[n,p] -= ∂q/∂Vp, C[n,n] -= ∂q/∂Vn

The current I = dq/dt flows from p to n (positive current leaves p).
"""
function stamp_charge_contribution!(
    ctx::MNAContext,
    p::Int, n::Int,
    q_fn,
    x::AbstractVector
)
    # Get node voltages (ground = 0)
    Vp = p == 0 ? 0.0 : x[p]
    Vn = n == 0 ? 0.0 : x[n]

    # Evaluate charge and extract capacitances
    result = evaluate_charge_contribution(q_fn, Vp, Vn)

    # Stamp capacitance (Jacobian of charge w.r.t. voltages)
    stamp_C!(ctx, p, p,  result.dq_dVp)
    stamp_C!(ctx, p, n,  result.dq_dVn)
    stamp_C!(ctx, n, p, -result.dq_dVp)
    stamp_C!(ctx, n, n, -result.dq_dVn)

    # For Newton iteration on DAE, we also need to stamp the charge residual
    # into the RHS. This is handled in the DAE formulation as:
    # F = C*du + G*u - b = 0
    # where u includes node voltages and du includes dV/dt.
    # The charge contribution adds to C, so no direct b stamp needed here.

    return nothing
end

#==============================================================================#
# Multi-Port Charge Stamping (Phase 6: MOSFET Gate/Drain/Source/Body charges)
#==============================================================================#

"""
    stamp_multiport_charge!(ctx::MNAContext, nodes::NTuple{N,Int}, q_fn, x::AbstractVector) where N

Stamp multi-terminal charge contributions into MNA C matrix.

For complex devices like MOSFETs, charges at each terminal depend on multiple
node voltages: qg(Vgs, Vds, Vbs), qd(Vgs, Vds, Vbs), etc.

This function stamps all ∂qi/∂Vj capacitance terms for N-terminal devices.

# Arguments
- `ctx`: MNA context
- `nodes`: Tuple of node indices (d, g, s, b for MOSFET)
- `q_fn`: Function `(V1, V2, ..., VN) -> (q1, q2, ..., qN)` computing terminal charges
- `x`: Current solution vector

# Example
```julia
# MOSFET with 4 terminals (d, g, s, b)
# Charges depend on all terminal voltages relative to source
function mosfet_charges(Vd, Vg, Vs, Vb)
    Vgs = Vg - Vs
    Vds = Vd - Vs
    Vbs = Vb - Vs

    qg = compute_qg(Vgs, Vds, Vbs)
    qd = compute_qd(Vgs, Vds, Vbs)
    qs = -(qg + qd)  # Charge conservation
    qb = compute_qb(Vgs, Vds, Vbs)

    return (qd, qg, qs, qb)
end

stamp_multiport_charge!(ctx, (d, g, s, b), mosfet_charges, x)
```

# MNA Formulation
For each terminal pair (i, j), stamps:
- C[i, j] += ∂qi/∂Vj (capacitance from node j affecting charge at node i)
"""
function stamp_multiport_charge!(
    ctx::MNAContext,
    nodes::NTuple{N,Int},
    q_fn,
    x::AbstractVector
) where N
    # Get node voltages
    V = ntuple(i -> nodes[i] == 0 ? 0.0 : x[nodes[i]], Val(N))

    # Create duals for each node voltage
    # Each dual has partials for all N nodes
    V_duals = ntuple(Val(N)) do i
        partials = ntuple(j -> i == j ? 1.0 : 0.0, Val(N))
        Dual{Nothing}(V[i], partials...)
    end

    # Evaluate all charges
    q_duals = q_fn(V_duals...)

    # Extract values and stamp
    for i in 1:N
        qi = q_duals[i]
        node_i = nodes[i]

        if node_i != 0
            qi_val = _extract_scalar(value(qi))
            for j in 1:N
                node_j = nodes[j]
                if node_j != 0
                    # ∂qi/∂Vj
                    dqi_dVj = _extract_scalar(partials(qi, j))
                    stamp_C!(ctx, node_i, node_j, dqi_dVj)
                end
            end
        end
    end

    return nothing
end

export stamp_multiport_charge!
