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

export va_ddt, stamp_contribution!, ContributionTag, JacobianTag

#==============================================================================#
# S-Dual for ddt() (time derivative in Laplace domain)
#==============================================================================#

"""
    ContributionTag

Tag type for ForwardDiff duals used in VA contribution evaluation.
Distinguishes contribution duals from other ForwardDiff usage.

ContributionTag is defined to have the highest precedence among our tags,
so it becomes the outermost dual when nested with voltage duals.
"""
struct ContributionTag end

"""
    JacobianTag

Tag type for ForwardDiff duals used in voltage Jacobian extraction.
This is the "inner" dual that carries ∂/∂V partial derivatives.

Precedence ordering: JacobianTag ≺ ContributionTag
- JacobianTag is inner (lower precedence) - carries voltage partials
- ContributionTag is outer (higher precedence) - separates resistive/reactive

Both tags are distinct from any external ForwardDiff tags (e.g., sensitivity analysis),
preventing confusion when duals are passed through from outer contexts.
"""
struct JacobianTag end

# Define tag ordering for our two tags:
# In ForwardDiff, A ≺ B means A is "inner" (lower precedence), B is "outer" (higher precedence)
#
# Ordering: External tags ≺ JacobianTag ≺ ContributionTag
# - ContributionTag is outermost (separates resistive/reactive via va_ddt)
# - JacobianTag is middle (carries ∂/∂V partials for Jacobian extraction)
# - External tags (e.g., sensitivity) are innermost

# Self-comparisons: no tag is less than itself
ForwardDiff.:≺(::Type{ContributionTag}, ::Type{ContributionTag}) = false
ForwardDiff.:≺(::Type{JacobianTag}, ::Type{JacobianTag}) = false

# Explicit ordering between our two tags
ForwardDiff.:≺(::Type{JacobianTag}, ::Type{ContributionTag}) = true   # JacobianTag IS inner to ContributionTag
ForwardDiff.:≺(::Type{ContributionTag}, ::Type{JacobianTag}) = false  # ContributionTag is NOT inner to JacobianTag

# ContributionTag vs external tags: ContributionTag is outermost
ForwardDiff.:≺(::Type{ContributionTag}, ::Type) = false  # ContributionTag is NOT less than any external tag
ForwardDiff.:≺(::Type, ::Type{ContributionTag}) = true   # All external tags ARE less than ContributionTag

# JacobianTag vs external tags: JacobianTag is outer to external tags
ForwardDiff.:≺(::Type{JacobianTag}, ::Type) = false      # JacobianTag is NOT less than external tags
ForwardDiff.:≺(::Type, ::Type{JacobianTag}) = true       # External tags ARE less than JacobianTag

# Explicit rules for ForwardDiff.Tag{F,V} parametric types
# ForwardDiff.Tag is INNERMOST - lower precedence than our tags
# This means when ForwardDiff.derivative wraps values, our tags stay outer
ForwardDiff.:≺(::Type{JacobianTag}, ::Type{ForwardDiff.Tag{F,V}}) where {F,V} = false
ForwardDiff.:≺(::Type{ForwardDiff.Tag{F,V}}, ::Type{JacobianTag}) where {F,V} = true
ForwardDiff.:≺(::Type{ContributionTag}, ::Type{ForwardDiff.Tag{F,V}}) where {F,V} = false
ForwardDiff.:≺(::Type{ForwardDiff.Tag{F,V}}, ::Type{ContributionTag}) where {F,V} = true

#==============================================================================#
# Voltage-Dependent Capacitor Detection (Charge Formulation Support)
#
# To detect if a capacitor is voltage-dependent (C(V) = ∂Q/∂V varies with V),
# we compare C at two different voltages using first-order AD.
#
# See doc/voltage_dependent_capacitors.md for full design.
#==============================================================================#

export is_voltage_dependent_charge


"""
    is_voltage_dependent_charge(contrib_fn, Vp::Real, Vn::Real) -> Bool

Detect if a contribution contains voltage-dependent capacitance.

Compares capacitance C = ∂Q/∂V at several voltages using ForwardDiff.
If C differs between voltages, the capacitance is voltage-dependent
and requires charge formulation for a constant mass matrix.

# Arguments
- `contrib_fn`: Function `V -> I` that may contain `va_ddt(Q(V))`
- `Vp`, `Vn`: Operating point voltages (unused, kept for API compatibility)

# Returns
- `true` if the contribution has voltage-dependent capacitance
- `false` if purely resistive or has constant capacitance

# Examples
```julia
# Linear capacitor: I = C * ddt(V) -> C is constant
is_voltage_dependent_charge(V -> 1e-12 * va_ddt(V), 1.0, 0.0)  # false

# Nonlinear capacitor: I = ddt(C0 * V^2) -> C = 2*C0*V varies with V
is_voltage_dependent_charge(V -> va_ddt(1e-12 * V^2), 1.0, 0.0)  # true

# Junction cap: I = ddt(Cj0 * φ * (1 - (1-V/φ)^(1-m))) -> C varies with V
is_voltage_dependent_charge(V -> va_ddt(...), 0.3, 0.0)  # true
```

# Implementation
Uses ForwardDiff.derivative at multiple voltage points:
1. Extract charge Q(V) via the reactive part of ContributionTag dual
2. Compute C = dQ/dV using ForwardDiff.derivative
3. If capacitances differ significantly across voltages, it's voltage-dependent

Tag ordering rules ensure ForwardDiff.Tag{F,V} is inner to our tags.
"""
@inline function is_voltage_dependent_charge(contrib_fn, Vp::Real, Vn::Real)::Bool
    # Extract charge from contribution - returns the reactive part
    # NOTE: Detection runs BEFORE dual_creation (see vasim.jl detection_block)
    # so contrib_fn receives plain Float64 values, not JacobianTag duals
    function extract_charge(Vpn)
        result = contrib_fn(Vpn)
        if result isa Dual{ContributionTag}
            return partials(result, 1)
        else
            # No reactive component
            return zero(Vpn)
        end
    end

    # Test at several voltages to avoid special cases
    test_voltages = (0.17, 0.42, 0.73, 0.91)

    # Compute reference capacitance at first voltage
    C_ref = ForwardDiff.derivative(extract_charge, test_voltages[1])

    # Check if capacitance differs at other voltages
    for V in test_voltages[2:end]
        C = ForwardDiff.derivative(extract_charge, V)

        # If capacitance differs significantly, it's voltage-dependent
        diff = abs(C - C_ref)
        maxC = max(abs(C), abs(C_ref))

        if diff > 1e-12 && (maxC < 1e-30 || diff / maxC > 1e-6)
            return true
        end
    end

    return false
end

"""
    detect_or_cached!(ctx::MNAContext, name::Symbol, contrib_fn, Vp::Real, Vn::Real) -> Bool

Check cache for voltage-dependent charge detection result, or run detection if not cached.

This is a build-time optimization: detection only needs to run once per branch.
After the first call, results are cached in the context.

# Arguments
- `ctx`: MNA context containing the detection cache
- `name`: Unique name for this branch (used as cache key)
- `contrib_fn`: Contribution function `V -> I` that may contain `va_ddt(Q(V))`
- `Vp`, `Vn`: Operating point voltages

# Returns
- `true` if the contribution has voltage-dependent capacitance
- `false` if purely resistive or constant capacitance

# Example
```julia
# In generated stamp code:
_is_vdep = detect_or_cached!(ctx, :Q_branch1, V -> va_ddt(C0 * V^2), Vp, Vn)
if _is_vdep
    # Use charge formulation
else
    # Use C matrix
end
```
"""
@inline function detect_or_cached!(ctx::MNAContext, name::Symbol, contrib_fn, Vp::Real, Vn::Real)::Bool
    cache = ctx.charge_is_vdep
    cached = get(cache, name, nothing)
    if cached !== nothing
        return cached
    end
    # First time: run detection via finite-difference (comparing C at two voltages)
    result = is_voltage_dependent_charge(contrib_fn, Vp, Vn)
    cache[name] = result
    return result
end

export detect_or_cached!

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
    Vp::Real=0.0, Vn::Real=0.0, has_reactive::Bool=true
)
    # Stamp conductance (Jacobian of resistive current)
    stamp_G!(ctx, p, p,  I_jac_p)
    stamp_G!(ctx, p, n,  I_jac_n)
    stamp_G!(ctx, n, p, -I_jac_p)
    stamp_G!(ctx, n, n, -I_jac_n)

    # Only stamp capacitance if the device has reactive (ddt) components.
    # Devices WITHOUT ddt() should not stamp into C at all.
    # Devices WITH ddt() must always stamp (even if values are zero at this
    # operating point) to maintain consistent COO structure for precompilation.
    if has_reactive
        stamp_C!(ctx, p, p,  q_jac_p)
        stamp_C!(ctx, p, n,  q_jac_n)
        stamp_C!(ctx, n, p, -q_jac_p)
        stamp_C!(ctx, n, n, -q_jac_n)
    end

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
function evaluate_contribution(contrib_fn, Vp::Real, Vn::Real)
    # Create voltage duals for Jacobian computation
    # JacobianTag ≺ ContributionTag ensures ContributionTag is always outer
    Vp_dual = Dual{JacobianTag}(Vp, one(Vp), zero(Vp))  # ∂/∂Vp = 1, ∂/∂Vn = 0
    Vn_dual = Dual{JacobianTag}(Vn, zero(Vn), one(Vn))  # ∂/∂Vp = 0, ∂/∂Vn = 1

    # Compute contribution with branch voltage
    Vpn_dual = Vp_dual - Vn_dual  # ∂Vpn/∂Vp = 1, ∂Vpn/∂Vn = -1

    result = contrib_fn(Vpn_dual)

    # Result structure depends on whether va_ddt() was called:
    #
    # 1. Pure resistive (no ddt): Dual{JacobianTag}(I, dI/dVp, dI/dVn)
    # 2. Has ddt (pure or mixed): Dual{ContributionTag}(resist_dual, react_dual)
    #    where resist_dual = Dual{JacobianTag}(I, dI/dVp, dI/dVn)
    #    and   react_dual  = Dual{JacobianTag}(q, dq/dVp, dq/dVn)
    #
    # Tag ordering (JacobianTag ≺ ContributionTag) guarantees ContributionTag is always
    # the outer wrapper when present. No need to check for reversed nesting.

    if result isa ForwardDiff.Dual{ContributionTag}
        # Has ddt: ContributionTag wraps voltage duals
        resist_dual = value(result)       # Dual{JacobianTag} for I and ∂I/∂V
        react_dual = partials(result, 1)  # Dual{JacobianTag} for q and ∂q/∂V

        I = Float64(value(resist_dual))
        dI_dVp = Float64(partials(resist_dual, 1))
        dI_dVn = Float64(partials(resist_dual, 2))

        q = Float64(value(react_dual))
        dq_dVp = Float64(partials(react_dual, 1))
        dq_dVn = Float64(partials(react_dual, 2))
        has_reactive = true

    elseif result isa ForwardDiff.Dual
        # Pure resistive: just voltage dual, no ContributionTag
        I = Float64(value(result))
        dI_dVp = Float64(partials(result, 1))
        dI_dVn = Float64(partials(result, 2))
        q = 0.0
        dq_dVp = 0.0
        dq_dVn = 0.0
        has_reactive = false

    else
        # Scalar result (constant contribution)
        I = Float64(result)
        dI_dVp = 0.0
        dI_dVn = 0.0
        q = 0.0
        dq_dVp = 0.0
        dq_dVn = 0.0
        has_reactive = false
    end

    return (
        I = I,
        dI_dVp = dI_dVp,
        dI_dVn = dI_dVn,
        q = q,
        dq_dVp = dq_dVp,
        dq_dVn = dq_dVn,
        has_reactive = has_reactive,
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
        Vp=Vp, Vn=Vn, has_reactive=result.has_reactive)

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
    # Create voltage duals for Jacobian computation
    Vp_dual = Dual{JacobianTag}(Vp, one(Vp), zero(Vp))  # ∂/∂Vp = 1, ∂/∂Vn = 0
    Vn_dual = Dual{JacobianTag}(Vn, zero(Vn), one(Vn))  # ∂/∂Vp = 0, ∂/∂Vn = 1

    # Compute q with branch voltage
    Vpn_dual = Vp_dual - Vn_dual  # ∂Vpn/∂Vp = 1, ∂Vpn/∂Vn = -1

    result = q_fn(Vpn_dual)

    # Result should be Dual{JacobianTag}(q, dq/dVp, dq/dVn) - no ContributionTag
    q = Float64(value(result))
    dq_dVp = Float64(partials(result, 1))
    dq_dVn = Float64(partials(result, 2))

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
    # Always stamp - this function is explicitly for charge contributions
    # so it always has reactive components
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
        Dual{JacobianTag}(V[i], partials...)
    end

    # Evaluate all charges
    q_duals = q_fn(V_duals...)

    # Extract values and stamp
    # Result qi should be Dual{JacobianTag}(qi, ∂qi/∂V1, ..., ∂qi/∂VN) - no ContributionTag
    for i in 1:N
        qi = q_duals[i]
        node_i = nodes[i]

        if node_i != 0
            for j in 1:N
                node_j = nodes[j]
                if node_j != 0
                    # ∂qi/∂Vj - always stamp for charge contributions
                    dqi_dVj = Float64(partials(qi, j))
                    stamp_C!(ctx, node_i, node_j, dqi_dVj)
                end
            end
        end
    end

    return nothing
end

export stamp_multiport_charge!

#==============================================================================#
# Charge State Variable Stamping (Voltage-Dependent Capacitors)
#
# For voltage-dependent capacitors, we reformulate using charge as an explicit
# state variable to achieve a constant mass matrix. See doc/voltage_dependent_capacitors.md
#==============================================================================#

export stamp_charge_state!

"""
    stamp_charge_state!(ctx, p, n, q_fn, x, charge_name) -> Int

Stamp a voltage-dependent capacitor using charge formulation for constant mass matrix.

Instead of stamping C(V) into the mass matrix (which would be state-dependent),
this adds charge `q` as an explicit state variable with:
- `dq/dt = I` (differential equation, constant mass entry of 1)
- `q = Q(V)` (algebraic constraint enforced via Newton)

This yields a constant mass matrix suitable for SciML Rosenbrock solvers.

# Arguments
- `ctx`: MNA context
- `p`, `n`: Branch nodes (current flows from p to n)
- `q_fn`: Function `V -> Q(V)` that computes charge from branch voltage
- `x`: Current solution vector
- `charge_name`: Unique name for the charge variable

# Returns
The system index of the allocated charge variable.

# MNA Formulation

The charge formulation adds these equations:

1. **KCL coupling** (current I = dq/dt flows from p to n):
   - `C[p, q_idx] = +1`  (current leaves p)
   - `C[n, q_idx] = -1`  (current enters n)

2. **Charge dynamics** (dq/dt with constant mass):
   - `C[q_idx, q_idx] = 1`  (constant coefficient!)

3. **Algebraic constraint** (q - Q(V) = 0):
   - `G[q_idx, q_idx] = 1`  (∂F/∂q = 1)
   - `G[q_idx, p] = -∂Q/∂Vp`
   - `G[q_idx, n] = -∂Q/∂Vn`
   - `b[q_idx]` = Newton companion for constraint

# Example
```julia
# Nonlinear junction capacitor: Q(V) = Cj0 * φ * (1 - (1-V/φ)^(1-m))
Cj0, phi, m = 1e-12, 0.7, 0.5
q_fn(V) = Cj0 * phi * (1 - (1 - V/phi)^(1-m)) / (1-m)

q_idx = stamp_charge_state!(ctx, a, c, q_fn, x, :Q_Cj_D1)
```

See doc/voltage_dependent_capacitors.md for full derivation.
"""
function stamp_charge_state!(
    ctx::MNAContext,
    p::Int, n::Int,
    q_fn,
    x::AbstractVector,
    charge_name::Symbol
)
    # Allocate charge state variable
    q_idx = alloc_charge!(ctx, charge_name, p, n)

    # Get operating point voltages
    Vp = p == 0 ? 0.0 : x[p]
    Vn = n == 0 ? 0.0 : x[n]

    # Get current charge value (for Newton companion)
    # If x doesn't have the charge variable yet, use the equilibrium value
    q_current = length(x) >= q_idx ? x[q_idx] : 0.0

    # Evaluate charge function and capacitances
    result = evaluate_charge_contribution(q_fn, Vp, Vn)

    # --- 1. Mass matrix entries (constant coefficients!) ---
    #
    # NOTE: The constraint row (q_idx) is ALGEBRAIC - no C entry on diagonal.
    # The dq/dt terms only appear in the KCL equations (rows p and n).
    # This correctly models: q = Q(V) as an algebraic constraint, with
    # the current I = dq/dt appearing in KCL.

    # KCL coupling: current I = dq/dt flows from p to n
    # At node p: current leaves → +dq/dt contribution
    # At node n: current enters → -dq/dt contribution
    if p != 0
        stamp_C!(ctx, p, q_idx, 1.0)
    end
    if n != 0
        stamp_C!(ctx, n, q_idx, -1.0)
    end

    # --- 2. Conductance matrix (constraint Jacobian) ---

    # Constraint equation: F = q - Q(V) = 0
    # Jacobian: ∂F/∂q = 1, ∂F/∂Vp = -∂Q/∂Vp, ∂F/∂Vn = -∂Q/∂Vn

    stamp_G!(ctx, q_idx, q_idx, 1.0)

    if p != 0
        stamp_G!(ctx, q_idx, p, -result.dq_dVp)
    end
    if n != 0
        stamp_G!(ctx, q_idx, n, -result.dq_dVn)
    end

    # --- 3. RHS (Newton companion model for constraint) ---

    # Newton companion: b = F(x₀) - J(x₀)*x₀ + J(x₀)*x
    # Rearranging: J*x = b where b = F(x₀) - J*x₀
    #
    # For constraint F = q - Q(V):
    # F(x₀) = q₀ - Q(Vp₀, Vn₀) = q₀ - result.q
    #
    # J*x₀ = 1*q₀ + (-∂Q/∂Vp)*Vp₀ + (-∂Q/∂Vn)*Vn₀
    #      = q₀ - result.dq_dVp*Vp₀ - result.dq_dVn*Vn₀
    #
    # b = F(x₀) - J*x₀
    #   = (q₀ - result.q) - (q₀ - result.dq_dVp*Vp₀ - result.dq_dVn*Vn₀)
    #   = -result.q + result.dq_dVp*Vp₀ + result.dq_dVn*Vn₀
    #
    # Note: This ensures that at equilibrium (q = Q(V)), the constraint is satisfied.

    b_constraint = -result.q + result.dq_dVp * Vp + result.dq_dVn * Vn
    stamp_b!(ctx, q_idx, b_constraint)

    return q_idx
end

"""
    stamp_charge_state!(ctx, p, n, q_fn, x, charge_name::String) -> Int

String convenience overload.
"""
stamp_charge_state!(ctx::MNAContext, p::Int, n::Int, q_fn, x::AbstractVector, charge_name::String) =
    stamp_charge_state!(ctx, p, n, q_fn, x, Symbol(charge_name))

#==============================================================================#
# Automatic Detection and Stamping for VA Code Generation
#==============================================================================#

export stamp_reactive_with_detection!

"""
    stamp_reactive_with_detection!(ctx, p, n, contrib_fn, x, charge_name) -> Bool

Detect if a contribution has voltage-dependent capacitance and stamp accordingly.

This function is designed for use by VA code generation. It:
1. Detects if the contribution has voltage-dependent capacitance
2. If voltage-dependent: uses charge formulation (stamp_charge_state!)
3. If constant capacitance: uses standard C matrix stamping

# Arguments
- `ctx`: MNA context
- `p`, `n`: Branch nodes
- `contrib_fn`: Function `V -> I` that may contain `va_ddt(Q(V))`
- `x`: Current solution vector
- `charge_name`: Name for charge variable (used if voltage-dependent)

# Returns
- `true` if charge formulation was used (voltage-dependent)
- `false` if standard C matrix stamping was used (constant capacitance)

# Example
```julia
# In generated VA stamp code:
contrib_fn = V -> V/R + va_ddt(C * V^2)  # Voltage-dependent capacitor

used_charge = stamp_reactive_with_detection!(ctx, p, n, contrib_fn, x, :Q_branch)
# Returns true, allocated charge variable, and stamped charge formulation
```
"""
function stamp_reactive_with_detection!(
    ctx::MNAContext,
    p::Int, n::Int,
    contrib_fn,
    x::AbstractVector,
    charge_name::Symbol
)
    # Get operating point
    Vp = p == 0 ? 0.0 : (length(x) >= p ? x[p] : 0.0)
    Vn = n == 0 ? 0.0 : (length(x) >= n ? x[n] : 0.0)

    # Detect voltage dependence
    is_vdep = is_voltage_dependent_charge(contrib_fn, Vp, Vn)

    if is_vdep
        # Extract charge function from contribution
        # contrib_fn(V) returns I which may include va_ddt(Q(V))
        # We need to extract Q(V) for stamp_charge_state!
        q_fn = V -> begin
            result = contrib_fn(V)
            if result isa Dual{ContributionTag}
                # Extract charge from reactive part
                return value(partials(result, 1))
            else
                return 0.0
            end
        end

        stamp_charge_state!(ctx, p, n, q_fn, x, charge_name)
        return true
    else
        # Use standard C matrix stamping
        result = evaluate_contribution(contrib_fn, Vp, Vn)

        if result.has_reactive
            stamp_C!(ctx, p, p, result.dq_dVp)
            stamp_C!(ctx, p, n, result.dq_dVn)
            stamp_C!(ctx, n, p, -result.dq_dVp)
            stamp_C!(ctx, n, n, -result.dq_dVn)
        end
        return false
    end
end

stamp_reactive_with_detection!(ctx::MNAContext, p::Int, n::Int, contrib_fn, x::AbstractVector, name::String) =
    stamp_reactive_with_detection!(ctx, p, n, contrib_fn, x, Symbol(name))
