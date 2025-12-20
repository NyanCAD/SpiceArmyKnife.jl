# Phase 3: Verilog-A and Analysis Types

## Overview

This phase completes the MNA migration by supporting Verilog-A models, nonlinear devices, and all analysis types (DC, transient, AC, noise). This is the most complex phase as it requires updating the Verilog-A codegen (`vasim.jl`) and implementing the full analysis framework.

## Deliverables

1. `src/mna/nonlinear.jl` - Nonlinear device infrastructure
2. `src/mna/ac.jl` - AC small-signal analysis
3. Updates to `src/vasim.jl` - Emit MNA primitives for VA
4. `src/mna/noise.jl` - Noise analysis framework
5. Full test suite compatibility

## Verilog-A Integration

### Understanding the Current VA Pipeline

The existing `vasim.jl` compiles Verilog-A to Julia code that uses DAECompiler primitives:

```
Verilog-A source
    ↓
VerilogAParser.jl (AST)
    ↓
make_spice_device() in vasim.jl
    ↓
Julia struct + callable using DAECompiler
```

Key DAECompiler primitives used:
- `variable()` - Create state variable
- `equation!()` - Add equation constraint
- `ddt()` - Time derivative
- `branch!()` - Create branch with KCL

### New VA-to-MNA Translation

The new pipeline emits MNA stamping calls:

```
Verilog-A source
    ↓
VerilogAParser.jl (AST)
    ↓
make_spice_device_mna() in vasim.jl
    ↓
Julia struct + MNA evaluate! function
```

### Contribution Detection via Dual Numbers

Use ForwardDiff Dual numbers to detect resistive vs reactive contributions at trace time:

```julia
"""
Time-tagged Dual for detecting ddt() contributions.
The t-partial indicates reactive (ddt) terms.
"""
struct TimeDual{T}
    value::T
    t_partial::T  # ∂/∂t indicates ddt() contribution
end

# ddt() implementation for MNA tracing
function mna_ddt(x::TimeDual)
    # The input has the charge Q
    # ddt(Q) returns the current with t-partial = 1
    TimeDual(x.value, one(x.value))
end

# Detect contribution type from final result
function classify_contribution(result::TimeDual)
    if result.t_partial == 0
        :resistive  # Goes in G matrix
    else
        :reactive   # Goes in C matrix
    end
end
```

### VA Scope Implementation for MNA

```julia
# Modified from vasim.jl for MNA backend

"""
MNA-aware scope for VA code generation.
Tracks node voltages, branch currents, and contributions.
"""
struct MNAVAScope
    ctx::MNAContext
    nodes::Dict{Symbol, Union{Node, GroundNode}}
    branches::Dict{Symbol, Branch}
    parameters::Dict{Symbol, Any}

    # Accumulated contributions
    contributions::Vector{VAContribution}
end

struct VAContribution
    branch::Branch
    resist_expr::Expr   # Resistive part (no ddt)
    react_expr::Expr    # Reactive part (ddt terms)
    is_voltage::Bool    # V(p,n) <+ vs I(p,n) <+
end
```

### Translating VA Constructs

#### I(p,n) <+ expr

```julia
function emit_current_contribution(scope::MNAVAScope, p, n, expr)
    # Create or get branch
    branch = get_or_create_branch!(scope, p, n)

    # At trace time, evaluate with TimeDual to separate resistive/reactive
    # At runtime, stamp into appropriate matrices

    quote
        let val = $expr
            # Resistive contribution to KCL
            stamp_kcl_contribution!(ctx, $(branch.pos), $(branch.neg), val)
        end
    end
end
```

#### I(p,n) <+ ddt(Q_expr)

```julia
function emit_reactive_contribution(scope::MNAVAScope, p, n, Q_expr)
    branch = get_or_create_branch!(scope, p, n)

    quote
        # Q is the charge, dQ/dt is the current
        # Stamp ∂Q/∂V into C matrix
        let Q_func = V -> $Q_expr
            dQ_dV = ForwardDiff.gradient(Q_func, V_current)
            stamp_capacitance_jacobian!(ctx, $(branch), dQ_dV)

            # Residual contribution
            stamp_reactive_residual!(ctx, $(branch), Q_current, du)
        end
    end
end
```

#### V(p,n) <+ expr

```julia
function emit_voltage_contribution(scope::MNAVAScope, p, n, expr)
    # Voltage contributions require a branch current variable
    branch = create_branch!(scope.ctx, gensym(:vbranch), p, n)

    quote
        # KCL: branch current participates
        # Constraint: V(p) - V(n) = expr
        stamp_voltage_constraint!(ctx, $(branch), $expr)
    end
end
```

#### ddx(expr, V(n)) for AC Analysis

The existing `ddx()` implementation using ForwardDiff remains unchanged:

```julia
function emit_ddx(scope::MNAVAScope, expr, var)
    quote
        # ForwardDiff partial for AC small-signal
        ForwardDiff.derivative(x -> let $var = x; $expr end, $var)
    end
end
```

## Nonlinear Device Framework

### Nonlinear Evaluation Interface

```julia
# src/mna/nonlinear.jl

"""
Nonlinear device that requires Newton-Raphson iteration.
"""
abstract type NonlinearMNADevice <: MNADevice end

"""
Evaluate nonlinear device at current operating point.
Returns residual contribution and Jacobian updates.
"""
function evaluate!(device::NonlinearMNADevice,
                   residual::Vector{Float64},
                   G::SparseMatrixCSC,
                   x::Vector{Float64},
                   t::Float64)
    error("evaluate! not implemented for $(typeof(device))")
end
```

### Diode Implementation

```julia
"""
PN junction diode.
I = Is * (exp(V/Vt) - 1)
"""
struct MNADiode <: NonlinearMNADevice
    name::Symbol
    anode::Symbol
    cathode::Symbol
    Is::Float64      # Saturation current
    N::Float64       # Emission coefficient
    Vt::Float64      # Thermal voltage (kT/q)

    # Internal state
    a_idx::Int
    c_idx::Int
end

function setup!(d::MNADiode, ctx::MNAContext)
    d.a_idx = node_index(get_node!(ctx, d.anode))
    d.c_idx = node_index(get_node!(ctx, d.cathode))
end

function evaluate!(d::MNADiode, residual, G, x, t)
    # Diode voltage
    Va = d.a_idx == 0 ? 0.0 : x[d.a_idx]
    Vc = d.c_idx == 0 ? 0.0 : x[d.c_idx]
    Vd = Va - Vc

    # Voltage limiting for convergence (ngspice style)
    Vd = limit_pn_voltage(Vd, d.Vt * d.N)

    # Current and conductance
    Vte = d.N * d.Vt
    Id = d.Is * (exp(Vd / Vte) - 1)
    Gd = d.Is * exp(Vd / Vte) / Vte  # dI/dV

    # Linearized: I ≈ Gd*V + (Id - Gd*Vd)
    Ieq = Id - Gd * Vd  # Equivalent current source

    # Stamp conductance (Jacobian)
    if d.a_idx != 0
        G[d.a_idx, d.a_idx] += Gd
        if d.c_idx != 0
            G[d.a_idx, d.c_idx] -= Gd
        end
    end
    if d.c_idx != 0
        G[d.c_idx, d.c_idx] += Gd
        if d.a_idx != 0
            G[d.c_idx, d.a_idx] -= Gd
        end
    end

    # Stamp equivalent current source into residual
    if d.a_idx != 0
        residual[d.a_idx] += Ieq
    end
    if d.c_idx != 0
        residual[d.c_idx] -= Ieq
    end
end

"""
Voltage limiting for PN junction convergence.
Prevents excessive voltage steps.
"""
function limit_pn_voltage(vnew, vt)
    vcrit = vt * log(vt / (sqrt(2) * 1e-14))  # Critical voltage
    if vnew > vcrit
        vnew = vcrit + vt * log(1 + (vnew - vcrit) / vt)
    end
    return vnew
end
```

### MOSFET Framework

```julia
"""
Generic MOSFET interface for compact models.
Subtype for specific models (BSIM, etc.)
"""
abstract type MNAMosfet <: NonlinearMNADevice end

"""
BSIM4 MOSFET model (simplified structure).
Full implementation would include all BSIM parameters.
"""
struct MNABSIM4 <: MNAMosfet
    name::Symbol
    drain::Symbol
    gate::Symbol
    source::Symbol
    bulk::Symbol

    # Model parameters (hundreds in real BSIM4)
    params::Dict{Symbol, Float64}

    # Node indices
    d_idx::Int
    g_idx::Int
    s_idx::Int
    b_idx::Int

    # Charge storage (for transient)
    Qg::Float64  # Gate charge
    Qd::Float64  # Drain charge
    Qs::Float64  # Source charge
    Qb::Float64  # Bulk charge
end

function evaluate!(m::MNABSIM4, residual, G, C, x, du, t)
    # Extract terminal voltages
    Vd = get_voltage(x, m.d_idx)
    Vg = get_voltage(x, m.g_idx)
    Vs = get_voltage(x, m.s_idx)
    Vb = get_voltage(x, m.b_idx)

    # Compute BSIM4 model (very simplified)
    Vgs = Vg - Vs
    Vds = Vd - Vs
    Vbs = Vb - Vs

    # Drain current (placeholder - real BSIM4 is much more complex)
    Ids, Gm, Gds, Gmbs = compute_bsim4_ids(m.params, Vgs, Vds, Vbs)

    # Charges (nonlinear capacitances)
    Qg, Qd, Qs, Qb = compute_bsim4_charges(m.params, Vgs, Vds, Vbs)

    # Stamp transconductances into G
    stamp_mosfet_conductance!(G, m, Ids, Gm, Gds, Gmbs)

    # Stamp charge Jacobians into C
    stamp_mosfet_capacitance!(C, m, Qg, Qd, Qs, Qb, Vgs, Vds, Vbs)

    # Residual contributions
    stamp_mosfet_residual!(residual, du, m, Ids, Qg, Qd, Qs, Qb)
end
```

## Newton-Raphson Solver for Nonlinear Circuits

```julia
# Enhanced DC solver for nonlinear circuits

"""
Newton-Raphson DC solver with convergence aids.
"""
function solve_dc_newton(circuit::MNACircuit;
                         maxiter = 100,
                         abstol = 1e-12,
                         reltol = 1e-6,
                         gmin_steps = 5,
                         source_steps = 10)
    n = circuit.n_vars
    x = zeros(n)

    # Try direct solve first
    converged, x = newton_iteration(circuit, x, maxiter, abstol, reltol)
    if converged
        return DCSolution(circuit, x), 1
    end

    # GMIN stepping (add small conductance to ground)
    for gmin in logspace(0, -12, gmin_steps)
        circuit_gmin = add_gmin(circuit, gmin)
        converged, x = newton_iteration(circuit_gmin, x, maxiter, abstol, reltol)
        if !converged
            break
        end
    end

    if converged
        return DCSolution(circuit, x), 2
    end

    # Source stepping
    for scale in range(0, 1, length=source_steps)
        circuit_scaled = scale_sources(circuit, scale)
        converged, x = newton_iteration(circuit_scaled, x, maxiter, abstol, reltol)
        if !converged
            error("DC convergence failed at source scale $scale")
        end
    end

    return DCSolution(circuit, x), 3
end

function newton_iteration(circuit, x0, maxiter, abstol, reltol)
    x = copy(x0)
    n = length(x)
    residual = zeros(n)
    G = copy(circuit.G)

    for iter in 1:maxiter
        # Reset to linear part
        copyto!(G, circuit.G)
        residual .= circuit.G * x .- circuit.b

        # Evaluate nonlinear devices
        for device in circuit.devices
            evaluate!(device, residual, G, x, 0.0)
        end

        # Residual-based convergence (VACASK style)
        converged = true
        for i in 1:n
            tol = abs(x[i]) * reltol + abstol
            if abs(residual[i]) > tol
                converged = false
                break
            end
        end

        if converged
            return true, x
        end

        # Newton step
        dx = G \ (-residual)
        x .+= dx
    end

    return false, x
end
```

## AC Small-Signal Analysis

### Linearization at Operating Point

```julia
# src/mna/ac.jl

"""
AC analysis: linearize around DC operating point,
solve (G + jωC) * X = B for frequency sweep.
"""
struct ACAnalysis
    circuit::MNACircuit
    dc_point::Vector{Float64}
    frequencies::Vector{Float64}
end

"""
Perform AC analysis over frequency range.
"""
function solve_ac(circuit::MNACircuit, fstart, fstop, npoints;
                  scale = :log)
    # First, get DC operating point
    dc_sol = solve_dc(circuit)
    x_dc = dc_sol.x

    # Linearize nonlinear devices at DC point
    G_lin = copy(circuit.G)
    C_lin = copy(circuit.C)
    b_ac = zeros(ComplexF64, circuit.n_vars)

    for device in circuit.devices
        linearize_ac!(device, G_lin, C_lin, x_dc)
    end

    # Frequency sweep
    if scale == :log
        freqs = 10 .^ range(log10(fstart), log10(fstop), length=npoints)
    else
        freqs = range(fstart, fstop, length=npoints)
    end

    # Find AC sources and set their contributions
    set_ac_sources!(b_ac, circuit)

    # Solve at each frequency
    results = Vector{Vector{ComplexF64}}(undef, length(freqs))

    for (i, f) in enumerate(freqs)
        ω = 2π * f
        # Y = G + jωC
        Y = G_lin + im * ω * C_lin

        # Solve Y * X = B
        results[i] = Y \ b_ac
    end

    ACSolution(circuit, freqs, results)
end

struct ACSolution
    circuit::MNACircuit
    frequencies::Vector{Float64}
    node_phasors::Vector{Vector{ComplexF64}}
end

"""
Get magnitude and phase at a node across frequencies.
"""
function ac_response(sol::ACSolution, node::Symbol)
    idx = findfirst(==(node), sol.circuit.node_names)
    mags = [abs(sol.node_phasors[i][idx]) for i in eachindex(sol.frequencies)]
    phases = [angle(sol.node_phasors[i][idx]) for i in eachindex(sol.frequencies)]
    (frequencies = sol.frequencies, magnitude = mags, phase = phases)
end
```

### Small-Signal Linearization

```julia
"""
Linearize a nonlinear device for AC analysis.
Compute small-signal conductances at operating point.
"""
function linearize_ac!(device::MNADiode, G, C, x_dc)
    Va = device.a_idx == 0 ? 0.0 : x_dc[device.a_idx]
    Vc = device.c_idx == 0 ? 0.0 : x_dc[device.c_idx]
    Vd = Va - Vc

    # Small-signal conductance
    Vte = device.N * device.Vt
    gd = device.Is * exp(Vd / Vte) / Vte

    # Stamp into linearized G matrix
    if device.a_idx != 0
        G[device.a_idx, device.a_idx] += gd
        device.c_idx != 0 && (G[device.a_idx, device.c_idx] -= gd)
    end
    if device.c_idx != 0
        G[device.c_idx, device.c_idx] += gd
        device.a_idx != 0 && (G[device.c_idx, device.a_idx] -= gd)
    end

    # Junction capacitance would go into C matrix
    # (simplified - real diode has Cj, Cd)
end

function linearize_ac!(device::MNABSIM4, G, C, x_dc)
    # Extract operating point voltages
    Vd = get_voltage(x_dc, device.d_idx)
    Vg = get_voltage(x_dc, device.g_idx)
    Vs = get_voltage(x_dc, device.s_idx)
    Vb = get_voltage(x_dc, device.b_idx)

    Vgs = Vg - Vs
    Vds = Vd - Vs
    Vbs = Vb - Vs

    # Compute small-signal parameters
    _, Gm, Gds, Gmbs = compute_bsim4_ids(device.params, Vgs, Vds, Vbs)

    # Small-signal capacitances
    Cgg, Cgd, Cgs, Cgb = compute_bsim4_small_signal_caps(device.params, Vgs, Vds, Vbs)

    # Stamp transconductances
    stamp_mosfet_ac_conductance!(G, device, Gm, Gds, Gmbs)

    # Stamp capacitances
    stamp_mosfet_ac_capacitance!(C, device, Cgg, Cgd, Cgs, Cgb)
end
```

## Transient Analysis with Nonlinear Devices

```julia
"""
Build DAEProblem for circuit with nonlinear devices.
"""
function build_nonlinear_dae(circuit::MNACircuit, tspan)
    n = circuit.n_vars

    # DC operating point for initial conditions
    dc_sol, _ = solve_dc_newton(circuit)
    u0 = dc_sol.x
    du0 = zeros(n)

    function residual!(res, du, u, p, t)
        # Linear part
        mul!(res, circuit.G, u)
        res .-= circuit.b

        # Reactive linear part
        mul!(res, circuit.C, du, 1.0, 1.0)

        # Nonlinear devices
        for device in circuit.devices
            evaluate_transient!(device, res, u, du, t)
        end
    end

    function jacobian!(J, du, u, p, gamma, t)
        # J = gamma * C + G + nonlinear Jacobian
        J .= gamma .* circuit.C .+ circuit.G

        for device in circuit.devices
            add_jacobian!(device, J, u, gamma)
        end
    end

    jac_prototype = circuit.G + circuit.C  # Sparsity pattern

    dae_fn = DAEFunction(residual!;
        jac = jacobian!,
        jac_prototype = jac_prototype)

    DAEProblem(dae_fn, du0, u0, tspan;
        differential_vars = circuit.differential_vars)
end
```

## vasim.jl Updates

### Mode Switch for MNA Backend

```julia
# In vasim.jl

const VA_MNA_MODE = Ref(false)

"""
Generate MNA-compatible device from Verilog-A module.
"""
function make_spice_device_mna(vm::VANode{VerilogModule})
    mod_name = vm.data.name
    pins = extract_pins(vm)
    params = extract_parameters(vm)
    internal_nodes = extract_internal_nodes(vm)

    # Generate struct
    struct_def = generate_mna_device_struct(mod_name, pins, params, internal_nodes)

    # Generate setup! function
    setup_fn = generate_mna_setup(mod_name, pins, internal_nodes)

    # Generate evaluate! function from analog block
    analog_block = extract_analog_block(vm)
    eval_fn = generate_mna_evaluate(mod_name, pins, params, analog_block)

    quote
        $struct_def
        $setup_fn
        $eval_fn
    end
end

function generate_mna_evaluate(mod_name, pins, params, analog_block)
    # Transform VA contributions to MNA stamps
    body = transform_analog_block_to_mna(analog_block)

    quote
        function evaluate!(device::$mod_name, ctx::MNAContext, x, du, t)
            # Unpack parameters
            $(unpack_params(params))

            # Unpack node voltages
            $(unpack_voltages(pins))

            # Transformed analog block
            $body
        end
    end
end
```

### Contribution Transformation

```julia
"""
Transform VA contribution statement to MNA stamping.
"""
function transform_contribution(stmt, scope)
    if is_current_contribution(stmt)
        # I(p,n) <+ expr
        branch = stmt.branch
        expr = stmt.expr

        if contains_ddt(expr)
            # Split into resistive and reactive parts
            resist, react = split_contribution(expr)
            quote
                # Resistive part → G matrix
                stamp_resist_contribution!(ctx, $branch, $resist, x)
                # Reactive part → C matrix
                stamp_react_contribution!(ctx, $branch, $react, x, du)
            end
        else
            quote
                stamp_resist_contribution!(ctx, $branch, $expr, x)
            end
        end

    elseif is_voltage_contribution(stmt)
        # V(p,n) <+ expr
        branch = stmt.branch
        expr = stmt.expr
        quote
            stamp_voltage_constraint!(ctx, $branch, $expr, x)
        end
    end
end
```

## Noise Analysis (Framework)

```julia
# src/mna/noise.jl

"""
Noise analysis at operating point.
Computes noise contributions from devices and propagates to output.
"""
struct NoiseAnalysis
    circuit::MNACircuit
    frequencies::Vector{Float64}
    output_node::Symbol
    input_source::Symbol  # Reference source for input-referred noise
end

"""
Noise contribution from a device.
"""
struct NoiseContribution
    device::Symbol
    type::Symbol  # :thermal, :shot, :flicker
    psd::Float64  # Power spectral density (A²/Hz or V²/Hz)
    node_p::Int
    node_n::Int
end

"""
Compute noise at output node.
"""
function solve_noise(analysis::NoiseAnalysis)
    circuit = analysis.circuit

    # Get DC operating point
    dc_sol = solve_dc(circuit)
    x_dc = dc_sol.x

    # Linearize circuit at DC point
    G_lin, C_lin = linearize_circuit(circuit, x_dc)

    output_idx = findfirst(==(analysis.output_node), circuit.node_names)

    results = []

    for f in analysis.frequencies
        ω = 2π * f
        Y = G_lin + im * ω * C_lin

        # Compute transfer function from each noise source to output
        total_noise_psd = 0.0

        for device in circuit.devices
            contributions = get_noise_sources(device, x_dc, f)

            for nc in contributions
                # Inject unit noise current, measure output
                b_noise = zeros(ComplexF64, circuit.n_vars)
                b_noise[nc.node_p] = 1.0
                if nc.node_n != 0
                    b_noise[nc.node_n] = -1.0
                end

                v_out = (Y \ b_noise)[output_idx]
                transfer = abs(v_out)^2

                total_noise_psd += nc.psd * transfer
            end
        end

        push!(results, (f, sqrt(total_noise_psd)))
    end

    results
end

"""
Get noise sources from diode at operating point.
"""
function get_noise_sources(d::MNADiode, x_dc, f)
    Va = d.a_idx == 0 ? 0.0 : x_dc[d.a_idx]
    Vc = d.c_idx == 0 ? 0.0 : x_dc[d.c_idx]
    Vd = Va - Vc

    Id = d.Is * (exp(Vd / (d.N * d.Vt)) - 1)

    # Shot noise: PSD = 2qI
    shot_psd = 2 * 1.602e-19 * abs(Id)

    [NoiseContribution(d.name, :shot, shot_psd, d.a_idx, d.c_idx)]
end
```

## Test Suite Integration

### VA Regression Tests

```julia
@testset "Verilog-A MNA" begin

    @testset "varesistor" begin
        # Simple VA resistor should match linear resistor
        va_code = """
        module resistor(p, n);
            inout p, n;
            electrical p, n;
            parameter real r = 1k;
            analog I(p, n) <+ V(p, n) / r;
        endmodule
        """

        # Compile with MNA backend
        VA_MNA_MODE[] = true
        device = compile_va_device(va_code)
        VA_MNA_MODE[] = false

        # Test in voltage divider
        devices = [
            MNAVoltageSource(:V1, :in, :0, 10.0),
            device(:in, :mid, name=:R1, r=1e3),
            device(:mid, :0, name=:R2, r=1e3)
        ]

        circuit = build_circuit(devices)
        sol = solve_dc(circuit)

        @test sol.node_voltages[:mid] ≈ 5.0 atol=1e-10
    end

    @testset "vacap with ddt()" begin
        va_code = """
        module capacitor(p, n);
            inout p, n;
            electrical p, n;
            parameter real c = 1p;
            analog I(p, n) <+ ddt(c * V(p, n));
        endmodule
        """

        VA_MNA_MODE[] = true
        cap_device = compile_va_device(va_code)
        VA_MNA_MODE[] = false

        # RC circuit
        devices = [
            MNAVoltageSource(:V1, :in, :0, 1.0),
            MNAResistor(:R1, :in, :out, 1e3),
            cap_device(:out, :0, name=:C1, c=1e-6)
        ]

        circuit = build_circuit(devices)
        sol = solve_transient(circuit, (0.0, 5e-3))

        @test voltage(sol, circuit, :out, 1e-3) ≈ 1 - exp(-1) atol=1e-2
    end

    @testset "vadiode nonlinear" begin
        va_code = """
        module diode(a, c);
            inout a, c;
            electrical a, c;
            parameter real is = 1e-14;
            parameter real n = 1.0;
            analog begin
                I(a, c) <+ is * (exp(V(a, c) / (n * $vt)) - 1);
            end
        endmodule
        """

        VA_MNA_MODE[] = true
        diode_device = compile_va_device(va_code)
        VA_MNA_MODE[] = false

        # Diode with resistor
        devices = [
            MNAVoltageSource(:V1, :in, :0, 0.7),
            MNAResistor(:R1, :in, :a, 1e3),
            diode_device(:a, :0, name=:D1)
        ]

        circuit = build_circuit(devices)
        sol, iters = solve_dc_newton(circuit)

        # Diode should conduct, voltage around 0.6-0.7V
        @test sol.node_voltages[:a] > 0.5
        @test sol.node_voltages[:a] < 0.8
    end

end
```

### AC Analysis Tests

```julia
@testset "AC Analysis" begin

    @testset "RC Low-Pass Filter" begin
        # R = 1k, C = 1µF, fc = 1/(2πRC) ≈ 159 Hz
        devices = [
            MNAVSourceAC(:V1, :in, :0, 1.0, 0.0),  # 1V AC
            MNAResistor(:R1, :in, :out, 1e3),
            MNACapacitor(:C1, :out, :0, 1e-6)
        ]

        circuit = build_circuit(devices)
        ac_sol = solve_ac(circuit, 1.0, 10e3, 100)

        resp = ac_response(ac_sol, :out)

        # At cutoff frequency, gain should be -3dB (≈ 0.707)
        fc = 1 / (2π * 1e3 * 1e-6)
        fc_idx = argmin(abs.(resp.frequencies .- fc))
        @test resp.magnitude[fc_idx] ≈ 1/sqrt(2) atol=0.05

        # At high frequency, gain should be low
        @test resp.magnitude[end] < 0.1
    end

end
```

## Success Criteria

Phase 3 is complete when:

1. Verilog-A devices compile and run with MNA backend
2. Nonlinear devices (diode, basic MOSFET) work correctly
3. Newton-Raphson converges with GMIN/source stepping
4. AC analysis produces correct frequency response
5. Transient analysis with nonlinear devices works
6. Full test suite passes (≥95% of original tests)
7. BSIM model runs (may have some parameter limitations)

## Performance Considerations

- Sparse Jacobian essential for large circuits
- Device evaluation should be allocation-free in inner loop
- Consider SIMD for vector operations
- Profile and optimize hot paths after correctness established

## Future Enhancements

After Phase 3:
- Noise analysis completion
- Sensitivity analysis (∂output/∂parameter)
- Periodic steady-state (PSS) analysis
- Harmonic balance for RF circuits
- Parallel device evaluation
