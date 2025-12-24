#==============================================================================#
# MNA Phase 1: Basic Device Stamps
#
# This module provides stamp! methods for basic circuit elements.
# Each device type has a corresponding stamp! method that adds its
# contribution to the MNA matrices (G, C) and RHS vector (b).
#
# Devices implemented:
# - Resistor: stamps conductance G = 1/R
# - Capacitor: stamps capacitance C
# - Inductor: stamps inductance L (with current variable)
# - VoltageSource: stamps voltage constraint (with current variable)
# - CurrentSource: stamps source current into RHS
# - VCVS, VCCS: voltage-controlled sources
#==============================================================================#

export stamp!
export Resistor, Capacitor, Inductor
export VoltageSource, CurrentSource
export VCVS, VCCS, CCVS, CCCS

#==============================================================================#
# Device Types
#==============================================================================#

"""
    Resistor(r::Float64; name::Symbol=:R)

A linear resistor with resistance `r` ohms.

MNA stamps G matrix only (resistive, no dynamics).
"""
struct Resistor
    r::Float64
    name::Symbol
end
Resistor(r::Real; name::Symbol=:R) = Resistor(Float64(r), name)

"""
    Capacitor(c::Float64; name::Symbol=:C)

A linear capacitor with capacitance `c` farads.

MNA stamps C matrix only (reactive dynamics).
"""
struct Capacitor
    c::Float64
    name::Symbol
end
Capacitor(c::Real; name::Symbol=:C) = Capacitor(Float64(c), name)

"""
    Inductor(l::Float64; name::Symbol=:L)

A linear inductor with inductance `l` henries.

MNA requires a current variable (I_L is a state variable).
Stamps G matrix (KCL and voltage constraint) and C matrix (V = L*dI/dt).
"""
struct Inductor
    l::Float64
    name::Symbol
end
Inductor(l::Real; name::Symbol=:L) = Inductor(Float64(l), name)

"""
    VoltageSource(v::Float64; name::Symbol=:V)

An independent DC voltage source with voltage `v` volts.

MNA requires a current variable (current through source is unknown).
Stamps G matrix (KCL and voltage constraint) and b vector (source value).
"""
struct VoltageSource
    v::Float64
    name::Symbol
end
VoltageSource(v::Real; name::Symbol=:V) = VoltageSource(Float64(v), name)

"""
    CurrentSource(i::Float64; name::Symbol=:I)

An independent DC current source with current `i` amperes.

Stamps b vector only (source current into KCL equations).
Positive current flows from n to p (into the positive terminal).
"""
struct CurrentSource
    i::Float64
    name::Symbol
end
CurrentSource(i::Real; name::Symbol=:I) = CurrentSource(Float64(i), name)

#==============================================================================#
# Time-Dependent Sources (Mode-Aware)
#==============================================================================#

"""
    TimeDependentVoltageSource{F}

A voltage source with time-dependent value, respecting simulation mode.

In `:dcop` mode, returns `dc_value` (steady state).
In `:tran` mode, calls `value_fn(t)` for time-dependent behavior.

# Example
```julia
# Pulse source: 0V at t<1ms, 5V at t>=1ms
pulse = TimeDependentVoltageSource(
    t -> t < 1e-3 ? 0.0 : 5.0,
    dc_value = 0.0,  # DC operating point value
    name = :Vpulse
)
```
"""
struct TimeDependentVoltageSource{F}
    value_fn::F      # Function t -> voltage
    dc_value::Float64  # Value for DC analysis (mode = :dcop)
    name::Symbol
end

function TimeDependentVoltageSource(value_fn::F; dc_value::Real=0.0, name::Symbol=:V) where {F}
    TimeDependentVoltageSource{F}(value_fn, Float64(dc_value), name)
end

export TimeDependentVoltageSource

"""
    get_source_value(src::TimeDependentVoltageSource, t::Real, mode::Symbol) -> Float64

Get the source value at time `t` for the given simulation mode.
"""
function get_source_value(src::TimeDependentVoltageSource, t::Real, mode::Symbol)
    if mode == :dcop
        return src.dc_value
    else
        return src.value_fn(t)
    end
end

export get_source_value

"""
    PWLVoltageSource

Piecewise-linear voltage source defined by time-value pairs.

# Example
```julia
# Ramp from 0V to 5V over 1ms
pwl = PWLVoltageSource([0.0, 1e-3], [0.0, 5.0]; name=:Vramp)
```
"""
struct PWLVoltageSource
    times::Vector{Float64}
    values::Vector{Float64}
    name::Symbol

    function PWLVoltageSource(times::AbstractVector, values::AbstractVector; name::Symbol=:V)
        @assert length(times) == length(values) "times and values must have same length"
        @assert issorted(times) "times must be sorted"
        new(Float64.(times), Float64.(values), name)
    end
end

export PWLVoltageSource

"""
    pwl_value(src::PWLVoltageSource, t::Real) -> Float64

Evaluate PWL source at time t using linear interpolation.
"""
function pwl_value(src::PWLVoltageSource, t::Real)
    ts, vs = src.times, src.values
    n = length(ts)

    # Before first point
    if t <= ts[1]
        return vs[1]
    end

    # After last point
    if t >= ts[end]
        return vs[end]
    end

    # Find interval and interpolate
    for i in 1:(n-1)
        if ts[i] <= t <= ts[i+1]
            # Linear interpolation
            dt = ts[i+1] - ts[i]
            dv = vs[i+1] - vs[i]
            return vs[i] + dv * (t - ts[i]) / dt
        end
    end

    return vs[end]
end

function get_source_value(src::PWLVoltageSource, t::Real, mode::Symbol)
    if mode == :dcop
        # DC: use value at t=0
        return pwl_value(src, 0.0)
    else
        return pwl_value(src, t)
    end
end

export pwl_value

"""
    PWLCurrentSource

Piecewise-linear current source defined by time-value pairs.

# Example
```julia
# Ramp from 0A to 1mA over 1ms
pwl = PWLCurrentSource([0.0, 1e-3], [0.0, 1e-3]; name=:Iramp)
```
"""
struct PWLCurrentSource
    times::Vector{Float64}
    values::Vector{Float64}
    name::Symbol

    function PWLCurrentSource(times::AbstractVector, values::AbstractVector; name::Symbol=:I)
        @assert length(times) == length(values) "times and values must have same length"
        @assert issorted(times) "times must be sorted"
        new(Float64.(times), Float64.(values), name)
    end
end

export PWLCurrentSource

function pwl_value(src::PWLCurrentSource, t::Real)
    ts, vs = src.times, src.values
    n = length(ts)

    if t <= ts[1]
        return vs[1]
    end
    if t >= ts[end]
        return vs[end]
    end

    for i in 1:(n-1)
        if ts[i] <= t <= ts[i+1]
            dt = ts[i+1] - ts[i]
            dv = vs[i+1] - vs[i]
            return vs[i] + dv * (t - ts[i]) / dt
        end
    end
    return vs[end]
end

function get_source_value(src::PWLCurrentSource, t::Real, mode::Symbol)
    if mode == :dcop
        return pwl_value(src, 0.0)
    else
        return pwl_value(src, t)
    end
end

"""
    SinVoltageSource

Sinusoidal voltage source with SPICE-compatible parameters.

V(t) = vo + va * exp(-(t-td)*theta) * sin(2π*freq*(t-td) + phase)

For t < td: V(t) = vo + va * sin(phase)

# Fields
- `vo::Float64`: DC offset voltage
- `va::Float64`: Amplitude
- `freq::Float64`: Frequency in Hz
- `td::Float64`: Delay time (default: 0)
- `theta::Float64`: Damping factor (default: 0)
- `phase::Float64`: Phase in degrees (default: 0)
- `name::Symbol`: Source name

# Example
```julia
# 1kHz sine wave, 1V amplitude, 0.5V offset
sin_src = SinVoltageSource(0.5, 1.0, 1000.0; name=:Vsin)
```
"""
struct SinVoltageSource
    vo::Float64      # DC offset
    va::Float64      # Amplitude
    freq::Float64    # Frequency (Hz)
    td::Float64      # Delay
    theta::Float64   # Damping factor
    phase::Float64   # Phase (degrees)
    name::Symbol
end

function SinVoltageSource(vo::Real, va::Real, freq::Real;
                          td::Real=0.0, theta::Real=0.0, phase::Real=0.0,
                          name::Symbol=:V)
    SinVoltageSource(Float64(vo), Float64(va), Float64(freq),
                     Float64(td), Float64(theta), Float64(phase), name)
end

export SinVoltageSource

"""
    sin_value(src::SinVoltageSource, t::Real) -> Float64

Evaluate sinusoidal source at time t.
"""
function sin_value(src::SinVoltageSource, t::Real)
    if t < src.td
        # Before delay: DC offset + initial phase
        return src.vo + src.va * sind(src.phase)
    else
        # Damped sinusoid
        t_eff = t - src.td
        damping = src.theta == 0.0 ? 1.0 : exp(-t_eff * src.theta)
        return src.vo + src.va * damping * sind(360.0 * src.freq * t_eff + src.phase)
    end
end

function get_source_value(src::SinVoltageSource, t::Real, mode::Symbol)
    if mode == :dcop
        # DC: use value at t=0 (before delay)
        return src.vo + src.va * sind(src.phase)
    else
        return sin_value(src, t)
    end
end

export sin_value

"""
    SinCurrentSource

Sinusoidal current source with SPICE-compatible parameters.

I(t) = io + ia * exp(-(t-td)*theta) * sin(2π*freq*(t-td) + phase)

# Fields
Same as SinVoltageSource but for current.
"""
struct SinCurrentSource
    io::Float64      # DC offset
    ia::Float64      # Amplitude
    freq::Float64    # Frequency (Hz)
    td::Float64      # Delay
    theta::Float64   # Damping factor
    phase::Float64   # Phase (degrees)
    name::Symbol
end

function SinCurrentSource(io::Real, ia::Real, freq::Real;
                          td::Real=0.0, theta::Real=0.0, phase::Real=0.0,
                          name::Symbol=:I)
    SinCurrentSource(Float64(io), Float64(ia), Float64(freq),
                     Float64(td), Float64(theta), Float64(phase), name)
end

export SinCurrentSource

function sin_value(src::SinCurrentSource, t::Real)
    if t < src.td
        return src.io + src.ia * sind(src.phase)
    else
        t_eff = t - src.td
        damping = src.theta == 0.0 ? 1.0 : exp(-t_eff * src.theta)
        return src.io + src.ia * damping * sind(360.0 * src.freq * t_eff + src.phase)
    end
end

function get_source_value(src::SinCurrentSource, t::Real, mode::Symbol)
    if mode == :dcop
        return src.io + src.ia * sind(src.phase)
    else
        return sin_value(src, t)
    end
end

"""
    VCVS(gain::Float64; name::Symbol=:E)

Voltage-Controlled Voltage Source.
V(out+, out-) = gain * V(in+, in-)

Requires a current variable for the output branch.
"""
struct VCVS
    gain::Float64
    name::Symbol
end
VCVS(gain::Real; name::Symbol=:E) = VCVS(Float64(gain), name)

"""
    VCCS(gm::Float64; name::Symbol=:G)

Voltage-Controlled Current Source (transconductance amplifier).
I(out+, out-) = gm * V(in+, in-)

Does not require a current variable.
"""
struct VCCS
    gm::Float64
    name::Symbol
end
VCCS(gm::Real; name::Symbol=:G) = VCCS(Float64(gm), name)

"""
    CCVS(rm::Float64; name::Symbol=:H)

Current-Controlled Voltage Source (transresistance amplifier).
V(out+, out-) = rm * I(in+, in-)

Requires current variables for both input and output branches.
"""
struct CCVS
    rm::Float64
    name::Symbol
end
CCVS(rm::Real; name::Symbol=:H) = CCVS(Float64(rm), name)

"""
    CCCS(gain::Float64; name::Symbol=:F)

Current-Controlled Current Source.
I(out+, out-) = gain * I(in+, in-)

Requires a current variable for the input (sensing) branch.
"""
struct CCCS
    gain::Float64
    name::Symbol
end
CCCS(gain::Real; name::Symbol=:F) = CCCS(Float64(gain), name)

#==============================================================================#
# Stamp Methods
#==============================================================================#

"""
    stamp!(device, ctx::MNAContext, ports...)

Stamp a device into the MNA context.

The number and meaning of ports depends on the device type:
- 2-terminal devices (R, C, L, V, I): stamp!(dev, ctx, p, n)
- 4-terminal sources (VCVS, VCCS): stamp!(dev, ctx, out_p, out_n, in_p, in_n)
"""
function stamp! end

#------------------------------------------------------------------------------#
# Resistor: G matrix only
#------------------------------------------------------------------------------#

"""
    stamp!(R::Resistor, ctx::MNAContext, p::Int, n::Int)

Stamp a resistor between nodes p and n.

The resistor contributes I = (Vp - Vn)/R to KCL at each node.
Using "current leaving node is positive" convention:
- KCL at p: +G*Vp - G*Vn (current leaves p)
- KCL at n: -G*Vp + G*Vn (current enters n)

Matrix pattern (for G = 1/R):
```
     Vp  Vn
Vp [ +G  -G ]
Vn [ -G  +G ]
```
"""
function stamp!(R::Resistor, ctx::MNAContext, p::Int, n::Int)
    G = 1.0 / R.r
    stamp_conductance!(ctx, p, n, G)
    return nothing
end

#------------------------------------------------------------------------------#
# Capacitor: C matrix only
#------------------------------------------------------------------------------#

"""
    stamp!(C::Capacitor, ctx::MNAContext, p::Int, n::Int)

Stamp a capacitor between nodes p and n.

The capacitor contributes I = C * d(Vp - Vn)/dt to KCL.
This stamps into the C matrix with the same pattern as a resistor.

Matrix pattern:
```
     Vp  Vn
Vp [ +C  -C ]
Vn [ -C  +C ]
```
"""
function stamp!(C::Capacitor, ctx::MNAContext, p::Int, n::Int)
    stamp_capacitance!(ctx, p, n, C.c)
    return nothing
end

#------------------------------------------------------------------------------#
# Inductor: G and C matrices, needs current variable
#------------------------------------------------------------------------------#

"""
    stamp!(L::Inductor, ctx::MNAContext, p::Int, n::Int) -> Int

Stamp an inductor between nodes p and n.
Returns the index of the inductor current variable.

The inductor uses a current variable I_L because current is a state variable.
Equations:
1. KCL at p: current I_L leaves p
2. KCL at n: current I_L enters n
3. Voltage equation: V_p - V_n = L * dI_L/dt

Matrix pattern:
```
        Vp  Vn  I_L
    Vp [ .   .  +1  ]     KCL: I_L leaves p
    Vn [ .   .  -1  ]     KCL: I_L enters n
G = I_L[+1  -1   .  ]     Voltage: Vp - Vn

        Vp  Vn  I_L
    Vp [ .   .   .  ]
    Vn [ .   .   .  ]
C = I_L[ .   .  -L  ]     Voltage: ... = L * dI/dt
```

Note: The -L in C comes from rewriting V = L*dI/dt as:
      G*x + C*dx/dt = b
      Vp - Vn - L*dI/dt = 0
"""
function stamp!(L::Inductor, ctx::MNAContext, p::Int, n::Int)
    # Allocate current variable
    I_idx = alloc_current!(ctx, Symbol(:I_, L.name))

    # KCL: current I_L flows from p to n
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # Voltage equation: Vp - Vn - L*dI/dt = 0
    # G part: Vp - Vn
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)

    # C part: -L * dI/dt
    stamp_C!(ctx, I_idx, I_idx, -L.l)

    return I_idx
end

#------------------------------------------------------------------------------#
# Voltage Source: G matrix and b vector, needs current variable
#------------------------------------------------------------------------------#

"""
    stamp!(V::VoltageSource, ctx::MNAContext, p::Int, n::Int) -> Int

Stamp a voltage source between nodes p and n (V_p - V_n = V.v).
Returns the index of the source current variable.

The voltage source needs a current variable because current is unknown.
Equations:
1. KCL at p: current I_V leaves p (enters source at + terminal)
2. KCL at n: current I_V enters n
3. Voltage equation: V_p - V_n = V_dc

Matrix pattern:
```
        Vp  Vn  I_V
    Vp [ .   .  +1  ]     KCL: I_V leaves p
    Vn [ .   .  -1  ]     KCL: I_V enters n
G = I_V[+1  -1   .  ]     Voltage constraint: Vp - Vn

    I_V
b = V_dc                  Source voltage value
```
"""
function stamp!(V::VoltageSource, ctx::MNAContext, p::Int, n::Int)
    # Allocate current variable
    I_idx = alloc_current!(ctx, Symbol(:I_, V.name))

    # KCL: current I_V flows from p through source to n
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # Voltage equation: Vp - Vn = V_dc
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)
    stamp_b!(ctx, I_idx, V.v)

    return I_idx
end

#------------------------------------------------------------------------------#
# Current Source: b vector only
#------------------------------------------------------------------------------#

"""
    stamp!(I::CurrentSource, ctx::MNAContext, p::Int, n::Int)

Stamp a current source between nodes p and n.
Positive current flows from n to p (into the positive terminal p).

KCL contributions:
- At p: current I.i enters (positive term)
- At n: current I.i leaves (negative term)

This follows the passive sign convention where current enters
the positive terminal.

RHS pattern:
```
    p
b = +I   (current enters p)
    n
    -I   (current leaves n)
```
"""
function stamp!(I::CurrentSource, ctx::MNAContext, p::Int, n::Int)
    # Current flows from n to p (into positive terminal)
    # KCL at p: current I enters (add to RHS)
    # KCL at n: current I leaves (subtract from RHS)
    stamp_b!(ctx, p,  I.i)
    stamp_b!(ctx, n, -I.i)
    return nothing
end

#------------------------------------------------------------------------------#
# VCVS: Voltage-Controlled Voltage Source
#------------------------------------------------------------------------------#

"""
    stamp!(E::VCVS, ctx::MNAContext, out_p::Int, out_n::Int, in_p::Int, in_n::Int) -> Int

Stamp a voltage-controlled voltage source.
V(out_p, out_n) = E.gain * V(in_p, in_n)

Returns the index of the output current variable.

Matrix pattern:
```
            out_p out_n in_p in_n I_E
out_p      [  .     .    .    .   +1 ]  KCL at out_p
out_n      [  .     .    .    .   -1 ]  KCL at out_n
G = I_E    [ +1    -1   -A   +A    . ]  Voltage: Vout - A*Vin = 0
```
where A = E.gain
"""
function stamp!(E::VCVS, ctx::MNAContext, out_p::Int, out_n::Int, in_p::Int, in_n::Int)
    # Allocate current variable for output branch
    I_idx = alloc_current!(ctx, Symbol(:I_, E.name))

    # KCL at output nodes
    stamp_G!(ctx, out_p, I_idx,  1.0)
    stamp_G!(ctx, out_n, I_idx, -1.0)

    # Voltage equation: V(out_p) - V(out_n) - gain * (V(in_p) - V(in_n)) = 0
    stamp_G!(ctx, I_idx, out_p,  1.0)
    stamp_G!(ctx, I_idx, out_n, -1.0)
    stamp_G!(ctx, I_idx, in_p,  -E.gain)
    stamp_G!(ctx, I_idx, in_n,   E.gain)

    return I_idx
end

#------------------------------------------------------------------------------#
# VCCS: Voltage-Controlled Current Source
#------------------------------------------------------------------------------#

"""
    stamp!(G_dev::VCCS, ctx::MNAContext, out_p::Int, out_n::Int, in_p::Int, in_n::Int)

Stamp a voltage-controlled current source (transconductance).
I(out_p, out_n) = G_dev.gm * V(in_p, in_n)

Current flows from out_n to out_p (into out_p).
No current variable needed.

Matrix pattern (MNA convention: current leaving is positive):
```
            out_p out_n in_p in_n
out_p      [  .     .   -gm  +gm ]  Current leaving out_p = -gm*Vin
G = out_n  [  .     .   +gm  -gm ]  Current leaving out_n = +gm*Vin
```
"""
function stamp!(G_dev::VCCS, ctx::MNAContext, out_p::Int, out_n::Int, in_p::Int, in_n::Int)
    gm = G_dev.gm
    # I = gm * V(in_p, in_n) flows INTO out_p (from out_n)
    # MNA uses "current leaving is positive", so:
    # - Current leaving out_p = -gm * V(in_p, in_n)
    # - Current leaving out_n = +gm * V(in_p, in_n)
    stamp_G!(ctx, out_p, in_p, -gm)
    stamp_G!(ctx, out_p, in_n,  gm)
    stamp_G!(ctx, out_n, in_p,  gm)
    stamp_G!(ctx, out_n, in_n, -gm)
    return nothing
end

#------------------------------------------------------------------------------#
# CCVS: Current-Controlled Voltage Source
#------------------------------------------------------------------------------#

"""
    stamp!(H::CCVS, ctx::MNAContext, out_p::Int, out_n::Int, in_p::Int, in_n::Int) -> Tuple{Int,Int}

Stamp a current-controlled voltage source (transresistance).
V(out_p, out_n) = H.rm * I(in_p, in_n)

Returns (I_out_idx, I_in_idx) - indices of output and input current variables.

The input branch is a zero-volt voltage source (ammeter) to sense current.
"""
function stamp!(H::CCVS, ctx::MNAContext, out_p::Int, out_n::Int, in_p::Int, in_n::Int)
    # Input current variable (sensing branch: V = 0)
    I_in_idx = alloc_current!(ctx, Symbol(:I_, H.name, :_in))

    # Output current variable
    I_out_idx = alloc_current!(ctx, Symbol(:I_, H.name, :_out))

    # Input sensing branch (zero-volt source)
    stamp_G!(ctx, in_p, I_in_idx,  1.0)
    stamp_G!(ctx, in_n, I_in_idx, -1.0)
    stamp_G!(ctx, I_in_idx, in_p,  1.0)
    stamp_G!(ctx, I_in_idx, in_n, -1.0)
    # b[I_in_idx] = 0 (implicit)

    # Output branch
    stamp_G!(ctx, out_p, I_out_idx,  1.0)
    stamp_G!(ctx, out_n, I_out_idx, -1.0)

    # Voltage equation: V(out) = rm * I(in)
    # V(out_p) - V(out_n) - rm * I_in = 0
    stamp_G!(ctx, I_out_idx, out_p,     1.0)
    stamp_G!(ctx, I_out_idx, out_n,    -1.0)
    stamp_G!(ctx, I_out_idx, I_in_idx, -H.rm)

    return (I_out_idx, I_in_idx)
end

#------------------------------------------------------------------------------#
# CCCS: Current-Controlled Current Source
#------------------------------------------------------------------------------#

"""
    stamp!(F::CCCS, ctx::MNAContext, out_p::Int, out_n::Int, in_p::Int, in_n::Int) -> Int

Stamp a current-controlled current source.
I(out_p, out_n) = F.gain * I(in_p, in_n)

Returns I_in_idx - the index of the input current variable.

The input branch is a zero-volt voltage source (ammeter) to sense current.
"""
function stamp!(F::CCCS, ctx::MNAContext, out_p::Int, out_n::Int, in_p::Int, in_n::Int)
    # Input current variable (sensing branch)
    I_in_idx = alloc_current!(ctx, Symbol(:I_, F.name, :_in))

    # Input sensing branch (zero-volt source)
    stamp_G!(ctx, in_p, I_in_idx,  1.0)
    stamp_G!(ctx, in_n, I_in_idx, -1.0)
    stamp_G!(ctx, I_in_idx, in_p,  1.0)
    stamp_G!(ctx, I_in_idx, in_n, -1.0)

    # Output current: gain * I_in flows into out_p (out of out_n)
    # Negative stamp means current entering the node
    stamp_G!(ctx, out_p, I_in_idx, -F.gain)
    stamp_G!(ctx, out_n, I_in_idx,  F.gain)

    return I_in_idx
end

#------------------------------------------------------------------------------#
# SPICE-style Current-Controlled Sources (reference existing V-source)
#------------------------------------------------------------------------------#

"""
    stamp!(H::CCVS, ctx::MNAContext, out_p::Int, out_n::Int, I_in_idx::Int) -> Int

Stamp a CCVS using an existing current variable (SPICE-style).
V(out_p, out_n) = H.rm * I_in

I_in_idx is the index of an existing current variable (e.g., from a voltage source).
This is used when the SPICE netlist references a voltage source name for current sensing.

Returns I_out_idx - the index of the output current variable.
"""
function stamp!(H::CCVS, ctx::MNAContext, out_p::Int, out_n::Int, I_in_idx::Int)
    # Output current variable
    I_out_idx = alloc_current!(ctx, Symbol(:I_, H.name))

    # Output branch
    stamp_G!(ctx, out_p, I_out_idx,  1.0)
    stamp_G!(ctx, out_n, I_out_idx, -1.0)

    # Voltage equation: V(out) = rm * I(in)
    # V(out_p) - V(out_n) - rm * I_in = 0
    stamp_G!(ctx, I_out_idx, out_p,     1.0)
    stamp_G!(ctx, I_out_idx, out_n,    -1.0)
    stamp_G!(ctx, I_out_idx, I_in_idx, -H.rm)

    return I_out_idx
end

"""
    stamp!(F::CCCS, ctx::MNAContext, out_p::Int, out_n::Int, I_in_idx::Int) -> Nothing

Stamp a CCCS using an existing current variable (SPICE-style).
I(out_p, out_n) = F.gain * I_in

I_in_idx is the index of an existing current variable (e.g., from a voltage source).
This is used when the SPICE netlist references a voltage source name for current sensing.
"""
function stamp!(F::CCCS, ctx::MNAContext, out_p::Int, out_n::Int, I_in_idx::Int)
    # Output current: gain * I_in flows into out_p (out of out_n)
    # Negative stamp means current entering the node
    stamp_G!(ctx, out_p, I_in_idx, -F.gain)
    stamp_G!(ctx, out_n, I_in_idx,  F.gain)

    return nothing
end

#------------------------------------------------------------------------------#
# Time-Dependent Voltage Sources
#------------------------------------------------------------------------------#

"""
    stamp!(V::TimeDependentVoltageSource, ctx::MNAContext, p::Int, n::Int; t::Real=0.0, mode::Symbol=:dcop)

Stamp a time-dependent voltage source.
"""
function stamp!(V::TimeDependentVoltageSource, ctx::MNAContext, p::Int, n::Int;
                t::Real=0.0, mode::Symbol=:dcop)
    I_idx = alloc_current!(ctx, Symbol(:I_, V.name))

    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)

    v = get_source_value(V, t, mode)
    stamp_b!(ctx, I_idx, v)

    return I_idx
end

"""
    stamp!(V::PWLVoltageSource, ctx::MNAContext, p::Int, n::Int; t::Real=0.0, mode::Symbol=:dcop)

Stamp a PWL voltage source evaluated at time t.
"""
function stamp!(V::PWLVoltageSource, ctx::MNAContext, p::Int, n::Int;
                t::Real=0.0, mode::Symbol=:dcop)
    I_idx = alloc_current!(ctx, Symbol(:I_, V.name))

    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)

    v = get_source_value(V, t, mode)
    stamp_b!(ctx, I_idx, v)

    return I_idx
end

"""
    stamp!(V::SinVoltageSource, ctx::MNAContext, p::Int, n::Int; t::Real=0.0, mode::Symbol=:dcop)

Stamp a sinusoidal voltage source evaluated at time t.
"""
function stamp!(V::SinVoltageSource, ctx::MNAContext, p::Int, n::Int;
                t::Real=0.0, mode::Symbol=:dcop)
    I_idx = alloc_current!(ctx, Symbol(:I_, V.name))

    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)

    v = get_source_value(V, t, mode)
    stamp_b!(ctx, I_idx, v)

    return I_idx
end

#------------------------------------------------------------------------------#
# Time-Dependent Current Sources
#------------------------------------------------------------------------------#

"""
    stamp!(I::PWLCurrentSource, ctx::MNAContext, p::Int, n::Int; t::Real=0.0, mode::Symbol=:dcop)

Stamp a PWL current source evaluated at time t.
"""
function stamp!(I::PWLCurrentSource, ctx::MNAContext, p::Int, n::Int;
                t::Real=0.0, mode::Symbol=:dcop)
    i = get_source_value(I, t, mode)
    stamp_b!(ctx, p,  i)
    stamp_b!(ctx, n, -i)
    return nothing
end

"""
    stamp!(I::SinCurrentSource, ctx::MNAContext, p::Int, n::Int; t::Real=0.0, mode::Symbol=:dcop)

Stamp a sinusoidal current source evaluated at time t.
"""
function stamp!(I::SinCurrentSource, ctx::MNAContext, p::Int, n::Int;
                t::Real=0.0, mode::Symbol=:dcop)
    i = get_source_value(I, t, mode)
    stamp_b!(ctx, p,  i)
    stamp_b!(ctx, n, -i)
    return nothing
end

#==============================================================================#
# Convenience: Stamp by node names
#==============================================================================#

"""
    stamp!(device, ctx::MNAContext, p::Symbol, n::Symbol, ...)

Stamp a device using node names (automatically allocated).
"""
function stamp!(device, ctx::MNAContext, p::Symbol, n::Symbol)
    stamp!(device, ctx, get_node!(ctx, p), get_node!(ctx, n))
end

function stamp!(device, ctx::MNAContext, out_p::Symbol, out_n::Symbol, in_p::Symbol, in_n::Symbol)
    stamp!(device, ctx,
           get_node!(ctx, out_p), get_node!(ctx, out_n),
           get_node!(ctx, in_p), get_node!(ctx, in_n))
end

# Convenience for time-dependent sources with symbols
function stamp!(device::Union{TimeDependentVoltageSource, PWLVoltageSource, SinVoltageSource,
                              PWLCurrentSource, SinCurrentSource},
                ctx::MNAContext, p::Symbol, n::Symbol; t::Real=0.0, mode::Symbol=:dcop)
    stamp!(device, ctx, get_node!(ctx, p), get_node!(ctx, n); t=t, mode=mode)
end

#==============================================================================#
# Behavioral Sources (B-sources)
#==============================================================================#

"""
    BehavioralVoltageSource{F}

A behavioral voltage source where V(p,n) is determined by evaluating an expression
that can reference other node voltages.

The `value_fn` is a function `(get_voltage) -> voltage` where `get_voltage(node)` returns
the voltage at a node by its symbol name.

# Example
```julia
# B-source: v=V(1)*2 makes V(p,n) = 2 * V(node_1)
bsrc = BehavioralVoltageSource(
    get_v -> 2.0 * get_v(Symbol("1")),
    name = :B5
)
```
"""
struct BehavioralVoltageSource{F}
    value_fn::F      # Function (get_voltage) -> value
    name::Symbol
end

function BehavioralVoltageSource(value_fn::F; name::Symbol=:B) where {F}
    BehavioralVoltageSource{F}(value_fn, name)
end

export BehavioralVoltageSource

"""
    BehavioralCurrentSource{F}

A behavioral current source where I(p,n) is determined by evaluating an expression
that can reference other node voltages.

The `value_fn` is a function `(get_voltage) -> current` where `get_voltage(node)` returns
the voltage at a node by its symbol name.

# Example
```julia
# B-source: i=V(1)*0.001 makes I(p,n) = V(node_1) / 1000
bsrc = BehavioralCurrentSource(
    get_v -> get_v(Symbol("1")) * 0.001,
    name = :B1
)
```
"""
struct BehavioralCurrentSource{F}
    value_fn::F      # Function (get_voltage) -> value
    name::Symbol
end

function BehavioralCurrentSource(value_fn::F; name::Symbol=:B) where {F}
    BehavioralCurrentSource{F}(value_fn, name)
end

export BehavioralCurrentSource

#------------------------------------------------------------------------------#
# Behavioral Voltage Source Stamping
#------------------------------------------------------------------------------#

"""
    stamp!(B::BehavioralVoltageSource, ctx::MNAContext, p::Int, n::Int;
           get_voltage=nothing) -> Int

Stamp a behavioral voltage source. The source voltage is computed by calling
`B.value_fn(get_voltage)` where `get_voltage(node_name)` returns the voltage at that node.

For DC analysis, this requires an iterative Newton solver since the source value
depends on other node voltages.

In the linear approximation (first Newton iteration), we evaluate the expression
at the current solution estimate and stamp as a fixed voltage source.

Returns the index of the source current variable.
"""
function stamp!(B::BehavioralVoltageSource, ctx::MNAContext, p::Int, n::Int;
                get_voltage=nothing)
    # Allocate current variable
    I_idx = alloc_current!(ctx, Symbol(:I_, B.name))

    # KCL: current I flows from p through source to n
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # Voltage equation: Vp - Vn = V_behavioral
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)

    # Compute the behavioral voltage value
    v = if get_voltage !== nothing
        B.value_fn(get_voltage)
    else
        # If no voltage accessor provided, use 0 (will be updated in Newton iteration)
        0.0
    end
    stamp_b!(ctx, I_idx, v)

    return I_idx
end

#------------------------------------------------------------------------------#
# Behavioral Current Source Stamping
#------------------------------------------------------------------------------#

"""
    stamp!(B::BehavioralCurrentSource, ctx::MNAContext, p::Int, n::Int;
           get_voltage=nothing)

Stamp a behavioral current source. The source current is computed by calling
`B.value_fn(get_voltage)` where `get_voltage(node_name)` returns the voltage at that node.

Positive current flows from n to p (into the positive terminal p).
"""
function stamp!(B::BehavioralCurrentSource, ctx::MNAContext, p::Int, n::Int;
                get_voltage=nothing)
    # Compute the behavioral current value
    i = if get_voltage !== nothing
        B.value_fn(get_voltage)
    else
        # If no voltage accessor provided, use 0 (will be updated in Newton iteration)
        0.0
    end

    # Current flows from n to p (into positive terminal)
    stamp_b!(ctx, p,  i)
    stamp_b!(ctx, n, -i)
    return nothing
end

# Convenience for behavioral sources with symbol nodes
function stamp!(B::Union{BehavioralVoltageSource, BehavioralCurrentSource},
                ctx::MNAContext, p::Symbol, n::Symbol; get_voltage=nothing)
    stamp!(B, ctx, get_node!(ctx, p), get_node!(ctx, n); get_voltage=get_voltage)
end

#==============================================================================#
# Phase 6: Nonlinear Devices
#==============================================================================#

"""
    Diode(; Is=1e-14, Vt=0.026, n=1.0, name=:D)

Ideal diode with exponential I-V characteristic: I = Is * (exp(V/(n*Vt)) - 1)

This is a nonlinear device that requires Newton iteration for DC analysis.
The stamp! method takes `x` parameter for the current operating point.

# Parameters
- `Is`: Saturation current (default: 1e-14 A)
- `Vt`: Thermal voltage (default: 0.026 V ≈ kT/q at 300K)
- `n`: Ideality factor (default: 1.0)
- `name`: Device name

# Example
```julia
function build_circuit(params, spec; x=Float64[])
    ctx = MNAContext()
    vin = get_node!(ctx, :vin)
    out = get_node!(ctx, :out)

    stamp!(VoltageSource(5.0), ctx, vin, 0)
    stamp!(Resistor(1000.0), ctx, vin, out)
    stamp!(Diode(Is=1e-14), ctx, out, 0; x=x)

    return ctx
end

sol = solve_dc(build_circuit, (;), MNASpec(mode=:dcop))
```
"""
struct Diode
    Is::Float64    # Saturation current
    Vt::Float64    # Thermal voltage
    n::Float64     # Ideality factor
    name::Symbol
end

function Diode(; Is::Real=1e-14, Vt::Real=0.026, n::Real=1.0, name::Symbol=:D)
    Diode(Float64(Is), Float64(Vt), Float64(n), name)
end

export Diode

"""
    stamp!(D::Diode, ctx::MNAContext, p::Int, n::Int; x::AbstractVector=Float64[])

Stamp a nonlinear diode into MNA matrices.

Uses Newton companion model: linearize I(V) at operating point V0.
I ≈ I(V0) + G(V0) * (V - V0) = G*V + (I0 - G*V0)

where G = dI/dV = Is/(n*Vt) * exp(V0/(n*Vt)) at operating point.

Stamps:
- G matrix: conductance G(V0) between p and n
- b vector: Newton companion current Ieq = I0 - G*V0

The x parameter provides the current solution for V0.
"""
function stamp!(D::Diode, ctx::MNAContext, p::Int, n::Int;
                x::AbstractVector=Float64[])
    # Get operating point voltage
    Vp = p == 0 ? 0.0 : (isempty(x) ? 0.0 : x[p])
    Vn = n == 0 ? 0.0 : (isempty(x) ? 0.0 : x[n])
    V0 = Vp - Vn

    # Diode equation: I = Is * (exp(V/(n*Vt)) - 1)
    Is, Vt, n_factor = D.Is, D.Vt, D.n
    nVt = n_factor * Vt

    # Current at operating point
    expterm = exp(V0 / nVt)
    I0 = Is * (expterm - 1.0)

    # Conductance (dI/dV) at operating point
    G = Is / nVt * expterm

    # Newton companion model: I = G*V + Ieq
    # where Ieq = I0 - G*V0
    Ieq = I0 - G * V0

    # Stamp conductance (same pattern as resistor)
    stamp_conductance!(ctx, p, n, G)

    # Stamp companion current source
    # Current I flows from p to n (out of p, into n in MNA convention)
    # So: -Ieq enters p, +Ieq enters n
    stamp_b!(ctx, p, -Ieq)
    stamp_b!(ctx, n,  Ieq)

    return nothing
end

"""
    DiodeWithCap(; Is=1e-14, Vt=0.026, n=1.0, Cj0=1e-12, Vj=0.7, m=0.5, name=:D)

Diode with nonlinear junction capacitance.

Combines the exponential I-V characteristic with voltage-dependent
depletion capacitance model: Cj(V) = Cj0 / (1 - V/Vj)^m

For forward bias (V > Vj), the capacitance is clamped to avoid singularity.

# Parameters
- `Is`: Saturation current (default: 1e-14 A)
- `Vt`: Thermal voltage (default: 0.026 V)
- `n`: Ideality factor (default: 1.0)
- `Cj0`: Zero-bias junction capacitance (default: 1e-12 F)
- `Vj`: Junction potential (default: 0.7 V)
- `m`: Grading coefficient (default: 0.5)
- `name`: Device name

# Charge Model
The junction charge is:
    q(V) = Cj0 * Vj * (1 - (1 - V/Vj)^(1-m)) / (1-m)  for V < Vj

For transient analysis, the capacitor current is I = dq/dt.
"""
struct DiodeWithCap
    Is::Float64    # Saturation current
    Vt::Float64    # Thermal voltage
    n::Float64     # Ideality factor
    Cj0::Float64   # Zero-bias junction capacitance
    Vj::Float64    # Junction potential
    m::Float64     # Grading coefficient
    name::Symbol
end

function DiodeWithCap(; Is::Real=1e-14, Vt::Real=0.026, n::Real=1.0,
                       Cj0::Real=1e-12, Vj::Real=0.7, m::Real=0.5, name::Symbol=:D)
    DiodeWithCap(Float64(Is), Float64(Vt), Float64(n),
                 Float64(Cj0), Float64(Vj), Float64(m), name)
end

export DiodeWithCap

"""
    diode_junction_cap(V, Cj0, Vj, m) -> Float64

Compute the junction capacitance at voltage V.

Cj(V) = Cj0 / (1 - V/Vj)^m

For V approaching Vj, we use a linear extrapolation to avoid singularity.
"""
function diode_junction_cap(V, Cj0, Vj, m)
    # Clamp to avoid singularity near V = Vj
    Vmax = 0.9 * Vj
    if V < Vmax
        return Cj0 / (1 - V/Vj)^m
    else
        # Linear extrapolation from Vmax
        C_at_max = Cj0 / (1 - Vmax/Vj)^m
        dC_dV = Cj0 * m / Vj / (1 - Vmax/Vj)^(m+1)
        return C_at_max + dC_dV * (V - Vmax)
    end
end

"""
    diode_junction_charge(V, Cj0, Vj, m) -> Float64

Compute the junction charge at voltage V.

q(V) = ∫ Cj(V) dV = Cj0 * Vj / (1-m) * (1 - (1 - V/Vj)^(1-m))

For V approaching Vj, we use continuation.
"""
function diode_junction_charge(V, Cj0, Vj, m)
    # Clamp to avoid singularity
    Vmax = 0.9 * Vj
    if V < Vmax
        if abs(m - 1.0) < 1e-10
            # Special case: m = 1 -> q = Cj0 * Vj * log(1 - V/Vj)
            return -Cj0 * Vj * log(1 - V/Vj)
        else
            return Cj0 * Vj / (1 - m) * (1 - (1 - V/Vj)^(1-m))
        end
    else
        # Continuation: q(V) = q(Vmax) + Cj(Vmax) * (V - Vmax)
        q_at_max = Cj0 * Vj / (1 - m) * (1 - (1 - Vmax/Vj)^(1-m))
        C_at_max = Cj0 / (1 - Vmax/Vj)^m
        return q_at_max + C_at_max * (V - Vmax)
    end
end

export diode_junction_cap, diode_junction_charge

"""
    stamp!(D::DiodeWithCap, ctx::MNAContext, p::Int, n::Int; x::AbstractVector=Float64[])

Stamp a diode with junction capacitance.

This stamps both:
1. Resistive part: Nonlinear I-V (same as Diode)
2. Reactive part: Voltage-dependent capacitance Cj(V)

For transient analysis, the capacitor current is I = dq/dt = C(V) * dV/dt.
"""
function stamp!(D::DiodeWithCap, ctx::MNAContext, p::Int, n::Int;
                x::AbstractVector=Float64[])
    # Get operating point voltage
    Vp = p == 0 ? 0.0 : (isempty(x) ? 0.0 : x[p])
    Vn = n == 0 ? 0.0 : (isempty(x) ? 0.0 : x[n])
    V0 = Vp - Vn

    # === Resistive Part (DC current) ===
    Is, Vt, n_factor = D.Is, D.Vt, D.n
    nVt = n_factor * Vt

    expterm = exp(V0 / nVt)
    I0 = Is * (expterm - 1.0)
    G = Is / nVt * expterm

    Ieq = I0 - G * V0

    # Stamp conductance
    stamp_conductance!(ctx, p, n, G)

    # Stamp companion current
    stamp_b!(ctx, p, -Ieq)
    stamp_b!(ctx, n,  Ieq)

    # === Reactive Part (Junction Capacitance) ===
    Cj0, Vj, m = D.Cj0, D.Vj, D.m

    # Capacitance at operating point
    C = diode_junction_cap(V0, Cj0, Vj, m)

    # Stamp capacitance
    stamp_capacitance!(ctx, p, n, C)

    return nothing
end

#==============================================================================#
# Phase 6: Simple MOSFET Model
#==============================================================================#

"""
    SimpleMOSFET(; Vth=0.5, K=1e-3, lambda=0.0, Cgd=1e-15, Cgs=1e-15, name=:M)

Simple long-channel MOSFET model for testing.

Uses square-law model in saturation:
    Ids = K/2 * (Vgs - Vth)^2 * (1 + lambda*Vds)  for Vgs > Vth, Vds > Vgs - Vth

Linear region:
    Ids = K * ((Vgs - Vth)*Vds - Vds^2/2)  for Vgs > Vth, Vds < Vgs - Vth

# Parameters
- `Vth`: Threshold voltage (default: 0.5 V)
- `K`: Transconductance parameter (default: 1e-3 A/V²)
- `lambda`: Channel length modulation (default: 0)
- `Cgd`: Gate-drain capacitance (default: 1e-15 F)
- `Cgs`: Gate-source capacitance (default: 1e-15 F)
- `name`: Device name

This is NOT a BSIM4-level model. It's a simple model for testing the
multi-port stamping infrastructure.
"""
struct SimpleMOSFET
    Vth::Float64     # Threshold voltage
    K::Float64       # Transconductance parameter
    lambda::Float64  # Channel length modulation
    Cgd::Float64     # Gate-drain capacitance
    Cgs::Float64     # Gate-source capacitance
    name::Symbol
end

function SimpleMOSFET(; Vth::Real=0.5, K::Real=1e-3, lambda::Real=0.0,
                       Cgd::Real=1e-15, Cgs::Real=1e-15, name::Symbol=:M)
    SimpleMOSFET(Float64(Vth), Float64(K), Float64(lambda),
                 Float64(Cgd), Float64(Cgs), name)
end

export SimpleMOSFET

"""
    stamp!(M::SimpleMOSFET, ctx::MNAContext, d::Int, g::Int, s::Int;
           x::AbstractVector=Float64[])

Stamp a simple MOSFET (3-terminal: drain, gate, source).

Uses ForwardDiff to compute transconductances gm, gds from the square-law model.
"""
function stamp!(M::SimpleMOSFET, ctx::MNAContext, d::Int, g::Int, s::Int;
                x::AbstractVector=Float64[])
    # Get operating point voltages
    Vd = d == 0 ? 0.0 : (isempty(x) ? 0.0 : x[d])
    Vg = g == 0 ? 0.0 : (isempty(x) ? 0.0 : x[g])
    Vs = s == 0 ? 0.0 : (isempty(x) ? 0.0 : x[s])

    Vgs = Vg - Vs
    Vds = Vd - Vs

    Vth, K, lambda = M.Vth, M.K, M.lambda

    # Compute drain current and derivatives
    if Vgs <= Vth
        # Cutoff
        Ids = 0.0
        gm = 0.0    # dIds/dVgs
        gds = 0.0   # dIds/dVds
    elseif Vds <= Vgs - Vth
        # Linear region
        Ids = K * ((Vgs - Vth) * Vds - Vds^2 / 2)
        gm = K * Vds
        gds = K * (Vgs - Vth - Vds)
    else
        # Saturation
        Ids = K / 2 * (Vgs - Vth)^2 * (1 + lambda * Vds)
        gm = K * (Vgs - Vth) * (1 + lambda * Vds)
        gds = K / 2 * (Vgs - Vth)^2 * lambda
    end

    # Newton companion: Ids = gm*(Vg-Vs) + gds*(Vd-Vs) + Ieq
    # where Ieq = Ids - gm*Vgs - gds*Vds
    Ieq = Ids - gm * Vgs - gds * Vds

    # Stamp drain current (flows from drain to source)
    # Current leaves drain (d), enters source (s)
    # dIds/dVd = gds, dIds/dVg = gm, dIds/dVs = -(gm + gds)

    # G matrix contributions
    stamp_G!(ctx, d, d,  gds)
    stamp_G!(ctx, d, g,  gm)
    stamp_G!(ctx, d, s, -(gds + gm))

    stamp_G!(ctx, s, d, -gds)
    stamp_G!(ctx, s, g, -gm)
    stamp_G!(ctx, s, s,  gds + gm)

    # Companion current
    stamp_b!(ctx, d, -Ieq)
    stamp_b!(ctx, s,  Ieq)

    # === Capacitances ===
    # Simple linear caps (not voltage-dependent for this simple model)
    # Cgs between g and s
    stamp_capacitance!(ctx, g, s, M.Cgs)
    # Cgd between g and d
    stamp_capacitance!(ctx, g, d, M.Cgd)

    return nothing
end

# 4-terminal version with body (bulk)
function stamp!(M::SimpleMOSFET, ctx::MNAContext, d::Int, g::Int, s::Int, b::Int;
                x::AbstractVector=Float64[])
    # For the simple model, body is not connected (ignore body effect)
    # Just call 3-terminal version
    stamp!(M, ctx, d, g, s; x=x)
    return nothing
end
