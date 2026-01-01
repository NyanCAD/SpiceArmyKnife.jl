using ChainRulesCore, StaticArrays

# This gets imported by all generated Spectre code. The function names
# exported here should correspond to what is made available by Spectre

struct PWLConstructError <: CedarException
    ts
    ys
end

function Base.showerror(io::IO, err::PWLConstructError)
    println(io, "PWL must have an equal number of x and y values")
end

function find_t_in_ts(ts, t)
    idx = Base.searchsortedfirst(ts, t)
    if idx <= length(ts) && ts[idx] == t
        return idx + 1
    end
    return idx
end

# Disable autodiff through `t` search
function ChainRulesCore.frule((_, _), ::typeof(find_t_in_ts), ts, t)
    return find_t_in_ts(ts, t), ZeroTangent()
end


rem_right_semi(t, r) = t % r
function ChainRulesCore.frule((_, δt, δr), ::typeof(rem_right_semi), t, r)
    return (rem_right_semi(t, r), δt)
end

# Split our `wave` into `ts` and `ys`, hinting to the compiler what the length
# of these views are.
function wave_split(wave::SVector)
    idxs = SVector{div(length(wave),2)}(1:2:length(wave))
    ts = @view(wave[idxs])
    ys = @view(wave[idxs.+1])
    return (ts, ys)
end

function pwl_at_time(ts, ys, t)
    if length(ts) != length(ys)
        throw(CedarSim.PWLConstructError(ts, ys))
    end
    i = find_t_in_ts(ts, t)
    type_stable_time = 0. * t
    if i <= 1
        # Signal is before the first timepoint, hold the first value.
        return ys[1] + type_stable_time
    end
    if i > length(ts)
        # Signal is beyond the final timepoint, hold the final value.
        return ys[end] + type_stable_time
    end
    if ys[i-1] == ys[i]
        # signal is constant/flat (singularity in y)
        return ys[i] + type_stable_time
    end
    if ts[i] == ts[i-1]
        # signal is infinitely steep (singularity in t)
        # This can occur when loading a serialized signal that had its time digits truncated.
        return (ys[i-1] + ys[i])/2 + type_stable_time
    end
    # The general case, where we must perform linear interpolation
    slope = (ys[i] - ys[i-1])/(ts[i] - ts[i-1])
    return ys[i-1] + (t - ts[i-1])*slope
end

# Stub for time_periodic_singularities! - was used by DAECompiler for event handling
# MNA handles time-dependent sources differently
@generated function time_periodic_singularities!(ts::StaticArrays.SVector, period = ts[end], count = 1)
    # No-op in MNA backend - singularities are handled by adaptive time stepping
    return :nothing
end

baremodule SpectreEnvironment

import ..Base
import ..CedarSim
import ForwardDiff
import Compat
import Distributions
import StaticArrays

import Base:
    +, *, -, ==, !=, /, ^, >, <,  <=, >=,
    max, min, abs,
    log, exp, sqrt,
    sinh, cosh, tanh,
    sin, cos, tan,
    asinh, acosh, atanh,
    zero, atan,
    floor, ceil, trunc
import Base.Experimental: @overlay
import ..rem_right_semi, ..time_periodic_singularities!, ..pwl_at_time, ..wave_split

const arctan = atan
const ln = log
const pow = ^
int(x) = trunc(Int, x)
nint(x) = Base.round(Int, x)

export !, +, *, -, ==, !=, /, ^, >, <,  <=, >=,
    max, min, abs,
    ln, log, exp, sqrt,
    sinh, cosh, tanh,
    sin, cos, tan, atan, arctan,
    asinh, acosh, atanh,
    int, nint, floor, ceil, pow

# Scale parameter accessor
function var"$scale"()
    return CedarSim.undefault(CedarSim.spec[].scale)
end

const M_1_PI = 1/Base.pi

function pwl(wave)
    ts, ys = wave_split(wave)
    # Notify singularities at each of our timepoints (no-op in MNA)
    time_periodic_singularities!(ts)

    # Actually calculate the value to return
    return pwl_at_time(ts, ys, var"$time"())
end

function pulse(v1, v2, td, tr, tf, pw=Base.Inf, period=Base.Inf, count=-1)
    ts = StaticArrays.@SVector[
        td, td+tr, td+tr+pw, td+tr+pw+tf,
    ]
    ys = StaticArrays.@SVector[
        v1, v2, v2, v1,
    ]
    # Notify singularities at each of our timepoints (no-op in MNA)
    time_periodic_singularities!(ts, period, count)

    # Calculate value modulo our period
    t = rem_right_semi(CedarSim.spec[].time, period)
    return pwl_at_time(ts, ys, t)
end

# don't pirate Base.sin
function spsin(vo, va, freq, td=0, theta=0, phase=0, ncyles=Base.Inf)
    # see https://ltwiki.org/LTspiceHelp/LTspiceHelp/V_Voltage_Source.htm
    if td < var"$time"() < ncyles/freq
        vo+va*Base.exp(-(var"$time"()-td)*theta)*Base.sind(360*freq*(var"$time"()-td)+phase)
    else
        vo + va*Base.sind(phase)
    end
end

function agauss(nom, avar, sigma)
    rng = CedarSim.spec[].rng
    if rng === nothing
        nom
    else
        d = Distributions.Normal(0.0, avar)
        rn = Base.@noinline Base.rand(rng, d)
        nom + rn/sigma
    end
end

# Gets replaced by simulator time in the compiler override
function var"$time"()
    if CedarSim.sim_mode[] === :dcop || CedarSim.sim_mode[] === :tranop
        return 0.0
    else
        return CedarSim.spec[].time
    end
end

temper() = CedarSim.undefault(CedarSim.spec[].temp) # Celsius

const dc = :dc
const ac = :ac
const tran = :tran
export M_1_PI, dc, ac, tran, pwl, pulse, spsin, var"$time", agauss, temper, var"$scale"

end # baremodule SpectreEnvironment

export SpectreEnvironment
