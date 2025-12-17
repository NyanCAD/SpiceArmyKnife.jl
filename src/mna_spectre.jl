# MNA-Spectre Integration Layer
# Bridges the spectre.jl parameterization with MNA circuit analysis

using ScopedValues

#=
This module provides MNA-compatible versions of the circuit primitives
that work with the spectre.jl parameterization infrastructure.

Key concepts:
- mna_circuit: Scoped variable holding the current MNACircuit being built
- MNANetRef: Reference to a node in the MNA circuit (wraps MNANet)
- MNA device types: Callable structs that stamp into mna_circuit[] when called
=#

export mna_circuit, MNANetRef, mna_net, mna_ground
export MNASimpleResistor, MNASimpleCapacitor, MNASimpleInductor
export SpcVoltageSource, SpcCurrentSource, MNASimpleDiode
export SpcVCVS, SpcVCCS, MNAGnd

# Scoped variable to hold the circuit during construction
const mna_circuit = ScopedValue{Union{Nothing, MNACircuit}}(nothing)

# Simulation mode for MNA
const mna_sim_mode = ScopedValue{Symbol}(:dcop)

# Simulation spec (temperature, gmin, etc.)
Base.@kwdef struct MNASimSpec
    time::Float64 = 0.0
    temp::Float64 = 27.0
    gmin::Float64 = 1e-12
    scale::Float64 = 1.0
end

const mna_spec = ScopedValue{MNASimSpec}(MNASimSpec())

"""
    MNANetRef

Reference to a node in the MNA circuit.
This is the MNA equivalent of the DAECompiler Net type.
"""
struct MNANetRef
    net::MNANet
    multiplier::Float64
end

MNANetRef(net::MNANet) = MNANetRef(net, 1.0)
MNANetRef(ref::MNANetRef, m::Float64) = MNANetRef(ref.net, ref.multiplier * m)

# Get voltage of a net (for use in expressions)
voltage(ref::MNANetRef) = ref  # Returns self for now; actual voltage comes from solution

"""
    mna_net(name::Union{Symbol, String})

Create a new net in the current MNA circuit.
"""
function mna_net(name::Union{Symbol, String})
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")
    net = get_net!(ckt, Symbol(name))
    return MNANetRef(net)
end

function mna_net()
    mna_net(gensym(:net))
end

"""
    mna_ground()

Get or create the ground net.
"""
function mna_ground()
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")
    return MNANetRef(ckt.ground)
end

#=
MNA-compatible device types
These match the SpectreEnvironment API but stamp into MNA matrices
=#

"""
    MNASimpleResistor

MNA-compatible resistor.
"""
struct MNASimpleResistor
    r::Float64
    rsh::Float64
    w::Float64
    l::Float64
    narrow::Float64
    short::Float64
end

function MNASimpleResistor(; r=1.0, rsh=50.0, wdef=1e-6, w=wdef, l=wdef, narrow=0.0, short=0.0, kwargs...)
    MNASimpleResistor(
        Float64(undefault(r)),
        Float64(undefault(rsh)),
        Float64(undefault(w)),
        Float64(undefault(l)),
        Float64(undefault(narrow)),
        Float64(undefault(short))
    )
end

function (R::MNASimpleResistor)(A::MNANetRef, B::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    # Calculate effective resistance
    res = if R.r != 1.0  # non-default
        R.r
    else
        R.rsh * (R.l - R.short) / (R.w - R.narrow)
    end

    # Apply multiplier (parallel instances reduce resistance)
    # Note: we use m directly, not A.multiplier (which is for net scaling in subcircuits)
    res_eff = res / m

    # Stamp conductance
    stamp_conductance!(ckt, A.net, B.net, 1.0 / res_eff)
end

"""
    MNASimpleCapacitor

MNA-compatible capacitor.
"""
struct MNASimpleCapacitor
    capacitance::Float64
end

MNASimpleCapacitor(; c=1.0, kwargs...) = MNASimpleCapacitor(Float64(undefault(c)))

function (C::MNASimpleCapacitor)(A::MNANetRef, B::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    # Multiplier increases capacitance (parallel instances)
    cap_eff = C.capacitance * m

    stamp_capacitance!(ckt, A.net, B.net, cap_eff)
end

"""
    MNASimpleInductor

MNA-compatible inductor.
"""
struct MNASimpleInductor
    inductance::Float64
end

MNASimpleInductor(; l=1e-9, kwargs...) = MNASimpleInductor(Float64(undefault(l)))

function (L::MNASimpleInductor)(A::MNANetRef, B::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    # Create branch for inductor current
    branch_name = gensym(:L)
    branch = get_branch!(ckt, branch_name)

    # Multiplier reduces inductance (parallel instances)
    ind_eff = L.inductance / (m * A.multiplier)

    stamp_inductor!(ckt, A.net, B.net, branch, ind_eff)
end

"""
    SpcVoltageSource

Spectre-compatible voltage source (uses scoped mna_circuit[]).
"""
struct SpcVoltageSource{T}
    dc::T
    tran::T
    ac::Complex{Float64}
end

function SpcVoltageSource(; type=nothing, dc=nothing, ac=0.0+0.0im, tran=nothing, kwargs...)
    dc_val = something(dc, tran, 0.0)
    tran_val = something(tran, dc_val)
    SpcVoltageSource(Float64(undefault(dc_val)), Float64(undefault(tran_val)), Complex{Float64}(ac))
end

function (VS::SpcVoltageSource)(A::MNANetRef, B::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    # Create branch for voltage source current
    branch_name = gensym(:V)
    branch = get_branch!(ckt, branch_name)

    # Get voltage based on simulation mode
    mode = mna_sim_mode[]
    voltage = if mode === :dcop || mode === :tranop
        VS.dc
    else
        VS.tran
    end

    stamp_voltage_source!(ckt, A.net, B.net, branch, voltage)
end

"""
    SpcCurrentSource

Spectre-compatible current source (uses scoped mna_circuit[]).
"""
struct SpcCurrentSource{T}
    dc::T
    tran::T
    ac::Complex{Float64}
end

function SpcCurrentSource(; type=nothing, dc=nothing, ac=0.0+0.0im, tran=nothing, kwargs...)
    dc_val = something(dc, tran, 0.0)
    tran_val = something(tran, dc_val)
    SpcCurrentSource(Float64(undefault(dc_val)), Float64(undefault(tran_val)), Complex{Float64}(ac))
end

function (IS::SpcCurrentSource)(A::MNANetRef, B::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    # Get current based on simulation mode
    mode = mna_sim_mode[]
    current = if mode === :dcop || mode === :tranop
        IS.dc
    else
        IS.tran
    end

    # Apply multiplier (parallel sources add current)
    current_eff = current * m * A.multiplier

    stamp_current_source!(ckt, A.net, B.net, current_eff)
end

"""
    MNASimpleDiode

MNA-compatible diode.
"""
struct MNASimpleDiode
    model::DiodeModel
end

function MNASimpleDiode(;
    AREA=1.0, BV=Inf, BVJ=Inf, CJO=0.0, EG=1.11, FC=0.5,
    IBV=0.001, IS=1e-14, KF=0.0, M=0.5, N=1.0, RS=0.0,
    TNOM=27.0, TT=0.0, VJ=1.0, XTI=3.0, kwargs...)

    model = DiodeModel(
        IS=Float64(undefault(IS)),
        N=Float64(undefault(N)),
        RS=Float64(undefault(RS)),
        BV=Float64(undefault(BV)),
        IBV=Float64(undefault(IBV)),
        CJO=Float64(undefault(CJO)),
        VJ=Float64(undefault(VJ)),
        M=Float64(undefault(M)),
        TT=Float64(undefault(TT)),
        AREA=Float64(undefault(AREA))
    )
    MNASimpleDiode(model)
end

function (D::MNASimpleDiode)(A::MNANetRef, B::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    # Stamp diode as nonlinear element
    stamp_diode!(ckt, A.net, B.net, D.model, gensym(:D))
end

"""
    SpcVCVS

Spectre-compatible Voltage-Controlled Voltage Source.
"""
struct SpcVCVS
    gain::Float64
    voltage::Float64
end

SpcVCVS(; gain=1.0, vol=nothing, value=nothing, kwargs...) =
    SpcVCVS(Float64(undefault(gain)), Float64(something(undefault(vol), undefault(value), 0.0)))

function (S::SpcVCVS)(A::MNANetRef, B::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    branch = get_branch!(ckt, gensym(:E))
    stamp_voltage_source!(ckt, A.net, B.net, branch, S.voltage)
end

function (S::SpcVCVS)(A::MNANetRef, B::MNANetRef, C::MNANetRef, D::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    branch = get_branch!(ckt, gensym(:E))
    stamp_vcvs!(ckt, A.net, B.net, C.net, D.net, branch, S.gain)
end

"""
    SpcVCCS

Spectre-compatible Voltage-Controlled Current Source.
"""
struct SpcVCCS
    gain::Float64
    current::Float64
end

SpcVCCS(; gain=1.0, cur=nothing, value=nothing, kwargs...) =
    SpcVCCS(Float64(undefault(gain)), Float64(something(undefault(cur), undefault(value), 0.0)))

function (S::SpcVCCS)(A::MNANetRef, B::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    stamp_current_source!(ckt, A.net, B.net, S.current)
end

function (S::SpcVCCS)(A::MNANetRef, B::MNANetRef, C::MNANetRef, D::MNANetRef; dscope=nothing, m=1.0)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    stamp_vccs!(ckt, A.net, B.net, C.net, D.net, S.gain)
end

"""
    MNAGnd

MNA-compatible ground reference.
"""
struct MNAGnd end

function (::MNAGnd)(A::MNANetRef; dscope=nothing)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    # Ground stamps a large conductance to ground (node 0)
    # This ensures the node is tied to 0V
    # Actually, in MNA ground is implicit (node index 0)
    # We just need to ensure this net is the ground net
    if A.net.index != 0
        # Stamp a very large conductance to ground
        stamp_conductance!(ckt, A.net, ckt.ground, 1e12)
    end
end

#=
Parallel instances wrapper for MNA
=#

struct MNAParallelInstances
    device
    multiplier::Float64
end

MNAParallelInstances(device, m::Number) = MNAParallelInstances(device, Float64(undefault(m)))

function (pi::MNAParallelInstances)(nets...; kwargs...)
    # Apply multiplier to each net reference
    nets_scaled = map(nets) do net
        MNANetRef(net, pi.multiplier)
    end
    return pi.device(nets_scaled...; m=pi.multiplier, kwargs...)
end

#=
Circuit builder - creates MNA circuit from spectre netlist functions
=#

"""
    build_mna_circuit(circuit_func; temp=27.0, gmin=1e-12)

Build an MNA circuit from a spectre-style circuit function.
The circuit_func should be a callable that instantiates devices when called.

Returns a compiled MNAODESystem ready for simulation.
"""
function build_mna_circuit(circuit_func; temp=27.0, gmin=1e-12)
    ckt = MNACircuit(temp=temp, gmin=gmin)

    # Run the circuit function with the MNA circuit in scope
    @with mna_circuit => ckt begin
        @with mna_spec => MNASimSpec(temp=temp, gmin=gmin) begin
            circuit_func()
        end
    end

    return compile_ode_system(ckt)
end

"""
    simulate_dc(circuit_func; temp=27.0, gmin=1e-12, kwargs...)

Run DC operating point analysis on the circuit.
"""
function simulate_dc(circuit_func; temp=27.0, gmin=1e-12, maxiter=100, tol=1e-9)
    ckt = MNACircuit(temp=temp, gmin=gmin)

    @with mna_circuit => ckt begin
        @with mna_sim_mode => :dcop begin
            @with mna_spec => MNASimSpec(temp=temp, gmin=gmin) begin
                circuit_func()
            end
        end
    end

    return solve_dc!(ckt; maxiter=maxiter, tol=tol), ckt
end

"""
    simulate_tran(circuit_func, tspan; temp=27.0, gmin=1e-12, u0=nothing)

Set up transient simulation on the circuit.
Returns (f!, u0, tspan, mass_matrix, system) for use with DifferentialEquations.jl
"""
function simulate_tran(circuit_func, tspan; temp=27.0, gmin=1e-12, u0=nothing)
    ckt = MNACircuit(temp=temp, gmin=gmin)

    @with mna_circuit => ckt begin
        @with mna_sim_mode => :tran begin
            @with mna_spec => MNASimSpec(temp=temp, gmin=gmin) begin
                circuit_func()
            end
        end
    end

    return transient_problem(ckt, tspan; u0=u0)
end

#=
Named wrapper for MNA devices (compatible with spectre.jl codegen)
=#

struct MNANamed{T}
    element::T
    name::Symbol
end

MNANamed(element, name::String) = MNANamed(element, Symbol(name))

function (n::MNANamed)(args...; kwargs...)
    return n.element(args...; dscope=n.name, kwargs...)
end

# For net creation, return a named net
function (n::MNANamed{typeof(mna_net)})(args...)
    return mna_net(n.name)
end

#=
VA Device wrapper for MNA spectre integration
Allows VA-generated devices (from mna_va_load) to work with the spectre infrastructure
=#

"""
    MNAVADeviceWrapper{T}

Wraps a VA-generated device type (from mna_va_load) for use with MNA spectre infrastructure.
The VA device must have a constructor of form: DeviceType(circuit, pin1, pin2, ...; kwargs...)
"""
struct MNAVADeviceWrapper{T}
    device_type::Type{T}
    default_params::Dict{Symbol, Any}
end

MNAVADeviceWrapper(T::Type; kwargs...) = MNAVADeviceWrapper(T, Dict{Symbol, Any}(kwargs...))

function (w::MNAVADeviceWrapper)(nets::MNANetRef...; dscope=nothing, m=1.0, kwargs...)
    ckt = mna_circuit[]
    ckt === nothing && error("No MNA circuit in scope")

    # Merge default params with call-time kwargs
    all_params = merge(w.default_params, Dict{Symbol, Any}(kwargs...))

    # Create device name
    name = dscope !== nothing ? dscope : gensym(:va_dev)

    # Get raw MNANet from each MNANetRef
    raw_nets = [net.net for net in nets]

    # Create and stamp the VA device
    device = w.device_type(ckt, raw_nets...; name=name, all_params...)
    stamp!(device, ckt)

    return device
end

"""
    wrap_va_device(DeviceType; kwargs...)

Create a wrapper for a VA-generated device type that works with MNA spectre infrastructure.
Default parameters can be provided as keyword arguments.

Example:
    mna_va_load(@__MODULE__, "path/to/model.va")  # Creates MyDevice type
    my_device = wrap_va_device(MyDevice; param1=1.0, param2=2.0)
    # Use in circuit: my_device(net1, net2, net3, net4; other_param=3.0)
"""
wrap_va_device(T::Type; kwargs...) = MNAVADeviceWrapper(T; kwargs...)
