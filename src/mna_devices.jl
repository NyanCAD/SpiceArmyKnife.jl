# MNA Device Implementations
# Circuit elements that stamp into the MNA matrix system

export MNAResistor, MNACapacitor, MNAInductor, MNAVoltageSource, MNACurrentSource,
       MNADiode, MNAVCVS, MNAVCCS, MNAGround, stamp!

#=
Device Interface:
Each device implements stamp!(device, circuit::MNACircuit) which adds
the device's contributions to the MNA matrices.
=#

abstract type MNADevice end

"""
    MNAResistor

Linear resistor: V = I * R
"""
struct MNAResistor <: MNADevice
    n_pos::MNANet
    n_neg::MNANet
    resistance::Float64
    name::Symbol
end

function MNAResistor(circuit::MNACircuit, n_pos, n_neg, resistance; name=:R)
    net_pos = get_net!(circuit, n_pos)
    net_neg = get_net!(circuit, n_neg)
    MNAResistor(net_pos, net_neg, Float64(resistance), name)
end

function stamp!(device::MNAResistor, circuit::MNACircuit)
    g = 1.0 / device.resistance
    stamp_conductance!(circuit, device.n_pos, device.n_neg, g)
end

"""
    MNACapacitor

Linear capacitor: I = C * dV/dt
"""
struct MNACapacitor <: MNADevice
    n_pos::MNANet
    n_neg::MNANet
    capacitance::Float64
    name::Symbol
end

function MNACapacitor(circuit::MNACircuit, n_pos, n_neg, capacitance; name=:C)
    net_pos = get_net!(circuit, n_pos)
    net_neg = get_net!(circuit, n_neg)
    MNACapacitor(net_pos, net_neg, Float64(capacitance), name)
end

function stamp!(device::MNACapacitor, circuit::MNACircuit)
    stamp_capacitance!(circuit, device.n_pos, device.n_neg, device.capacitance)
end

"""
    MNAInductor

Linear inductor: V = L * dI/dt
Requires a branch current variable.
"""
struct MNAInductor <: MNADevice
    n_pos::MNANet
    n_neg::MNANet
    inductance::Float64
    branch::BranchVar
    name::Symbol
end

function MNAInductor(circuit::MNACircuit, n_pos, n_neg, inductance; name=:L)
    net_pos = get_net!(circuit, n_pos)
    net_neg = get_net!(circuit, n_neg)
    branch = get_branch!(circuit, Symbol(name, :_I))
    MNAInductor(net_pos, net_neg, Float64(inductance), branch, name)
end

function stamp!(device::MNAInductor, circuit::MNACircuit)
    stamp_inductor!(circuit, device.n_pos, device.n_neg, device.branch, device.inductance)
end

"""
    MNAVoltageSource

Independent voltage source: V(pos, neg) = voltage
"""
struct MNAVoltageSource <: MNADevice
    n_pos::MNANet
    n_neg::MNANet
    dc::Float64
    ac::ComplexF64
    tran_func::Union{Nothing, Function}  # t -> voltage
    branch::BranchVar
    name::Symbol
end

function MNAVoltageSource(circuit::MNACircuit, n_pos, n_neg;
                          dc=0.0, ac=0.0+0.0im, tran=nothing, name=:V)
    net_pos = get_net!(circuit, n_pos)
    net_neg = get_net!(circuit, n_neg)
    branch = get_branch!(circuit, Symbol(name, :_I))

    # Handle transient function
    tran_func = if tran isa Function
        tran
    elseif tran !== nothing
        t -> Float64(tran)  # constant
    else
        nothing
    end

    MNAVoltageSource(net_pos, net_neg, Float64(dc), ComplexF64(ac), tran_func, branch, name)
end

function stamp!(device::MNAVoltageSource, circuit::MNACircuit)
    # For DC analysis, use DC value
    voltage = device.dc

    if circuit.mode == :tran && device.tran_func !== nothing
        # Add time-dependent source
        branch_idx = branch_index(circuit, device.branch)
        add_time_source!(circuit, (t, p) -> begin
            v = device.tran_func(t)
            [(branch_idx, v)]
        end)
        voltage = 0.0  # Will be overridden by time source
    end

    stamp_voltage_source!(circuit, device.n_pos, device.n_neg, device.branch, voltage)
end

"""
    MNACurrentSource

Independent current source: I(pos -> neg) = current
"""
struct MNACurrentSource <: MNADevice
    n_pos::MNANet
    n_neg::MNANet
    dc::Float64
    ac::ComplexF64
    tran_func::Union{Nothing, Function}
    name::Symbol
end

function MNACurrentSource(circuit::MNACircuit, n_pos, n_neg;
                          dc=0.0, ac=0.0+0.0im, tran=nothing, name=:I)
    net_pos = get_net!(circuit, n_pos)
    net_neg = get_net!(circuit, n_neg)

    tran_func = if tran isa Function
        tran
    elseif tran !== nothing
        t -> Float64(tran)
    else
        nothing
    end

    MNACurrentSource(net_pos, net_neg, Float64(dc), ComplexF64(ac), tran_func, name)
end

function stamp!(device::MNACurrentSource, circuit::MNACircuit)
    current = device.dc

    if circuit.mode == :tran && device.tran_func !== nothing
        i_pos = node_index(device.n_pos)
        i_neg = node_index(device.n_neg)
        add_time_source!(circuit, (t, p) -> begin
            i = device.tran_func(t)
            contributions = Tuple{Int, Float64}[]
            if i_pos > 0
                push!(contributions, (i_pos, -i))
            end
            if i_neg > 0
                push!(contributions, (i_neg, i))
            end
            contributions
        end)
        current = 0.0
    end

    stamp_current_source!(circuit, device.n_pos, device.n_neg, current)
end

"""
    MNAGround

Ground reference: V = 0
"""
struct MNAGround <: MNADevice
    net::MNANet
    branch::BranchVar
    name::Symbol
end

function MNAGround(circuit::MNACircuit, net; name=:GND)
    n = get_net!(circuit, net)
    # Ground is implemented as a voltage source to ground (index 0)
    branch = get_branch!(circuit, Symbol(name, :_I))
    MNAGround(n, branch, name)
end

function stamp!(device::MNAGround, circuit::MNACircuit)
    # Ground sets the node voltage to 0
    # This is like a voltage source from the node to the reference ground
    i_n = node_index(device.net)
    i_br = branch_index(circuit, device.branch)

    # KCL at node: add branch current
    if i_n > 0
        push!(circuit.G_i, i_n)
        push!(circuit.G_j, i_br)
        push!(circuit.G_v, 1.0)
    end

    # Voltage constraint: V_n = 0
    if i_n > 0
        push!(circuit.G_i, i_br)
        push!(circuit.G_j, i_n)
        push!(circuit.G_v, 1.0)
    end

    # RHS: 0 (voltage = 0)
    push!(circuit.b_i, i_br)
    push!(circuit.b_v, 0.0)
end

"""
    MNADiode

Semiconductor diode with Shockley model.
"""
struct MNADiode <: MNADevice
    n_anode::MNANet
    n_cathode::MNANet
    model::DiodeModel
    name::Symbol
end

function MNADiode(circuit::MNACircuit, n_anode, n_cathode;
                  IS=1e-14, N=1.0, RS=0.0, BV=Inf, IBV=1e-3,
                  CJO=0.0, VJ=1.0, M=0.5, TT=0.0, AREA=1.0,
                  name=:D)
    net_anode = get_net!(circuit, n_anode)
    net_cathode = get_net!(circuit, n_cathode)
    model = DiodeModel(IS=IS, N=N, RS=RS, BV=BV, IBV=IBV,
                       CJO=CJO, VJ=VJ, M=M, TT=TT, AREA=AREA)
    MNADiode(net_anode, net_cathode, model, name)
end

function stamp!(device::MNADiode, circuit::MNACircuit)
    stamp_diode!(circuit, device.n_anode, device.n_cathode, device.model, device.name)
end

"""
    MNAVCVS

Voltage-Controlled Voltage Source: V(pos, neg) = gain * V(ctrl_pos, ctrl_neg)
"""
struct MNAVCVS <: MNADevice
    n_pos::MNANet
    n_neg::MNANet
    nc_pos::MNANet
    nc_neg::MNANet
    gain::Float64
    branch::BranchVar
    name::Symbol
end

function MNAVCVS(circuit::MNACircuit, n_pos, n_neg, nc_pos, nc_neg, gain; name=:E)
    net_pos = get_net!(circuit, n_pos)
    net_neg = get_net!(circuit, n_neg)
    net_c_pos = get_net!(circuit, nc_pos)
    net_c_neg = get_net!(circuit, nc_neg)
    branch = get_branch!(circuit, Symbol(name, :_I))
    MNAVCVS(net_pos, net_neg, net_c_pos, net_c_neg, Float64(gain), branch, name)
end

function stamp!(device::MNAVCVS, circuit::MNACircuit)
    stamp_vcvs!(circuit, device.n_pos, device.n_neg,
                device.nc_pos, device.nc_neg, device.branch, device.gain)
end

"""
    MNAVCCS

Voltage-Controlled Current Source: I(pos -> neg) = gain * V(ctrl_pos, ctrl_neg)
"""
struct MNAVCCS <: MNADevice
    n_pos::MNANet
    n_neg::MNANet
    nc_pos::MNANet
    nc_neg::MNANet
    gain::Float64
    name::Symbol
end

function MNAVCCS(circuit::MNACircuit, n_pos, n_neg, nc_pos, nc_neg, gain; name=:G)
    net_pos = get_net!(circuit, n_pos)
    net_neg = get_net!(circuit, n_neg)
    net_c_pos = get_net!(circuit, nc_pos)
    net_c_neg = get_net!(circuit, nc_neg)
    MNAVCCS(net_pos, net_neg, net_c_pos, net_c_neg, Float64(gain), name)
end

function stamp!(device::MNAVCCS, circuit::MNACircuit)
    stamp_vccs!(circuit, device.n_pos, device.n_neg,
                device.nc_pos, device.nc_neg, device.gain)
end

#=
Convenience functions for building circuits
=#

"""
    resistor!(circuit, n1, n2, r; name=:R)

Add a resistor to the circuit.
"""
function resistor!(circuit::MNACircuit, n1, n2, r; name=:R)
    dev = MNAResistor(circuit, n1, n2, r; name=name)
    stamp!(dev, circuit)
    return dev
end

"""
    capacitor!(circuit, n1, n2, c; name=:C)

Add a capacitor to the circuit.
"""
function capacitor!(circuit::MNACircuit, n1, n2, c; name=:C)
    dev = MNACapacitor(circuit, n1, n2, c; name=name)
    stamp!(dev, circuit)
    return dev
end

"""
    inductor!(circuit, n1, n2, l; name=:L)

Add an inductor to the circuit.
"""
function inductor!(circuit::MNACircuit, n1, n2, l; name=:L)
    dev = MNAInductor(circuit, n1, n2, l; name=name)
    stamp!(dev, circuit)
    return dev
end

"""
    vsource!(circuit, n_pos, n_neg; dc=0.0, ac=0.0, tran=nothing, name=:V)

Add a voltage source to the circuit.
"""
function vsource!(circuit::MNACircuit, n_pos, n_neg; dc=0.0, ac=0.0+0.0im, tran=nothing, name=:V)
    dev = MNAVoltageSource(circuit, n_pos, n_neg; dc=dc, ac=ac, tran=tran, name=name)
    stamp!(dev, circuit)
    return dev
end

"""
    isource!(circuit, n_pos, n_neg; dc=0.0, ac=0.0, tran=nothing, name=:I)

Add a current source to the circuit.
"""
function isource!(circuit::MNACircuit, n_pos, n_neg; dc=0.0, ac=0.0+0.0im, tran=nothing, name=:I)
    dev = MNACurrentSource(circuit, n_pos, n_neg; dc=dc, ac=ac, tran=tran, name=name)
    stamp!(dev, circuit)
    return dev
end

"""
    ground!(circuit, net; name=:GND)

Add a ground reference to the circuit.
"""
function ground!(circuit::MNACircuit, net; name=:GND)
    dev = MNAGround(circuit, net; name=name)
    stamp!(dev, circuit)
    return dev
end

"""
    diode!(circuit, n_anode, n_cathode; kwargs...)

Add a diode to the circuit.
"""
function diode!(circuit::MNACircuit, n_anode, n_cathode; kwargs...)
    dev = MNADiode(circuit, n_anode, n_cathode; kwargs...)
    stamp!(dev, circuit)
    return dev
end

"""
    vcvs!(circuit, n_pos, n_neg, nc_pos, nc_neg, gain; name=:E)

Add a voltage-controlled voltage source.
"""
function vcvs!(circuit::MNACircuit, n_pos, n_neg, nc_pos, nc_neg, gain; name=:E)
    dev = MNAVCVS(circuit, n_pos, n_neg, nc_pos, nc_neg, gain; name=name)
    stamp!(dev, circuit)
    return dev
end

"""
    vccs!(circuit, n_pos, n_neg, nc_pos, nc_neg, gain; name=:G)

Add a voltage-controlled current source.
"""
function vccs!(circuit::MNACircuit, n_pos, n_neg, nc_pos, nc_neg, gain; name=:G)
    dev = MNAVCCS(circuit, n_pos, n_neg, nc_pos, nc_neg, gain; name=name)
    stamp!(dev, circuit)
    return dev
end
