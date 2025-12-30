#==============================================================================#
# NonlinearStaticCircuit: Zero-Allocation Nonlinear Circuit Evaluation
#
# This extends StaticCircuit to support nonlinear devices by storing
# stamp evaluators that compute matrix values from operating point.
#
# Architecture:
# - Store stamp "programs" at compile time
# - Use @generated functions to inline evaluation
# - Return fresh SMatrix/SVector each iteration (stack allocated)
#==============================================================================#

using StaticArrays
using LinearAlgebra

export NonlinearStaticCircuit, compile_nonlinear_static
export nonlinear_residual

#==============================================================================#
# Stamp Types for Nonlinear Evaluation
#==============================================================================#

"""
    StampOp

Abstract type for stamp operations that compute values from operating point.
"""
abstract type StampOp end

"""
    ConstantStamp{V}

Stamp with constant value (resistor, fixed source, etc.).
"""
struct ConstantStamp{V} <: StampOp
    row::Int
    col::Int
    target::Symbol  # :G, :C, or :b
    value::V
end

"""
    LinearStamp{V}

Stamp that scales linearly with a state variable.
Example: g_m * V_gs for MOSFET transconductance.
"""
struct LinearStamp{V} <: StampOp
    row::Int
    col::Int
    target::Symbol
    coeff::V        # Coefficient
    state_idx::Int  # Index into u vector
end

"""
    QuadraticStamp{V}

Stamp that depends quadratically on state.
Example: (V_gs - V_th)^2 for MOSFET drain current.
"""
struct QuadraticStamp{V} <: StampOp
    row::Int
    col::Int
    target::Symbol
    coeff::V
    state_idx1::Int
    state_idx2::Int
end

"""
    FunctionStamp{F,V}

Stamp with arbitrary function evaluation.
Used for complex device models (diode exponential, etc.).
"""
struct FunctionStamp{F,V} <: StampOp
    row::Int
    col::Int
    target::Symbol
    func::F         # (u, t, params) -> V
    params::V       # Device parameters
end

#==============================================================================#
# NonlinearStaticCircuit Definition
#==============================================================================#

"""
    NonlinearStaticCircuit{N,T,K}

Circuit with nonlinear stamps for zero-allocation evaluation.

# Type Parameters
- `N`: System size (compile-time constant)
- `T`: Element type (Float64 or Float32)
- `K`: Number of stamps

# Fields
- `stamps`: Tuple of stamp operations
- `node_names`: Node names tuple

# How It Works
The `stamps` tuple contains operations that evaluate to matrix values.
The `nonlinear_residual` function iterates through stamps, computing
values and accumulating into fresh SMatrix/SVector objects.

Since StaticArrays are immutable and stack-allocated, this achieves
zero heap allocation while supporting nonlinear devices.
"""
struct NonlinearStaticCircuit{N,T,K,Stamps<:Tuple}
    stamps::Stamps
    node_names::NTuple{N,Symbol}
end

#==============================================================================#
# Evaluation Functions
#==============================================================================#

"""
    evaluate_stamp(stamp::ConstantStamp, u, t) -> value

Evaluate a constant stamp (no computation needed).
"""
@inline evaluate_stamp(s::ConstantStamp, u, t) = s.value

"""
    evaluate_stamp(stamp::LinearStamp, u, t) -> value

Evaluate a linear stamp: coeff * u[idx].
"""
@inline evaluate_stamp(s::LinearStamp, u, t) = s.coeff * u[s.state_idx]

"""
    evaluate_stamp(stamp::QuadraticStamp, u, t) -> value

Evaluate a quadratic stamp: coeff * u[idx1] * u[idx2].
"""
@inline evaluate_stamp(s::QuadraticStamp, u, t) = s.coeff * u[s.state_idx1] * u[s.state_idx2]

"""
    evaluate_stamp(stamp::FunctionStamp, u, t) -> value

Evaluate a function stamp: func(u, t, params).
"""
@inline evaluate_stamp(s::FunctionStamp, u, t) = s.func(u, t, s.params)

#==============================================================================#
# Matrix Building via @generated
#==============================================================================#

"""
    build_matrices(circuit::NonlinearStaticCircuit{N,T}, u, t) -> (G, C, b)

Build G, C, b matrices from operating point using compile-time unrolled evaluation.

This is a @generated function that produces specialized code for each circuit.
The generated code iterates through stamps at compile time, producing efficient
inlined evaluation with no dynamic dispatch.
"""
@generated function build_matrices(circuit::NonlinearStaticCircuit{N,T,K,Stamps},
                                   u::SVector{N,T}, t::Real) where {N,T,K,Stamps}
    # Generate code that evaluates all stamps and accumulates into matrices
    stamp_types = Stamps.parameters

    # Initialize accumulator expressions
    G_init = quote
        G_data = @MMatrix zeros($T, $N, $N)
    end
    C_init = quote
        C_data = @MMatrix zeros($T, $N, $N)
    end
    b_init = quote
        b_data = @MVector zeros($T, $N)
    end

    # Generate evaluation code for each stamp
    eval_exprs = Expr[]
    for (i, stype) in enumerate(stamp_types)
        push!(eval_exprs, quote
            s = circuit.stamps[$i]
            val = evaluate_stamp(s, u, t)
            if s.target === :G
                G_data[s.row, s.col] += val
            elseif s.target === :C
                C_data[s.row, s.col] += val
            else  # :b
                b_data[s.row] += val
            end
        end)
    end

    # Convert to immutable
    finalize = quote
        G = SMatrix{$N,$N,$T}(G_data)
        C = SMatrix{$N,$N,$T}(C_data)
        b = SVector{$N,$T}(b_data)
        return (G, C, b)
    end

    return quote
        $G_init
        $C_init
        $b_init
        $(eval_exprs...)
        $finalize
    end
end

"""
    nonlinear_residual(du::SVector{N,T}, u::SVector{N,T},
                       circuit::NonlinearStaticCircuit{N,T}, t) -> SVector{N,T}

Zero-allocation nonlinear DAE residual evaluation.

Computes: F(du, u, t) = C(u)*du + G(u)*u - b(u,t) = 0

All intermediate matrices are stack-allocated StaticArrays.
"""
@inline function nonlinear_residual(du::SVector{N,T}, u::SVector{N,T},
                                     circuit::NonlinearStaticCircuit{N,T},
                                     t::Real) where {N,T}
    G, C, b = build_matrices(circuit, u, t)
    return C * du + G * u - b
end

#==============================================================================#
# Circuit Compilation
#==============================================================================#

"""
    compile_nonlinear_static(builder, params, spec, ::Val{N}, T=Float64) -> NonlinearStaticCircuit

Compile a circuit builder into a NonlinearStaticCircuit.

This analyzes the circuit structure and extracts stamp operations.
Currently supports:
- Constant stamps (resistors, fixed sources)
- TODO: Linear stamps (g_m transconductance)
- TODO: Nonlinear stamps (diode I-V)

For now, this falls back to constant stamps (same as StaticCircuit).
Full nonlinear support requires VA model integration.
"""
function compile_nonlinear_static(builder::F, params::P, spec::S,
                                   ::Val{N}, ::Type{T}=Float64) where {F,P,S,N,T}
    # Build circuit once to discover structure
    ctx = builder(params, spec; x=zeros(T, N))
    n = system_size(ctx)
    @assert n == N "Circuit size ($n) doesn't match Val($N)"

    # Extract stamps as ConstantStamp operations
    # (Full nonlinear support would analyze stamp types)
    stamps = StampOp[]

    # G stamps
    for (i_raw, j_raw, v) in zip(ctx.G_I, ctx.G_J, ctx.G_V)
        i = resolve_index(ctx, i_raw)
        j = resolve_index(ctx, j_raw)
        if 1 <= i <= N && 1 <= j <= N
            push!(stamps, ConstantStamp{T}(i, j, :G, T(v)))
        end
    end

    # C stamps
    for (i_raw, j_raw, v) in zip(ctx.C_I, ctx.C_J, ctx.C_V)
        i = resolve_index(ctx, i_raw)
        j = resolve_index(ctx, j_raw)
        if 1 <= i <= N && 1 <= j <= N
            push!(stamps, ConstantStamp{T}(i, j, :C, T(v)))
        end
    end

    # b stamps (direct)
    for i in 1:min(length(ctx.b), N)
        if ctx.b[i] != 0
            push!(stamps, ConstantStamp{T}(i, 1, :b, T(ctx.b[i])))
        end
    end

    # b stamps (deferred)
    for (idx, val) in zip(ctx.b_I, ctx.b_V)
        i = resolve_index(ctx, idx)
        if 1 <= i <= N
            push!(stamps, ConstantStamp{T}(i, 1, :b, T(val)))
        end
    end

    stamps_tuple = Tuple(stamps)
    K = length(stamps)

    names = ntuple(i -> i <= length(ctx.node_names) ? ctx.node_names[i] :
                        Symbol("I_", ctx.current_names[i - length(ctx.node_names)]), N)

    return NonlinearStaticCircuit{N,T,K,typeof(stamps_tuple)}(stamps_tuple, names)
end
