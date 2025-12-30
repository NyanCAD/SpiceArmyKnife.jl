#==============================================================================#
# Compiled Circuit: Zero-Allocation Residual Evaluation
#
# This module provides a compiled circuit representation that achieves:
# - Zero allocation per residual call
# - Parameterized array types (Vector, CuArray, SVector)
# - Both in-place and out-of-place formulations
#
# Architecture:
# 1. CompiledCircuit{VecType, MatType, T}: Generic parameterized type
# 2. CPUCircuit{T}: Sparse matrices, in-place evaluation
# 3. StaticCircuit{N,T}: Dense SMatrix/SVector, out-of-place, GPU-ready
#
# The key insight is that circuit structure is known at compile time.
# We extract the sparsity pattern and values once, then provide
# zero-allocation evaluation by writing directly to preallocated storage.
#==============================================================================#

using SparseArrays
using LinearAlgebra
using StaticArrays

export CompiledCircuit, CPUCircuit, StaticCircuit
export compile_static_circuit, compile_cpu_circuit
export static_residual, static_residual!
export cpu_residual!
export solve_dc_static, solve_dc_cpu, to_static

#==============================================================================#
# CPU Circuit: Sparse Matrices, In-Place Evaluation
#==============================================================================#

"""
    CPUCircuit{T}

Compiled circuit optimized for CPU evaluation with sparse matrices.
Supports zero-allocation in-place residual evaluation.

# Fields
- `n::Int`: System size
- `G::SparseMatrixCSC{T,Int}`: Conductance matrix (fixed structure)
- `C::SparseMatrixCSC{T,Int}`: Capacitance matrix (fixed structure)
- `b::Vector{T}`: RHS vector
- `node_names::Vector{Symbol}`: Node names for debugging
- `current_names::Vector{Symbol}`: Current variable names
"""
struct CPUCircuit{T}
    n::Int
    G::SparseMatrixCSC{T,Int}
    C::SparseMatrixCSC{T,Int}
    b::Vector{T}
    node_names::Vector{Symbol}
    current_names::Vector{Symbol}
end

"""
    get_rhs_typed(ctx::MNAContext, ::Type{T}) -> Vector{T}

Get the RHS vector from context with specific element type, resolving deferred stamps.
"""
function get_rhs_typed(ctx::MNAContext, ::Type{T}) where T
    n = system_size(ctx)
    b = zeros(T, n)

    # Copy direct stamps
    for i in 1:min(length(ctx.b), n)
        b[i] = T(ctx.b[i])
    end

    # Apply deferred stamps
    for (idx, val) in zip(ctx.b_I, ctx.b_V)
        resolved = resolve_index(ctx, idx)
        if 1 <= resolved <= n
            b[resolved] += T(val)
        end
    end

    return b
end

"""
    compile_cpu_circuit(ctx::MNAContext, T::Type=Float64) -> CPUCircuit{T}

Compile an MNAContext into a CPUCircuit with fixed structure.
This is for linear circuits where G, C, b don't change during simulation.
"""
function compile_cpu_circuit(ctx::MNAContext, ::Type{T}=Float64) where T
    n = system_size(ctx)

    if n == 0
        return CPUCircuit{T}(
            0,
            spzeros(T, 0, 0),
            spzeros(T, 0, 0),
            T[],
            Symbol[],
            Symbol[]
        )
    end

    # Resolve indices and build sparse matrices
    G_I = [resolve_index(ctx, i) for i in ctx.G_I]
    G_J = [resolve_index(ctx, j) for j in ctx.G_J]
    G_V = T.(ctx.G_V)

    C_I = [resolve_index(ctx, i) for i in ctx.C_I]
    C_J = [resolve_index(ctx, j) for j in ctx.C_J]
    C_V = T.(ctx.C_V)

    G = sparse(G_I, G_J, G_V, n, n)
    C = sparse(C_I, C_J, C_V, n, n)
    b = get_rhs_typed(ctx, T)

    return CPUCircuit{T}(
        n, G, C, b,
        copy(ctx.node_names),
        copy(ctx.current_names)
    )
end

"""
    cpu_residual!(resid, du, u, circuit::CPUCircuit, t)

Zero-allocation in-place DAE residual evaluation.
Computes: F(du, u, t) = C*du + G*u - b

Note: This assumes G, C, b are constant (linear circuit).
For nonlinear circuits, use PrecompiledCircuit which rebuilds
values each iteration.
"""
function cpu_residual!(resid::Vector{T}, du::Vector{T}, u::Vector{T},
                       circuit::CPUCircuit{T}, t::Real) where T
    # F = C*du + G*u - b
    mul!(resid, circuit.C, du)
    mul!(resid, circuit.G, u, one(T), one(T))  # resid += G*u
    resid .-= circuit.b
    return nothing
end

#==============================================================================#
# Static Circuit: Dense Matrices, Zero-Allocation Out-of-Place
#==============================================================================#

"""
    StaticCircuit{N,T,L}

Compiled circuit for StaticArrays-based evaluation.
Uses dense SMatrix/SVector for zero-allocation out-of-place evaluation.

This is ideal for:
- Small circuits (N < 20 nodes)
- GPU ensemble simulation (many parallel instances)
- Monte Carlo / parameter sweeps

# Type Parameters
- `N`: System size (compile-time constant)
- `T`: Element type (Float64 or Float32)
- `L`: Length of matrix storage (N*N)

# Fields
- `G::SMatrix{N,N,T,L}`: Conductance matrix (dense)
- `C::SMatrix{N,N,T,L}`: Capacitance matrix (dense)
- `b::SVector{N,T}`: RHS vector
"""
struct StaticCircuit{N,T,L}
    G::SMatrix{N,N,T,L}
    C::SMatrix{N,N,T,L}
    b::SVector{N,T}
    node_names::NTuple{N,Symbol}
end

"""
    compile_static_circuit(ctx::MNAContext, ::Val{N}, T::Type=Float64) -> StaticCircuit{N,T}

Compile an MNAContext into a StaticCircuit with compile-time size N.

The `Val{N}` parameter enables compile-time specialization, allowing the
returned StaticCircuit to use SMatrix{N,N} which is stack-allocated.

# Example
```julia
ctx = MNAContext()
# ... stamp circuit with 3 nodes ...
circuit = compile_static_circuit(ctx, Val(3))
```
"""
function compile_static_circuit(ctx::MNAContext, ::Val{N}, ::Type{T}=Float64) where {N,T}
    n = system_size(ctx)

    @assert n == N "Circuit size ($n) doesn't match requested size ($N)"

    # Build dense matrices
    G_dense = zeros(T, N, N)
    C_dense = zeros(T, N, N)

    # Accumulate G stamps
    for (i_raw, j_raw, v) in zip(ctx.G_I, ctx.G_J, ctx.G_V)
        i = resolve_index(ctx, i_raw)
        j = resolve_index(ctx, j_raw)
        if 1 <= i <= N && 1 <= j <= N
            G_dense[i, j] += T(v)
        end
    end

    # Accumulate C stamps
    for (i_raw, j_raw, v) in zip(ctx.C_I, ctx.C_J, ctx.C_V)
        i = resolve_index(ctx, i_raw)
        j = resolve_index(ctx, j_raw)
        if 1 <= i <= N && 1 <= j <= N
            C_dense[i, j] += T(v)
        end
    end

    # Build b vector
    b_vec = get_rhs_typed(ctx, T)

    # Convert to StaticArrays
    G = SMatrix{N,N,T}(G_dense)
    C = SMatrix{N,N,T}(C_dense)
    b = SVector{N,T}(b_vec)

    # Node names as tuple
    names = ntuple(i -> i <= length(ctx.node_names) ? ctx.node_names[i] :
                        Symbol("I_", ctx.current_names[i - length(ctx.node_names)]), N)

    return StaticCircuit{N,T,N*N}(G, C, b, names)
end

"""
    compile_static_circuit(builder, params, spec, ::Val{N}, T::Type=Float64) -> StaticCircuit{N,T}

Compile a circuit builder into a StaticCircuit.
"""
function compile_static_circuit(builder::F, params::P, spec::S,
                                 ::Val{N}, ::Type{T}=Float64) where {F,P,S,N,T}
    ctx = builder(params, spec; x=T[])
    return compile_static_circuit(ctx, Val(N), T)
end

"""
    static_residual(du::SVector{N,T}, u::SVector{N,T}, circuit::StaticCircuit{N,T}, t) -> SVector{N,T}

Zero-allocation out-of-place DAE residual evaluation.
Returns: F(du, u, t) = C*du + G*u - b

This is the ensemble-GPU compatible formulation where all operations
use stack-allocated StaticArrays and the result is returned (not written
to a mutable array).

# Example
```julia
circuit = compile_static_circuit(ctx, Val(3))
u = @SVector [1.0, 2.0, 0.5]
du = @SVector zeros(3)
resid = static_residual(du, u, circuit, 0.0)  # Returns SVector{3,Float64}
```
"""
@inline function static_residual(du::SVector{N,T}, u::SVector{N,T},
                                  circuit::StaticCircuit{N,T,L}, t::Real) where {N,T,L}
    # F = C*du + G*u - b (all stack-allocated)
    return circuit.C * du + circuit.G * u - circuit.b
end

"""
    static_residual!(resid::MVector{N,T}, du::SVector{N,T}, u::SVector{N,T},
                     circuit::StaticCircuit{N,T}, t) -> Nothing

In-place variant for compatibility with standard SciML interfaces.
Uses MVector for the output to allow in-place modification while
keeping inputs as immutable SVector.
"""
@inline function static_residual!(resid::MVector{N,T}, du::SVector{N,T}, u::SVector{N,T},
                                   circuit::StaticCircuit{N,T,L}, t::Real) where {N,T,L}
    r = static_residual(du, u, circuit, t)
    resid .= r
    return nothing
end

#==============================================================================#
# Generic CompiledCircuit Interface
#==============================================================================#

"""
    CompiledCircuit

Abstract supertype for all compiled circuit representations.
Concrete types: CPUCircuit{T}, StaticCircuit{N,T}
"""
abstract type CompiledCircuit end

# Make CPUCircuit and StaticCircuit subtypes conceptually
# (We use duck typing rather than actual inheritance for simplicity)

"""
    system_size(circuit) -> Int

Return the system size (number of unknowns).
"""
system_size(c::CPUCircuit) = c.n
system_size(::StaticCircuit{N,T,L}) where {N,T,L} = N

#==============================================================================#
# Time-Dependent Source Support for StaticCircuit
#==============================================================================#

"""
    StaticCircuitTD{N,T,L,B,P,S}

StaticCircuit with time-dependent sources.
Stores the builder function and parameters to allow rebuilding b(t).

# Fields
- `base::StaticCircuit{N,T,L}`: Base circuit (G and C are constant)
- `builder::B`: Circuit builder function
- `params`: Circuit parameters
- `spec`: Base specification (mode, temp)
"""
struct StaticCircuitTD{N,T,L,B,P,S}
    base::StaticCircuit{N,T,L}
    builder::B
    params::P
    spec::S
end

"""
    compile_static_circuit_td(builder, params, spec, ::Val{N}, T=Float64) -> StaticCircuitTD

Compile a circuit with time-dependent sources into a StaticCircuitTD.
"""
function compile_static_circuit_td(builder::B, params::P, spec::S,
                                    ::Val{N}, ::Type{T}=Float64) where {B,P,S,N,T}
    base = compile_static_circuit(builder, params, spec, Val(N), T)
    L = N * N
    return StaticCircuitTD{N,T,L,B,P,S}(base, builder, params, spec)
end

"""
    static_residual(du, u, circuit::StaticCircuitTD{N,T}, t) -> SVector{N,T}

Residual for time-dependent circuits. Rebuilds b(t) each call.
Note: This allocates due to builder call - use only when sources vary with time.
"""
function static_residual(du::SVector{N,T}, u::SVector{N,T},
                         circuit::StaticCircuitTD{N,T,L}, t::Real) where {N,T,L}
    # Rebuild context at current time to get b(t)
    spec_t = MNASpec(temp=circuit.spec.temp, mode=:tran, time=Float64(t))
    ctx = circuit.builder(circuit.params, spec_t; x=Vector(u))
    b_vec = get_rhs_typed(ctx, T)
    b = SVector{N,T}(b_vec)

    # F = C*du + G*u - b(t)
    return circuit.base.C * du + circuit.base.G * u - b
end

#==============================================================================#
# Conversion Utilities
#==============================================================================#

"""
    to_static(circuit::CPUCircuit{T}, ::Val{N}) -> StaticCircuit{N,T}

Convert a CPUCircuit to StaticCircuit for ensemble GPU use.
"""
function to_static(circuit::CPUCircuit{T}, ::Val{N}) where {T,N}
    @assert circuit.n == N "Circuit size ($(circuit.n)) doesn't match Val($N)"

    G = SMatrix{N,N,T}(Matrix(circuit.G))
    C = SMatrix{N,N,T}(Matrix(circuit.C))
    b = SVector{N,T}(circuit.b)

    names = ntuple(i -> i <= length(circuit.node_names) ? circuit.node_names[i] :
                        Symbol("I_", circuit.current_names[i - length(circuit.node_names)]), N)

    return StaticCircuit{N,T,N*N}(G, C, b, names)
end

#==============================================================================#
# DC Solve for StaticCircuit
#==============================================================================#

"""
    solve_dc_static(circuit::StaticCircuit{N,T}) -> SVector{N,T}

Solve DC operating point for a static circuit.
Returns x such that G*x = b (assumes C*dx/dt = 0 at DC).
"""
function solve_dc_static(circuit::StaticCircuit{N,T,L}) where {N,T,L}
    # At DC, du = 0, so G*u = b
    return circuit.G \ circuit.b
end

"""
    solve_dc_cpu(circuit::CPUCircuit{T}) -> Vector{T}

Solve DC operating point for a CPU circuit.
"""
function solve_dc_cpu(circuit::CPUCircuit{T}) where T
    return circuit.G \ circuit.b
end
