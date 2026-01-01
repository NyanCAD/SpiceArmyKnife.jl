using Accessors
using AxisKeys
using OrdinaryDiffEq, SciMLBase, Sundials
using Base.Iterators

# Phase 0: Use stubs instead of DAECompiler
@static if CedarSim.USE_DAECOMPILER
    using DAECompiler: arg1_from_sys
else
    using ..DAECompilerStubs: arg1_from_sys
end

export alter, dc!, tran!, Sweep, CircuitSweep, ProductSweep, TandemSweep, SerialSweep, sweepvars, split_axes, sweepify

# This `alter()` is to make it easy to apply directly to a struct or named tuple with the optics.
function alter(x::T, params) where {T}
    lens(selector::Union{PropertyLens,ComposedFunction}) = selector
    lens(selector::Symbol) = Accessors.opticcompose(PropertyLens.(Symbol.(split(string(selector), ".")))...)

    if isa(params, NamedTuple)
        params = pairs(params)
    end

    for (selector, value) in params
        # If `value` is `nothing`, ignore it.  This is our sentinel for "use default value".
        if value === nothing
            continue
        end

        l = lens(selector)
        # Auto-convert other numeric types to Float
        # Ideally, we'd do something more clever at sweep creation time,
        # to automatically convert e.g. `1:10` in a sweep to `Float64.(1:10)`
        # but we'd only want to do that if we knew the type of the thing being swept.
        # Without the machinery to go from parameter overrides to types (within a
        # function circuit or a struct) we can't safely do that.  In practice,
        # at the moment of this writing, we're only ever sweeping over floats,
        # so I expect this `if` statement to get triggered essentially always.
        if isa(value, Number) && isa(l(x), Float64)
            value = Float64(value)
        end
        x = Accessors.set(x, l, value)::T
    end
    return x::T
end

# Special flattening iterator that takes combined Sweeps
# and flattens them to a proper tuple: ((a, b), c) -> (a, b, c)
"""
    SweepFlattener

Flattening iterator that takes combined iterators and flattens them to a proper
tuple.  Allows nesting of sweep types while still providing a flat result.
If the iterators would result in `((a, b), c)` instead returns `(a, b, c)`.

This type is automatically used by [`ProductSweep`](@ref), [`TandemSweep`](@ref)
and [`SerialSweep`](@ref), it is not intended for use by end-users.
"""
struct SweepFlattener{T}
    iterator::T
end
Base.length(sf::SweepFlattener) = Base.length(sf.iterator)
Base.size(sf::SweepFlattener) = Base.size(sf.iterator)
Base.size(sf::SweepFlattener, d::Integer) = get(size(sf), d, 1)
Base.IteratorSize(sf::SweepFlattener) = Base.IteratorSize(sf.iterator)
sweepvars(sf::SweepFlattener) = sweepvars(sf.iterator)
sweep_example(sf::SweepFlattener) = isempty(sf) ? () : first(sf)

function Base.show(io::IO, sf::SweepFlattener)
    indent = get(io, :indent, 0)
    indent_str = " "^indent
    if isa(sf.iterator, Base.Iterators.ProductIterator)
        print(io, indent_str, "Product sweep of:")
    elseif isa(sf.iterator, Base.Iterators.Zip)
        print(io, indent_str, "Tandem sweep of:")
    elseif isa(sf.iterator, SerialSweep)
        print(io, indent_str, "Serial sweep of:")
    end
    for it in (isa(sf.iterator, Base.Iterators.Zip) ? sf.iterator.is : sf.iterator.iterators)
        println(io)
        show(IOContext(io, :indent => indent + 2), it)
    end
end


"""
    split_axes(sweep, axes)

Advanced users of sweeps may want to extract large `ProductSweep` runs into
"inner" and "outer" sweeps; this allows e.g. running an external simulator
within a loop over the "outer" sweeps, while allowing the external simulator
to run a limited subset of the sweeps internally.  Note that this only works
on `ProductSweep` objects for now, and that the inverse of this operation is
`ProductSweep(outer, inner)`.  `axes` should be some kind of container that
yields `Symbol`s, such as a `Vector{Symbol}`.

Example:

    outer_sweep, inner_sweep = split_axes(ProductSweep(A=1:10, B=1:8, C=1:6, D=1:4), [:A, :C])
    @test sweepvars(outer_sweep) == Set([:B, :D])
    @test length(outer_sweep) == 32
    @test sweepvars(inner_sweep) == Set([:A, :C])
    @test length(inner_sweep) == 60
"""
function split_axes(sf::SweepFlattener, axes)
    if !isa(sf.iterator, Base.Iterators.ProductIterator)
        throw(ArgumentError("split_axes only works with ProductSweep objects!"))
    end

    # Find the iterator that corresponds to the given axis, or error out
    function find_axis_iterator_idx(axis_name, iterators)
        for (idx, it) in enumerate(iterators)
            if it.selector == axis_name
                return idx
            end
        end
        throw(ArgumentError("Unable to find product axis matching '$(axis_name)'"))
    end

    # Get indices for each axis we're extracting
    inner_idxs = Int[]
    for axis in axes
        push!(inner_idxs, find_axis_iterator_idx(axis, sf.iterator.iterators))
    end

    # Slice them out:
    inner_iterators = sf.iterator.iterators[inner_idxs]
    outer_iterators = sf.iterator.iterators[(1:length(sf.iterator.iterators)) .∉ (inner_idxs,)]

    # Create new ProductSweeps with new selections of iterators
    inner_ps = SweepFlattener(Base.Iterators.product(inner_iterators...))
    outer_ps = SweepFlattener(Base.Iterators.product(outer_iterators...))

    return outer_ps, inner_ps
end


expand(x::Tuple{Symbol,Any}) = tuple(x)
function expand(xs)
    rets = Tuple{Symbol,Any}[]
    for v in xs
        append!(rets, expand(v))
    end
    return tuple(sort(rets, by = ((k, v),) -> k)...)
end

function Base.iterate(sf::SweepFlattener{T}, state...) where {T}
    next = iterate(sf.iterator, state...)
    if next === nothing
        return nothing
    end
    return (expand(next[1]), next[2])
end


"""
    Sweep

Provides a 1-dimensional "sweep" over a given circuit property.  To denote the
the property, you can specify either a Symbol, or an `@optic` selector from the
`Accessors` package.  The swept values can be any kind of iterable, but note
that only `AbstractRange`s will not be immediately expanded.

To combine multiple Sweep objects together, see the combining helper types:

- [`ProductSweep`](@ref)
- [`TandemSweep`](@ref)
- [`SerialSweep`](@ref)

Examples:

    # Argument usage
    Sweep(:R1, 0.1:0.1:1.0)
    Sweep(:R1 => 0.1:0.1:1.0)

    # kwargs usage
    Sweep(R1 = 0.1:0.1:1.0)

    # Nested properties can be accessed via var-strings:
    Sweep(var"a.b" = 1:10)
"""
struct Sweep{T}
    selector::Symbol
    values::Union{AbstractRange{T},Vector{T}}
end

# If someone provided some weird iterable (neither an `AbstractRange`
# nor a `Vector`), just `collect` it immediately.
Sweep(selector::Symbol, values) = Sweep(selector, collect(values)[:])

# Allow using a `Pair`
Sweep((selector, values)::Pair) = Sweep(selector, values)

# kwargs usage
function Sweep(;kwargs...)
    # Only support a single mapping at a time
    if length(kwargs) != 1
        throw(ArgumentError("`Sweep` takes a single variable at a time!"))
    end
    (selector, values) = only(pairs(kwargs))
    return Sweep(selector, values)
end

function Base.:(==)(a::Sweep, b::Sweep)
    return a.selector == b.selector &&
           a.values == b.values
end

function Base.show(io::IO, s::Sweep)
    indent = get(io, :indent, 0)
    indent_str = " " ^ indent
    prefix = indent > 0 ? "- " : "Sweep of "
    if length(s.values) > 1
        minval = minimum(s.values)
        maxval = maximum(s.values)
        print(io, indent_str, prefix, "$(s.selector) with $(length(s.values)) values over [$(minval) .. $(maxval)]")
    else
        print(io, indent_str, prefix, "$(s.selector) set to $(s.values[1])")
    end
end

# Nothing to flatten if it's just a single Sweep
SweepFlattener(s::Sweep) = s


# These are the kinds of things that our Sweep API generates;
# if someone passes a `ProductSweep` into another `ProductSweep`,
# we don't want to try and make a `Sweep` off of that.
const SweepLike = Union{Sweep,SweepFlattener}

# If we're already a Sweep... don't do anything
Sweep(x::SweepLike) = x

# Allow iteration over this `Sweep`, returning `(v.selector, element)`
function Base.iterate(v::Sweep, state...)
    sub_it = Base.iterate(v.values, state...)
    if sub_it === nothing
        return sub_it
    end
    return (((v.selector, sub_it[1]),), sub_it[2])
end
Base.length(v::Sweep) = Base.length(v.values)
Base.size(v::Sweep) = size(v.values)
Base.size(v::Sweep, d::Integer) = get(size(v), d, 1)
Base.IteratorSize(v::Sweep) = Base.IteratorSize(v.values)
sweep_example(v::Sweep) = first(v)

"""
    sweepvars(sweep)

Return the full `Set` of variables (in the form of symbols) that can be set when
iterating over the given sweep object.
"""
sweepvars(v::Sweep) = Set([v.selector])
sweepvars(v::Tuple{}) = Set{Symbol}()
sweepvars(vs...) = !isempty(vs) ? union(sweepvars.(vs)...) : Set{Symbol}()

"""
    ProductSweep(args...)

Takes the given `Sweep` specifications and produces the cartesian product of
the individual inputs:

    ProductSweep(R1 = 1:4, R2 = 1:4, R3 = 1:4)

    ProductSweep(:R1 => 1:4, :R2 => 1:4)
"""
function ProductSweep(args...; kwargs...)
    # Special-case; if we're given only one thing, just turn it into a `Sweep`
    if length(args) + length(kwargs) == 1
        return Sweep(collect(args)...; collect(kwargs)...)
    end
    return SweepFlattener(product(Sweep.(collect(args))..., Sweep.(collect(kwargs))...))
end
sweepvars(pi::Base.Iterators.ProductIterator) = sweepvars(pi.iterators...)

"""
    TandemSweep(args...)

Takes the given `Sweep` specifications and produces an iterator that emits
each input in tandem.  Note that all inputs must have the same length!

    TandemSweep(R1 = 1:4, R2 = 1:4)
"""
function TandemSweep(vs::Sweep...)
    # Special-case; if we're given only one thing, just turn it into a `Sweep`
    if length(vs) == 1
        return vs[1]
    end
    lengths = length.(vs)
    if !all(lengths .== lengths[1])
        throw(ArgumentError("TandemSweep requires all sweeps be of the same length!"))
    end
    return zip(vs...)
end
TandemSweep(args...; kwargs...) = SweepFlattener(TandemSweep(Sweep.(collect(args))..., Sweep.(collect(kwargs))...))
sweepvars(ts::Base.Iterators.Zip) = sweepvars(ts.is...)

"""
    SerialSweep(args...)

Takes the given `Sweep` specifications and produces an iterator that emits
each input in serial.

    SerialSweep(R1 = 1:4, R2 = 1:4)
"""
struct SerialSweep
    iterators::Vector
    # We keep track of our full set of variables so that we can always emit (v1 = nothing, v2 = x, v3 = nothing, v4 = nothing)
    vars::Set{Symbol}

    function SerialSweep(args...; kwargs...)
        iterators = [Sweep.(collect(args))..., Sweep.(collect(kwargs))...]
        # Special-case; if we're given only one thing, just turn it into a `Sweep`
        if length(iterators) == 1
            return iterators[1]
        end
        vars = sweepvars(iterators...)
        return SweepFlattener(new(iterators, vars))
    end
end

function Base.iterate(ss::SerialSweep, state = (1,))
    it_idx = state[1]
    if it_idx > length(ss.iterators)
        return nothing
    end
    next = iterate(ss.iterators[it_idx], Base.tail(state)...)
    while next === nothing
        it_idx += 1
        if it_idx > length(ss.iterators)
            return nothing
        end
        next = iterate(ss.iterators[it_idx])
    end
    mappings = Dict{Symbol,Any}(v => nothing for v in ss.vars)
    for (var, val) in expand(next[1])
        mappings[var] = val
    end
    return (tuple(((var, val) for (var, val) in mappings)...), (it_idx, next[2]))
end
Base.length(ss::SerialSweep) = sum(length(it) for it in ss.iterators; init=0)
Base.size(ss::SerialSweep) = (length(ss),)
Base.size(ss::SerialSweep, d::Integer) = get(size(ss), d, 1)
sweepvars(ss::SerialSweep) = sweepvars(ss.iterators...)

"""
    sweepify(obj)

Converts convenient shorthands for sweep specifications from julia syntax to
actual [`Sweep`](@ref) objects.  `NamedTuple` objects are turned into
[`SerialSweep`](@ref)'s, vectors are turned into [`SerialSweep`](@ref)'s.

    sweepify([(r1 = 1:10, r2 = 1:10), (r3 = 1:10, r4=1:10)])
"""
sweepify(x::AbstractArray) = SerialSweep(sweepify.(x)...)
function sweepify(x::NamedTuple)
    ProductSweep(;(k => v for (k, v) in pairs(x))...)
end
sweepify(x::SweepLike) = x
sweepify(x) = Sweep(x)


using .MNA: MNACircuit, MNASystem, assemble!, solve_dc, make_ode_problem

"""
    CircuitSweep

Provides a multi-dimensional sweep over sets of variables defined within a
circuit.  When iterated over, returns the altered circuit object (MNACircuit).
Parameter specification can be done with simple `Symbol`'s, or with more complex
`@optic` values from the `Accessors` package.

Examples:

    # Create a sweep over circuit parameters:
    cs = CircuitSweep(build_circuit, ProductSweep(R1 = 0.1:0.1:1.0); R2 = 100.0)

    # Iterate over the sweep:
    for circuit in cs
        sol = dc!(circuit)
        ...
    end

    # Or use dc!/tran! directly on the sweep:
    sols = dc!(cs)
"""
struct CircuitSweep{T<:Function,C<:MNA.MNACircuit}
    builder::T       # Builder function that takes (params, spec) and returns MNAContext
    circuit::C       # Base circuit for alter()
    iterator::SweepLike

    function CircuitSweep(builder::T, circuit::C, iterator::SweepLike) where {T<:Function,C<:MNA.MNACircuit}
        if !applicable(iterate, iterator)
            throw(ArgumentError("Must give some kind of iterator!"))
        end
        return new{T,C}(builder, circuit, iterator)
    end
end

# Convenience constructor: create an MNACircuit from a builder function with default params
function CircuitSweep(builder::Function, iterator::SweepLike; spec=MNA.MNASpec(), default_params...)
    first_params = sweep_example(iterator)
    merged_params = merge(NamedTuple(default_params), NamedTuple(first_params))
    circuit = MNA.MNACircuit(builder; spec=spec, merged_params...)
    return CircuitSweep(builder, circuit, iterator)
end

Base.length(cs::CircuitSweep) = length(cs.iterator)
Base.size(cs::CircuitSweep) = size(cs.iterator)
Base.size(cs::CircuitSweep, d::Integer) = get(size(cs), d, 1)
Base.IteratorSize(cs::CircuitSweep) = Base.IteratorSize(cs.iterator)
sweepvars(cs::CircuitSweep) = sweepvars(cs.iterator)

# Iteration returns MNACircuit objects via alter()
function Base.iterate(cs::CircuitSweep, state...)
    next = iterate(cs.iterator, state...)
    if next === nothing
        return nothing
    end
    new_circuit = MNA.alter(cs.circuit; next[1]...)
    return (new_circuit, next[2])
end

#==============================================================================#
# DC Analysis
#==============================================================================#

"""
    dc!(circuit::MNACircuit)

DC operating point analysis for MNACircuit.
Returns a `DCSolution` with voltage/current accessors.

# Example
```julia
circuit = MNACircuit(build_divider, (R1=1k, R2=1k), MNASpec())
sol = dc!(circuit)
voltage(sol, :out)  # Voltage at output node
```
"""
function dc!(circuit::MNA.MNACircuit)
    return MNA.solve_dc(circuit)
end

"""
    dc!(cs::CircuitSweep)

DC operating point analysis for a circuit sweep.
Returns a vector of `DCSolution` objects.
"""
function dc!(cs::CircuitSweep; kwargs...)
    return [dc!(circuit; kwargs...) for circuit in cs]
end

#==============================================================================#
# Transient Analysis
#==============================================================================#

"""
    tran!(circuit::MNACircuit, tspan; solver=IDA(), abstol=1e-10, reltol=1e-8, kwargs...)

Transient analysis for MNACircuit.

Automatically dispatches based on solver type:
- DAE solvers (IDA, DFBDF): Uses DAEProblem with full nonlinear handling
- ODE solvers (Rodas5P, etc.): Uses ODEProblem with mass matrix

For nonlinear circuits with voltage-dependent capacitors (MOSFETs, junction
capacitors), use a DAE solver. The circuit matrices are rebuilt at each Newton
iteration, ensuring correct handling of nonlinear devices.

# Arguments
- `circuit`: The MNA circuit
- `tspan`: Time span for simulation `(t0, tf)`
- `solver`: Solver algorithm (default: IDA with tuned parameters for circuits)
- `abstol`, `reltol`: Solver tolerances
- `explicit_jacobian`: Use explicit Jacobian (default: true for performance)

# Default IDA Configuration
The default IDA solver is configured for circuit simulation with:
- `max_error_test_failures=20`: More retries for difficult points (e.g., t=0 with
  time-dependent sources). The standard default of 7 is often too low.
- `max_nonlinear_iters=10`: More Newton iterations for nonlinear devices.
- For circuits with time-dependent sources (SIN, PWL), use `explicit_jacobian=false`.

# Example
```julia
circuit = MNACircuit(build_inverter; Vdd=1.8, W=1e-6, L=100e-9)
sol = tran!(circuit, (0.0, 1e-6))           # Uses IDA (default)
sol = tran!(circuit, (0.0, 1e-6); solver=Rodas5P())  # Uses ODEProblem
sol(1e-7)  # Get state at t=0.1μs
```
"""
function tran!(circuit::MNA.MNACircuit, tspan::Tuple{<:Real,<:Real};
               solver=nothing, abstol=1e-10, reltol=1e-8, kwargs...)
    # Default to IDA (DAE solver) with tuned parameters for circuit simulation.
    # Key settings:
    # - max_error_test_failures=20: Allows more retries at difficult points (like t=0 with
    #   time-dependent sources). Default of 7 is often too low for circuits.
    # - max_nonlinear_iters=10: More Newton iterations for nonlinear devices
    if solver === nothing
        solver = Sundials.IDA(max_error_test_failures=20, max_nonlinear_iters=10)
    end

    # Dispatch based on solver type
    return _tran_dispatch(circuit, tspan, solver; abstol=abstol, reltol=reltol, kwargs...)
end

# DAE solver dispatch (IDA, DFBDF, etc.)
# Note: explicit_jacobian defaults to true for performance. Set to false for circuits
# with time-dependent sources (SIN, PWL, etc.) if you encounter initialization issues.
function _tran_dispatch(circuit::MNA.MNACircuit, tspan::Tuple{<:Real,<:Real},
                        solver::SciMLBase.AbstractDAEAlgorithm;
                        abstol=1e-10, reltol=1e-8, explicit_jacobian=true,
                        initializealg=nothing, kwargs...)
    prob = SciMLBase.DAEProblem(circuit, tspan; explicit_jacobian=explicit_jacobian)
    # Use initializealg if provided, otherwise let the solver use its default
    if initializealg !== nothing
        return SciMLBase.solve(prob, solver; abstol=abstol, reltol=reltol, initializealg=initializealg, kwargs...)
    else
        return SciMLBase.solve(prob, solver; abstol=abstol, reltol=reltol, kwargs...)
    end
end

# ODE solver dispatch (Rodas5P, etc.)
# Uses NoInit() since we already have a valid DC operating point as u0.
# This avoids Julia 1.12 issues with DAE initialization for singular mass matrices.
function _tran_dispatch(circuit::MNA.MNACircuit, tspan::Tuple{<:Real,<:Real},
                        solver::SciMLBase.AbstractODEAlgorithm;
                        abstol=1e-10, reltol=1e-8, initializealg=OrdinaryDiffEq.NoInit(), kwargs...)
    prob = SciMLBase.ODEProblem(circuit, tspan)
    return SciMLBase.solve(prob, solver; abstol=abstol, reltol=reltol, initializealg=initializealg, kwargs...)
end

"""
    tran!(cs::CircuitSweep, tspan; kwargs...)

Transient analysis for a circuit sweep.
Returns a vector of solution objects.

# Arguments
- `cs`: The circuit sweep
- `tspan`: Time span for simulation `(t0, tf)`
- `solver`: Solver algorithm (default: IDA from Sundials.jl)
"""
function tran!(cs::CircuitSweep, tspan::Tuple{<:Real,<:Real}; kwargs...)
    return [tran!(circuit, tspan; kwargs...) for circuit in cs]
end

# Base case; a single parameter mapped on certain values:
function find_param_ranges(it::CedarSim.Sweep, param_ranges)
    if !haskey(param_ranges, it.selector)
        param_ranges[it.selector] = Tuple[]
    end
    push!(param_ranges[it.selector], (minimum(it.values), maximum(it.values), length(it.values)))
end
find_param_ranges(sf::CedarSim.SweepFlattener, param_ranges) = find_param_ranges(sf.iterator, param_ranges)
function find_param_ranges(it, param_ranges)
    children(it::Base.Iterators.Zip) = it.is
    children(it::Base.Iterators.ProductIterator) = it.iterators
    children(it::SerialSweep) = it.iterators
    for child in children(it)
        find_param_ranges(child, param_ranges)
    end
end

function collapse_ranges(ranges::Vector)
    total_min, total_max, total_len = ranges[1]
    for range in ranges[2:end]
        new_min, new_max, new_len = range
        total_min = min(new_min, total_min)
        total_max = max(new_max, total_max)
        total_len += new_len
    end
    return (total_min, total_max, total_len)
end

"""
    find_param_ranges(params)

Given a sweep specification, return a simplified view of the range of parameter
exploration and number of points within that range that will be explored.  This
loses all detail of product vs. serial vs. tandem, etc... but gives a rough idea
of the bounds along each dimension.
"""
function find_param_ranges(params)
    param_ranges = Dict{Symbol,Vector{Tuple}}()
    find_param_ranges(params, param_ranges)
    return Dict(name => collapse_ranges(ranges) for (name, ranges) in param_ranges)
end
