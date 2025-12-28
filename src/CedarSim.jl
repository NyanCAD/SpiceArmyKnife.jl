module CedarSim

# MNA Backend - DAECompiler has been removed
# AC analysis and some other features are kept for future porting

using DiffEqBase
using DynamicScope
using VectorPrisms

# MNA engine (the simulation backend)
include("mna/MNA.jl")
using .MNA
export MNA

# re-exports
export DAEProblem
export @dyn, @requires, @provides, @isckt_or
export solve

# Phase 4: MNA SPICE codegen exports
export make_mna_circuit, parse_spice_to_mna, solve_spice_mna


include("util.jl")
include("vasim.jl")

# Minimal runtime support for SpectreEnvironment and VA codegen
using Base.ScopedValues
Base.@kwdef struct SimSpec
    time::Float64=0.0
    temp::DefaultOr{Float64}=mkdefault(27.0)
    rng::Nothing=nothing
end
const spec = ScopedValue{SimSpec}(SimSpec())
const sim_mode = ScopedValue{Symbol}(:dcop)
abstract type CircuitElement end  # Base type for VA models

include("spectre_env.jl")

# Stubs for spectre.jl (DAECompiler codegen) - allows codegen without DAECompiler runtime
struct ParallelInstances end
struct SubCircuit{T}; ckt::T end
(X::SubCircuit)(args...; kwargs...) = error("SubCircuit requires DAECompiler")
macro subckt(args...) error("@subckt requires DAECompiler") end
abstract type AbstractSim{T} end
abstract type AbstractScope end
struct DebugScope <: AbstractScope
    parent::Union{DebugScope, Nothing}
    name::Symbol
    DebugScope() = new(nothing, :root)
    DebugScope(parent::DebugScope, name::Symbol) = new(parent, name)
end
const DScope = DebugScope
const debug_scope = ScopedValue{AbstractScope}(DScope())
observed!(args...) = nothing
struct ParamSim{T,S,P} <: AbstractSim{T}  # Stub type for alter()
    circuit::T
    mode::Symbol
    spec::S
    params::P
end
struct Named{T}  # Stub for DAECompiler codegen
    element::T
    name::Symbol
    Named(element::T, name::Union{String, Symbol}) where {T} = new{Core.Typeof(element)}(element, Symbol(name))
end
(n::Named)(args...; kwargs...) = error("Named circuit elements require DAECompiler")

include("spectre.jl")

# Phase 4: New SPC SPICE codegen (used by MNA backend)
include("spc/cache.jl")  # Must be before sema.jl (CedarParseCache)
include("spc/sema.jl")
include("spc/codegen.jl")
include("spc/interface.jl")
include("spc/query.jl")
include("spc/generated.jl")
include("va_env.jl")
include("sweeps.jl")
# AC/noise analysis not loaded - requires DAECompiler porting to MNA
# See ac.jl for reference implementation
include("ModelLoader.jl")
# netlist_utils.jl and circsummary.jl removed - DAECompiler-only

import .ModelLoader: load_VA_model
export load_VA_model

# Store the known-good julia version that we should be compiling against
_blessed_julia_version = begin
    julia_version_path = joinpath(@__DIR__, "../contrib/julia_build/julia_version.inc")
    Base.include_dependency(julia_version_path)
    strip(split(String(read(julia_version_path)), ":=")[2])
end

function check_version_match()
    # Only print this out if we're not precompiling, as it's annoying to see this
    # pop up for every extension precompile process.
    if _blessed_julia_version != Base.GIT_VERSION_INFO.commit && ccall(:jl_generating_output, Cint, ()) != 1
        @warn("""
        You are not running on the Cedar-blessed Julia version! (currently '$(_blessed_julia_version)')
        Try running './juliaup_cedar.sh', and remember to start julia with `julia +cedar`!
        """)
    end
end

function __init__()
    # Do this at `__init__()` time instead of precompile time because it's too easy to miss it during precompilation.
    check_version_match()
end

using PrecompileTools
using SpectreNetlistParser
@setup_workload let
    spice = """
    * my circuit
    v1 vcc 0 DC 5
    r1 vcc n1 1k
    l1 n1 n2 1m
    c1 n2 0 1u
    """
    @compile_workload @time begin
        # MNA-based SPICE parsing and codegen
        sa1 = VerilogAParser.parsefile(joinpath(@__DIR__, "../VerilogAParser.jl/test/inputs/resistor.va"))
        code1 = CedarSim.make_mna_module(sa1)
        sa3 = SpectreNetlistParser.parse(IOBuffer(spice); start_lang=:spice)
        code3 = CedarSim.make_mna_circuit(sa3)
    end
end

"
    @declare_MSLConnector(mtk_model, pin_ports...)

!!! note \"For this to be used ModelingToolkit must be loaded\"
    CedarSim itself only provides a stub-defination of this type.
    The full implementation is in the CedarSim-ModelingToolkit extension module.
    Which is automatically loaded if CedarSim and ModelingToolkit are both loaded.

Defined the functions needed to connect a MTK based model (defined using MSL `Pin`s) to Cedar.
As input provide the model (an `ODESystem`), and a list of pins defined using `ModelingToolkitStandardLibary.Electrical.Pin`s.
These pins can be as direct components of the model or subcomponents other components.
When you use this component as a subcircuit (as shown in the example) they be connected to CedarSim `AbstractNets` 
corresponding to the SPICE nodes, in the order you list them.


Note that the model does not have to be (and usually won't be) solvable in MTK -- it can be incomplete and unablanced.
The remaining variables coming from the rest of the circuit, e.g. as defined using SPICE.
The usual way to develop this would be to initially write the model in MTK using MSL,
then delete all the voltage/current sources and declare that the places they were connected are port pins usng this macro.



This returns a connector type for use with ModelingToolkit.

This means it is struct with a constructor that you can override the parameters to by keyword argument.
You can check what parameters are available by using `parameternames` on an instance of the type.
The struct will have call overriden (i.e. it will be a functor) to allow the connections CedarSim exposes to all be hooked up.

It is used for example as:
```julia
@mtkmodel Foo begin
    @parameters begin
        param=0.0
        ...
    end
    @components begin
        Pos = Pin()
        Neg = Pin()
        ...
    end
    ...
end

foo = foo(name=:foo1)
const FooConn = @declare_MSLConnector(foo, foo.Pos, foo.Neg)
circuit = sp\"\"\" ...
Xfoo 1 0 \$(FooConn(param = 42.0))
...
\"\"\"e
```
"
macro declare_MSLConnector(args...)
    error("ModelingToolkit must be loaded for this macro to be used")
end
export @declare_MSLConnector

end # module
