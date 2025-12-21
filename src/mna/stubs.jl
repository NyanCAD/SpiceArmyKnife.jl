# MNA Phase 0: Stubs for DAECompiler functions
# These allow the package to load without DAECompiler while parsing/codegen still works
# TODO: Replace with actual MNA implementation in later phases

module DAECompilerStubs

export variable, equation, equation!, observed!, singularity_root!, sim_time, GenScope
# Note: Scope is NOT exported - it conflicts with CedarSim.Scope in vasim.jl
# Access via DAECompilerStubs.Scope instead
export ddt, AbstractScope, IRODESystem, ScopeRef, epsilon, time_periodic_singularity!

# Nested module to match DAECompiler structure
module Intrinsics
    export _compute_new_nt_type

    # Helper for dynamic named tuple extension
    function _compute_new_nt_type(nt::NamedTuple{names}, s::Symbol) where {names}
        NamedTuple{(names..., s)}
    end
end
export get_transformed_sys, UnsupportedIRException
export default_parameters, DebugConfig, IRCodeRecords, SystemStructureRecords
export get_sys, batch_reconstruct, split_and_sort_syms
export compile_batched_reconstruct_derivatives, determine_num_tangents
export parameter_type, DAEReconstructedObserved, arg1_from_sys

# Placeholder types for DAECompiler primitives
# These are used as type annotations in simulate_ir.jl

abstract type AbstractScope end

# equation is both a type and callable - used as kcl!::equation in Net struct
struct equation
    scope::Union{AbstractScope, Nothing}
    equation(scope::Union{AbstractScope, Nothing}) = new(scope)
end
equation() = equation(nothing)
# Note: equation(scope::AbstractScope) dispatches to inner constructor
(eq::equation)(value) = nothing  # No-op for Phase 0 codegen testing

struct Scope <: AbstractScope
    parent::Union{Nothing, Scope}
    name::Symbol
    Scope() = new(nothing, :root)
    Scope(parent::Scope, name::Symbol) = new(parent, name)
    Scope(parent::AbstractScope, name::Symbol) = new(nothing, name)
end
(s::Scope)(name::Symbol) = Scope(s, name)

struct GenScope <: AbstractScope
    parent::AbstractScope
    name::Symbol
    # Inner constructors
    GenScope(parent::AbstractScope, name::Symbol) = new(parent, name)
    GenScope(parent::AbstractScope, name::String) = new(parent, Symbol(name))
end
(s::GenScope)(name::Symbol) = GenScope(s, name)

struct ScopeRef
    scope::AbstractScope
end

struct IRODESystem
    # Placeholder
end

struct UnsupportedIRException <: Exception
    msg::String
end

struct DebugConfig
    ir_levels::Any
    ss_levels::Any
    verify_ir_levels::Bool
    ir_log::String
end
DebugConfig(; ir_levels=nothing, ss_levels=nothing, verify_ir_levels=false, ir_log="") =
    DebugConfig(ir_levels, ss_levels, verify_ir_levels, ir_log)

struct IRCodeRecords end
struct SystemStructureRecords end
struct DAEReconstructedObserved end

# Phase 0: Placeholder functions that return stub values for codegen testing
# Actual simulation will not work, but code can be generated and parsed

# Placeholder variable type for codegen testing
struct StubVariable
    scope::Union{AbstractScope, Nothing}
end

function variable(scope::AbstractScope)
    StubVariable(scope)
end
function variable(args...)
    StubVariable(nothing)
end

# equation() constructor is now defined with the struct above

function equation!(val, scope)
    # No-op for Phase 0
    nothing
end
function equation!(args...)
    nothing
end

function observed!(args...)
    # No-op for Phase 0
    nothing
end

function singularity_root!(args...)
    # No-op for Phase 0
    nothing
end

function sim_time()
    # Return 0.0 for codegen testing
    0.0
end

function ddt(x)
    # Return the input unchanged for Phase 0 (no differentiation)
    x
end

function get_transformed_sys(args...)
    error("DAECompiler not available: get_transformed_sys() requires MNA implementation")
end

function get_sys(args...)
    error("DAECompiler not available: get_sys() requires MNA implementation")
end

function batch_reconstruct(args...)
    error("DAECompiler not available: batch_reconstruct() requires MNA implementation")
end

function split_and_sort_syms(args...)
    error("DAECompiler not available: split_and_sort_syms() requires MNA implementation")
end

function compile_batched_reconstruct_derivatives(args...)
    error("DAECompiler not available: compile_batched_reconstruct_derivatives() requires MNA implementation")
end

function determine_num_tangents(args...)
    error("DAECompiler not available: determine_num_tangents() requires MNA implementation")
end

function parameter_type(args...)
    error("DAECompiler not available: parameter_type() requires MNA implementation")
end

function default_parameters(args...)
    error("DAECompiler not available: default_parameters() requires MNA implementation")
end

function arg1_from_sys(args...)
    error("DAECompiler not available: arg1_from_sys() requires MNA implementation")
end

function epsilon(args...)
    error("DAECompiler not available: epsilon() requires MNA implementation (Phase 1+)")
end

function time_periodic_singularity!(args...)
    error("DAECompiler not available: time_periodic_singularity!() requires MNA implementation (Phase 1+)")
end

end # module DAECompilerStubs
