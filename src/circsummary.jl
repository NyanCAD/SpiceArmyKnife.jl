abstract type CircRef; end

# Various references to SPICE-namespaced things
struct InstanceRef
    ref::SNode
end
struct ParamRef
    ref::SNode
end
struct ModelRef
    ref::SNode
end
struct NetRef
    refs::Vector{SNode}
end

struct AmbiguousRef
    instance::Union{InstanceRef, Nothing}
    param::Union{ParamRef, Nothing}
    model::Union{ModelRef, Nothing}
    net::Union{NetRef, Nothing}
end

using SymbolicIndexingInterface
SymbolicIndexingInterface.symbolic_type(::NetRef) = ScalarSymbolic()
SymbolicIndexingInterface.symbolic_type(::Type{<:NetRef}) = ScalarSymbolic()
