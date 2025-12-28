#==============================================================================#
# Spectre/SPICE Parsing Utilities
#
# This file contains utility functions for parsing Spectre and SPICE netlists.
# The codegen (DAECompiler-based) has been removed - see spc/codegen.jl for
# MNA-based codegen.
#==============================================================================#

using SpectreNetlistParser
using SpectreNetlistParser: SpectreNetlistCSTParser, SPICENetlistParser
using .SPICENetlistParser: SPICENetlistCSTParser
using .SpectreNetlistCSTParser: SpectreNetlistSource
using .SPICENetlistCSTParser: SPICENetlistSource

const SNode = SpectreNetlistCSTParser.Node
const SC = SpectreNetlistCSTParser
const SP = SPICENetlistCSTParser

#==============================================================================#
# String/Symbol Utilities
#
# Case-insensitive string/symbol conversion used by SPICE parsing.
#==============================================================================#

LString(s::SNode{<:SP.Terminal}) = lowercase(String(s))
LString(s::SNode{<:SP.AbstractASTNode}) = lowercase(String(s))
LString(s::SNode{<:SC.Terminal}) = String(s)
LString(s::SNode{<:SC.AbstractASTNode}) = String(s)
LString(s::AbstractString) = lowercase(s)
LString(s::Symbol) = lowercase(String(s))
LSymbol(s) = Symbol(LString(s))

#==============================================================================#
# Parameter Utilities
#==============================================================================#

"""
    hasparam(params, name)

Check if a parameter with the given name exists in a parameter list.
Used by MNA codegen to check for specific parameters like 'l', 'r', etc.
"""
function hasparam(params, name)
    for p in params
        if LString(p.name) == name
            return true
        end
    end
    return false
end

#==============================================================================#
# SpcFile for Module Loading
#
# Allows loading SPICE/Spectre files as Julia modules.
#==============================================================================#

struct SpcFile
    file::String
    raw::Bool
end
SpcFile(file::String) = SpcFile(file, false)
Base.String(vaf::SpcFile) = vaf.file
Base.abspath(file::SpcFile) = SpcFile(Base.abspath(file.file))
Base.isfile(file::SpcFile) = Base.isfile(file.file)
Base.isabspath(file::SpcFile) = Base.isabspath(file.file)
Base.findfirst(str::SpcFile, file::SpcFile) = Base.findfirst(str, file.file)
Base.joinpath(str::SpcFile, file::SpcFile) = SpcFile(Base.joinpath(str, file.file))
Base.normpath(file::SpcFile) = SpcFile(Base.normpath(file.file))
export SpcFile

#==============================================================================#
# Error Types
#==============================================================================#

struct SpectreParseError
    sa
end

Base.show(io::IO, sap::SpectreParseError) = SpectreNetlistCSTParser.visit_errors(sap.sa; io)
