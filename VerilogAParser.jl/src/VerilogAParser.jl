module VerilogAParser
include("tokenize/VerilogATokenize.jl")
include("parse/VerilogACSTParser.jl")
using .VerilogACSTParser: parse, parsefile, EXPR, Node

using SnoopPrecompile
@precompile_all_calls begin
    # Use BSIM-CMG from CMCModels package location
    parsefile(joinpath(@__DIR__, "../../models/CMCModels.jl/va/bsimcmg.va"))
end

end
