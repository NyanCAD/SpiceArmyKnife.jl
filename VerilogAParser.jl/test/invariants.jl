import VerilogAParser
using AbstractTrees
using VerilogAParser.VerilogACSTParser: virtrange

# Use CMCModels package location for BSIM-CMG models
const cmc_models_dir = joinpath(dirname(dirname(@__DIR__)), "models", "CMCModels.jl", "va")

macro_va = VerilogAParser.parse("""
    `define f(arg) arg
    `define g(arg) `f(arg)

    module test(x);
            analog begin
                    `g(VMAX_s) = 1;
            end
    endmodule
    """)

# Test with local bsimcmg model
test_vas = [macro_va]
bsimcmg_path = joinpath(cmc_models_dir, "bsimcmg.va")
if isfile(bsimcmg_path)
    push!(test_vas, VerilogAParser.parsefile(bsimcmg_path))
end

for va in test_vas
    ls = collect(Leaves(VerilogAParser.VerilogACSTParser.ChunkTree(va.ps)))
    @test all(1:(length(ls)-1)) do i
        first(virtrange(ls[i+1])) == last(virtrange(ls[i]))+1
    end
end
