import VerilogAParser

# Use CMCModels package location for BSIM-CMG models
const cmc_models_dir = joinpath(dirname(dirname(@__DIR__)), "models", "CMCModels.jl", "va")

# Only test models available locally
# Full CMC test suite requires the CMC package
models = [
    "bsimcmg.va",
]

@testset "CMC Models (local)" begin
    for model in models
        model_path = joinpath(cmc_models_dir, model)
        if isfile(model_path)
            @testset "$model" begin
                local va = VerilogAParser.parsefile(model_path)
                # CMC models shouldn't have errors since they presumably work in Cadence, and others.
                buf = IOBuffer()
                out = IOContext(buf, :color=>true, :displaysize => (80, 240))
                VerilogAParser.VerilogACSTParser.visit_errors(va; io=out)

                out = String(take!(buf))
                @test isempty(out)
            end
        else
            @warn "Skipping $model (not found at $model_path)"
        end
    end
end
