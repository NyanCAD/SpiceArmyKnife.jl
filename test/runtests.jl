using Test
using Random

# avoid random CI failures due to randomness
# fixed seed chosen by fair dice roll
Random.seed!(10)

# Phase 0: Check if we're running with full dependencies or minimal (parsing only)
const PHASE0_MINIMAL = !isdefined(Main, :DAECompiler) &&
                       !@isdefined(BSIM4) &&
                       !@isdefined(Sky130PDK)

if PHASE0_MINIMAL
    @info "Running Phase 0/1 tests (parsing/codegen + MNA core)"

    # Load CedarSim for MNA tests
    using CedarSim

    # Tests that work with parsing only (no simulation required)
    @testset "Phase 0: Parsing Tests" begin
        @testset "spectre_expr.jl" include("spectre_expr.jl")
        @testset "sweep.jl" include("sweep.jl")
    end

    # Phase 1: MNA core tests (standalone, no DAECompiler required)
    @testset "Phase 1: MNA Core" begin
        @testset "mna/core.jl" include("mna/core.jl")
    end

    # Phase 5: VA integration tests (s-dual contribution stamping)
    @testset "Phase 5: MNA VA Integration" begin
        @testset "mna/va.jl" include("mna/va.jl")
        @testset "ddx.jl" include("ddx.jl")
        @testset "varegress.jl" include("varegress.jl")
    end

    # Phase 6: Multi-terminal VA devices and MOSFET tests
    @testset "Phase 6: MNA VA Multi-Terminal" begin
        @testset "mna/va_mosfet.jl" include("mna/va_mosfet.jl")
        @testset "mna/vadistiller.jl" include("mna/vadistiller.jl")
    end

    # Phase 4: Basic tests using MNA backend
    @testset "Phase 4: MNA Basic Tests" begin
        @testset "basic.jl" include("basic.jl")
        @testset "transients.jl" include("transients.jl")
        @testset "params.jl" include("params.jl")
    end
else
    @info "Running full test suite"

    @testset "basic.jl" include("basic.jl")
    @testset "transients.jl" include("transients.jl")
    @testset "compilation.jl" include("compilation.jl")
    @testset "params.jl" include("params.jl")
    @testset "ddx.jl" include("ddx.jl")
    @testset "alias.jl" include("alias.jl")
    @testset "varegress.jl" include("varegress.jl")
    @testset "compiler_sanity.jl" include("compiler_sanity.jl")
    @testset "binning/bins.jl" include("binning/bins.jl")
    @testset "bsimcmg/demo_bsimcmg.jl" include("bsimcmg/demo_bsimcmg.jl")
    #@testset "bsimcmg/bsimcmg_spectre.jl" include("bsimcmg/bsimcmg_spectre.jl")
    @testset "bsimcmg/inverter.jl" include("bsimcmg/inverter.jl")
    @testset "ac.jl" include("ac.jl")
    @testset "sky130" include("sky130/parse_unified.jl")
    @testset "spectre_expr.jl" include("spectre_expr.jl")
    @testset "sweep.jl" include("sweep.jl")
    @testset "inverter.jl" include("inverter.jl")
    @testset "gf180_dff.jl" include("gf180_dff.jl")
    @testset "sensitivity.jl" include("sensitivity.jl")
    @testset "inverter_noise.jl" include("inverter_noise.jl")
    @testset "MTK_extension.jl" include("MTK_extension.jl")
end
