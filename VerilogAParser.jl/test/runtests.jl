using VerilogAParser
using VerilogAParser: EXPR, Node
using VerilogAParser.VerilogACSTParser: VerilogSource
using Test

var = """
    module BasicVAResistor(p, n); // N.B.: Whitespace at the beginning of this line is part of the test

inout p, n;
electrical p, n;
parameter real R=1 exclude 0;

analog begin
    I(p,n) <+ V(p,n)/R;
end
endmodule
"""

# For now just test that this doesn't error
let res = VerilogAParser.parse(var)
    @test isa(res, Node{VerilogSource})
    @test UInt32(res.startof) == 0
end

include("sv_tests.jl")

# Phase 0: Check if CMC is available (requires CedarEDA registry)
const HAS_CMC = try
    @eval import CMC
    true
catch
    false
end

if HAS_CMC
    include("invariants.jl")
    include("cmc_models.jl")
else
    @info "Skipping CMC-dependent tests (invariants.jl, cmc_models.jl)"
end

include("regression.jl")
include("errors.jl")
