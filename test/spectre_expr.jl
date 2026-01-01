module test_spectre

using CedarSim
using SpectreNetlistParser
using SpectreNetlistParser.SPICENetlistParser: SPICENetlistCSTParser
using Test
using CedarSim.SpectreEnvironment

# MNA is always available - DAECompiler simulation was removed

code = """
parameters p1=23pf p2=.3 p3 = 1&2~^3 p4 = true && false || true p5 = M_1_PI * 3.0
r1 (1 0) resistor r=p1      // another simple expression // fdsfdsf
r2 (1 0) resistor r=p2*p2   // a binary multiply expression
r3 (1 0) resistor r=(p1+p2)/p3      // a more complex expression
r4 (1 0) resistor r=sqrt(p1+p2)     // an algebraic function call
r5 (1 0) resistor r=3+atan(p1/p2) //a trigonometric function call
r6 (1 0) resistor r=((p1<1) ? p4+1 : p3)  // the ternary operator
"""

@testset "spectre parameters" begin
    # Test parsing
    ast = SpectreNetlistParser.parse(code)
    @test ast !== nothing

    # Test code generation (produces valid Julia AST) using new MNA API
    fn = CedarSim.make_mna_circuit(ast)
    @test fn isa Expr
end

@testset "3 port BJT" begin
    str = """
    * 3 port BJT
    q0 c b e  vpnp_0p42x10  dtemp=dtemp
    """
    stmt = SPICENetlistCSTParser.parse(IOBuffer(str))
    # Just test that parsing works - the old SpcScope codegen is removed
    @test stmt !== nothing
end

@testset "spectre parsing" begin
    # Additional parsing tests for Phase 0
    spectre_code = """
    c1 (Y 0) capacitor c=100f
    r2 (Y VDD) resistor R=10k
    v1 (VDD 0) vsource type=dc dc=0.7
    """
    ast = SpectreNetlistParser.parse(spectre_code)
    @test ast !== nothing
    @test length(ast.stmts) >= 3

    # Use new MNA API
    code = CedarSim.make_mna_circuit(ast)
    @test code isa Expr
end

@testset "SPICE parsing" begin
    spice_code = """
    * Simple RC circuit
    V1 vcc 0 DC 5
    R1 vcc out 1k
    C1 out 0 1u
    """
    ast = SpectreNetlistParser.parse(IOBuffer(spice_code); start_lang=:spice, implicit_title=true)
    @test ast !== nothing

    # Use new MNA API
    code = CedarSim.make_mna_circuit(ast)
    @test code isa Expr
end

end # module test_spectre
