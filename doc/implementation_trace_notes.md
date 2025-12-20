# MNA Implementation Trace Notes

This document traces through the actual code path from a SPICE netlist with a Verilog-A
model to the generated Julia code. The goal is to understand exactly what exists and
define the minimal changes needed for MNA migration.

## Reference Test Case: test/ddx.jl

This test uses the NLVCR.va Verilog-A model which features:
- ddx() for small-signal derivative
- Current contribution: `I(d,s) <+ V(d,s)*ddx(cdrain, V(g,s))`
- Parameter: `R=1 exclude 0`

```julia
# test/ddx.jl
const NLVCR = load_VA_model(joinpath(@__DIR__, "NLVCR.va"))

function VRcircuit()
    vcc = Named(net, "vcc")()
    vg = Named(net, "vg")()
    gnd = Named(net, "gnd")()
    Named(V(5.), "V1")(vcc, gnd)
    Named(V(3.), "V2")(vg, gnd)
    Named(NLVCR(CedarSim.mknondefault(2.)), "R")(vcc, vg, gnd)  # Parameter override
    Gnd()(gnd)
end

sys, sol = solve_circuit(VRcircuit)
```

---

## Complete Code Path Trace

### 1. VA Model Loading

**Entry:** `load_VA_model("NLVCR.va")` in `src/ModelLoader.jl:20-60`

```
load_VA_model(path)
  ↓
parse_and_eval_vafile(mod, VAFile(path))   [src/vasim.jl:900-908]
  ↓
VerilogAParser.parsefile(file)             [external parser]
  ↓
make_module(va)                            [src/vasim.jl:890-898]
  ↓
make_spice_device(vamod)                   [src/vasim.jl:663-875]
```

### 2. make_spice_device() - Core VA Compilation

**Location:** `src/vasim.jl:663-875`

This is the heart of VA-to-Julia compilation. Key steps:

#### 2a. Parameter Extraction (lines 673-755)
```julia
# Creates struct fields with DefaultOr{T} for each VA parameter
push!(struct_fields,
    :($(Symbol(paramsym))::$(DefaultOr{pT}) = ...))
```

**Output pattern:**
```julia
@kwdef struct NLVCR <: VAModel
    R::DefaultOr{Float64} = 1.0
end
```

#### 2b. DDX Tracking (lines 649-661, 676)
```julia
find_ddx!(to_julia_global.ddx_order, vm)
# Finds all ddx() calls and tracks which voltages need dual-number treatment
```

For NLVCR: `ddx_order = [:g, :s]` (from `ddx(cdrain, V(g,s))`)

#### 2c. Branch Structure Setup (lines 757-764)
```julia
node_order = [ps; internal_nodes; Symbol("0")]
branch_order = collect(map(x->Pair(x...), combinations(node_order, 2)))
```

For NLVCR with ports (d,g,s): Creates all possible branch pairs.

#### 2d. Analog Block Compilation (lines 765-784)

Uses `Scope` functor to translate VA statements to Julia:
- `ContributionStatement` → Accumulates into branch_value_* variables
- `FunctionCall` with `:V` → Gets voltage (wrapped in Dual if tracked for ddx)
- `FunctionCall` with `:ddx` → Extracts ForwardDiff partial derivative

#### 2e. Generated Function Structure (lines 853-867)

```julia
function (#self#::NLVCR)(ports...; dscope=GenScope())
    # Port voltage assignment
    d, g, s = port.V...

    # Internal current variables - uses DAECompiler.variable()
    var"I(d, s)" = DAECompiler.variable(DScope(dscope, :...))

    # Parameters to locals
    R = undefault(getfield(#self#, :R))

    # Branch state tracking
    branch_state_d_s = CURRENT
    branch_value_d_s = 0.0

    # User analog block code (translated)
    cdrain = R*V(g,s)^2
    branch_value_d_s += ddx_result...

    # KCL equations for internal nodes
    DAECompiler.equation!(current_sum(node), DScope(...))

    # Branch equations
    DAECompiler.equation!(I(d,s) - branch_value_d_s, DScope(...))
end
```

### 3. Net and Branch Infrastructure

**Location:** `src/simulate_ir.jl`

#### Net Structure (lines 28-52)
```julia
struct Net{T} <: AbstractNet
    V::T           # Voltage variable (includes ddt for dynamics)
    kcl!::equation # KCL equation accumulator
    multiplier::Float64
end
```

**Critical:** `V` is actually `ddt(variable(name))` for dynamics support!

```julia
V = variable(name)
kcl! = equation(name)
dVdt = ddt(V)                    # Pre-compute derivative
obj = new{typeof(dVdt)}(V, ...)  # Store the Dual type
```

#### branch! Function (lines 112-140)
```julia
function branch!(scope, net₊, net₋)
    I = variable(scope(:I))      # Current variable
    kcl!(net₊, -I)               # Add to KCL equations
    kcl!(net₋,  I)
    V = net₊.V - net₋.V          # Voltage difference
    observed!(V, scope(:V))
    (V, I)
end
```

### 4. Circuit System Creation

**Location:** `src/circuitodesystem.jl`

#### DefaultSim (lines 12-25)
```julia
struct DefaultSim{T} <: AbstractSim{T}
    circuit::T
    mode::Symbol  # :tran, :dcop, etc.
end

function (sim::DefaultSim)()
    with(spec=>SimSpec(time=sim_time()), ...) do
        return circuit()
    end
end
```

#### ParamSim (lines 66-97)
```julia
struct ParamSim{T, S, P} <: AbstractSim{T}
    circuit::T
    mode::Symbol
    spec::S
    params::P
end

function (sim::ParamSim)()
    # Uses ParamLens for parameter override
    sim.circuit(ParamLens(sim.params))
end
```

#### CircuitIRODESystem (lines 147-165)
```julia
function CircuitIRODESystem(circuit::AbstractSim; kwargs...)
    tt = Tuple{typeof(circuit)}
    return IRODESystem(tt; kwargs...)  # DAECompiler entry point
end
```

### 5. Solution Pipeline

**Location:** `test/common.jl:36-42`

```julia
function solve_circuit(circuit::AbstractSim; time_bounds, reltol, abstol, u0)
    sys = CircuitIRODESystem(circuit)     # Creates DAECompiler system
    prob = ODEProblem(sys, u0, time_bounds, circuit)
    sol = solve(prob, FBDF(autodiff=false); ...)
    return sys, sol
end
```

---

## DAECompiler Primitives Used

| Primitive | Purpose | Location |
|-----------|---------|----------|
| `variable(scope)` | Create state variable | Net.V, branch current, internal nodes |
| `equation(scope)` | Create equation accumulator | Net.kcl! |
| `equation!(value, scope)` | Add equation to system | Branch equations, device equations |
| `ddt(x)` | Time derivative | Capacitors, inductors, reactive VA |
| `observed!(value, scope)` | Mark as observable output | Branch voltages, VA observables |
| `sim_time()` | Get simulation time | Time-dependent sources |

---

## DDX Implementation via ForwardDiff

**Location:** `src/vasim.jl:29-35`

```julia
struct SimTag; end  # Custom ForwardDiff tag

function DAECompiler.ddt(dual::ForwardDiff.Dual{SimTag})
    ForwardDiff.Dual{SimTag}(ddt(dual.value), map(ddt, dual.partials.values))
end
```

**In generated code (FunctionCall :V handler, lines 347-374):**
```julia
# When V(g,s) is a ddx probe, wrap in Dual:
id_idx = findfirst(==(id), to_julia.ddx_order)
return Expr(:call, Dual{SimTag}, id,
    Expr(:(...), ntuple(i->i == id_idx ? 1.0 : 0.0, length(ddx_order))))
```

**In ddx evaluation (lines 392-412):**
```julia
# Extract partial derivative:
:(let x = $(expr)
    $(isa)(x, $(Dual)) ? ForwardDiff.partials(SimTag, x, $id_idx) : 0.0
end)
```

---

## Key Insight: Implicit MNA

The current system **does not build an explicit MNA matrix**. Instead:

1. Each `variable()` call registers an unknown
2. Each `equation!()` call adds a residual equation
3. DAECompiler analyzes the function to discover structure
4. The Jacobian is computed via AD or symbolic methods

**This means MNA migration must:**
- Replace implicit structure discovery with explicit matrix stamping
- Convert residual form `f(x) = 0` to matrix form `Gx + C(dx/dt) = b`
- Or keep residual form but with known sparsity pattern

---

## spc Folder: New Semantic Analysis Framework

**Location:** `src/spc/`

### sema.jl - Semantic Analysis
```julia
mutable struct SemaResult
    ast::SNode
    CktID::Union{Nothing, Type}
    parse_cache::Union{Nothing, CedarParseCache}
    nets::Dict{Symbol, Vector{...}}
    params::OrderedDict{Symbol, Vector{...}}
    models::Dict{Symbol, Vector{...}}
    instances::OrderedDict{Symbol, Vector{...}}
    ...
end
```

### codegen.jl - Code Generation
```julia
function cg_spice_instance!(state, ports, name, model, param_exprs)
    # Generates: Named(spicecall(model; params...), name)(ports...)
end
```

### interface.jl - sp"..." String Macro
Entry point for SPICE netlist parsing. Creates `SpCircuit` type.

**This framework is for SPICE netlists, complementing the VA compilation in vasim.jl.**

---

## Minimal Changes for MNA Migration

### Option A: Keep DAECompiler, Add MNA Output

1. **Intercept at CircuitIRODESystem** - After DAECompiler analysis, extract structure
2. **Add MNA export** - Convert DAECompiler's internal representation to G, C, b matrices
3. **Modify solve path** - Allow direct MNA solver as alternative to ODE solver

**Pros:** Minimal code changes, preserves existing functionality
**Cons:** Still depends on DAECompiler for structure discovery

### Option B: Replace DAECompiler Primitives

1. **New MNA primitives** - `stamp_conductance!`, `stamp_capacitance!`, etc.
2. **Thin shim layer** - Map variable()/equation!() to MNA stamps during trace
3. **Modify vasim.jl** - Change generated code to use stamps instead of residuals

**Pros:** Clean separation, explicit control
**Cons:** More code changes, must replicate trace-time analysis

### Option C: Hybrid Approach

1. **Keep variable()/equation!() API** - Device code unchanged
2. **Add MNA backend to Net/branch!** - These are the aggregation points
3. **Replace CircuitIRODESystem** - New MNA system constructor

**Pros:** Device code unchanged, clear extension point
**Cons:** Still need trace-time structure discovery

---

## DAECompiler Deep Dive

### Core Entry Point: `factory_gen()`

**Location:** `/research/DAECompiler.jl/src/interface.jl:6-81`

```julia
function factory_gen(fT, settings, world)
    # 1. Type inference with AD support
    ci = ad_typeinf(world, Tuple{fT}; ...)

    # 2. Structural analysis - this is where intrinsics are found
    result = structural_analysis!(ci, world, settings)

    # 3. Build structure from analysis
    structure = make_structure_from_ipo(result)
    tstate = TransformationState(result, structure)

    # 4. Check consistency (balanced equations)
    err = StateSelection.check_consistency(tstate, nothing)

    # 5. State selection (differential vs algebraic)
    (diff_key, init_key) = top_level_state_selection!(tstate)

    # 6. Tearing/scheduling for efficient evaluation
    tearing_schedule!(tstate, ci, diff_key, world, settings)

    # 7. Generate ODE/DAE factory function
    ir_factory = ode_factory_gen(...) or dae_factory_gen(...)

    return src  # CodeInfo to be compiled
end
```

### Structural Analysis

**Location:** `/research/DAECompiler.jl/src/analysis/structural.jl:43-200`

The `_structural_analysis!` function:
1. Creates `DiffGraph` for variable-derivative relationships
2. Walks IR looking for intrinsic calls:
   - `variable()` → adds to var_to_diff graph
   - `equation()` → adds to equation list
   - `sim_time()` → marks time dependence
3. Annotates IR with `Incidence` types showing variable dependencies
4. Uses `StructuralRefiner` for dataflow analysis

**Key data structures:**
```julia
struct EqVarState
    var_to_diff        # DiffGraph for derivatives
    varclassification  # External vs Owned
    varkinds           # Continuous, Discrete, Epsilon
    total_incidence    # Which variables affect which equations
    eqclassification   # External vs Owned
    eqkinds            # Always, Initial, Observed
    eq_callee_mapping  # For equation reduction
end
```

### Intrinsic Recognition (lines 129-142)

```julia
if is_known_invoke(stmt, variable, compact)
    v = add_variable!(SSAValue(i))
    compact[SSAValue(i)][:type] = Incidence(v)  # Track variable number
elseif is_known_invoke(stmt, equation, compact)
    compact[SSAValue(i)][:type] = Eq(add_equation!(SSAValue(i)))
elseif is_known_invoke(stmt, sim_time, compact)
    compact[SSAValue(i)][:type] = Incidence(0)  # Time is variable 0
```

### The Handoff Point

**CircuitIRODESystem** wraps circuit in `DefaultSim` and calls `IRODESystem(tt)`:
```julia
function CircuitIRODESystem(circuit::AbstractSim; kwargs...)
    tt = Tuple{typeof(circuit)}
    return IRODESystem(tt; kwargs...)
end
```

**IRODESystem** is part of DAECompiler and triggers the full compilation pipeline:
1. When you create an `ODEProblem(sys, ...)`, DAECompiler's `factory()` is called
2. The circuit function is analyzed via `factory_gen()`
3. An `ODEFunction` with compiled RHS and Jacobian is produced

---

## Critical Insight for MNA Migration

The current flow is:
```
Circuit function → DAECompiler analysis → ODEFunction → SciML solver
```

For MNA migration, we have three options:

### Option 1: Post-Analysis Extraction
Intercept after `structural_analysis!` to extract:
- Variable list with names
- Equation list with dependencies
- Incidence structure (which vars affect which eqs)

Then build MNA matrices from this structure.

### Option 2: Trace-Time Collection
During circuit execution, collect stamps instead of calling DAECompiler:
- Replace `variable()/equation!()` with stamp collectors
- Build G, C, b matrices directly
- Generate SciML-compatible function from matrices

### Option 3: Parallel Representation
Keep DAECompiler path for validation, add MNA generation:
- Same circuit function produces both representations
- Compare results for debugging
- Switch backends based on circuit complexity

---

## Designing MNA Output (Working Backwards)

### Target: Simple RC Circuit

Consider this circuit:
```
V1 (vcc, gnd) = 5V
R1 (vcc, out) = 1kΩ
C1 (out, gnd) = 1µF
```

### Desired MNA Representation

Variables (x): `[V(vcc), V(out), I(V1)]`

MNA equations: `G*x + C*dx/dt = b`

```
G (conductance matrix):
         V(vcc)  V(out)  I(V1)
KCL(vcc) [  1/R   -1/R     1  ]   # Current into vcc: I(V1) - I(R1)
KCL(out) [ -1/R    1/R     0  ]   # Current into out: I(R1) - I(C1)
V1_eq    [   1      0      0  ]   # V(vcc) = 5V (voltage source)

C (capacitance matrix):
         V(vcc)  V(out)  I(V1)
KCL(vcc) [   0      0      0  ]
KCL(out) [   0      C      0  ]   # C1 contributes C*dV(out)/dt
V1_eq    [   0      0      0  ]

b (source vector):
[ 0  ]   # No current source at vcc
[ 0  ]   # No current source at out
[ 5  ]   # V1 = 5V
```

### Desired Julia Data Structures

```julia
struct MNASystem
    # Variable information
    node_names::Vector{Symbol}          # [:vcc, :out, :gnd]
    node_to_idx::Dict{Symbol, Int}      # gnd always maps to 0 (reference)
    current_vars::Vector{Symbol}        # [:I_V1] - voltage source currents

    # Matrix data (sparse)
    G::SparseMatrixCSC{Float64}         # Conductance matrix
    C::SparseMatrixCSC{Float64}         # Capacitance matrix
    b::Vector{Float64}                  # Source vector

    # For observable reconstruction
    observables::Dict{Symbol, Function} # name => (x) -> computed_value
end
```

### Working Backwards: What Code Would Generate This?

**For the resistor (R1):**
```julia
function stamp_resistor!(sys::MNASystem, n1::Int, n2::Int, G::Float64)
    # Stamp 2x2 conductance block
    if n1 > 0
        sys.G[n1, n1] += G
        if n2 > 0
            sys.G[n1, n2] -= G
        end
    end
    if n2 > 0
        sys.G[n2, n2] += G
        if n1 > 0
            sys.G[n2, n1] -= G
        end
    end
end
```

**For the capacitor (C1):**
```julia
function stamp_capacitor!(sys::MNASystem, n1::Int, n2::Int, C::Float64)
    # Stamp 2x2 capacitance block
    if n1 > 0
        sys.C[n1, n1] += C
        if n2 > 0
            sys.C[n1, n2] -= C
        end
    end
    if n2 > 0
        sys.C[n2, n2] += C
        if n1 > 0
            sys.C[n2, n1] -= C
        end
    end
end
```

**For the voltage source (V1):**
```julia
function stamp_vsource!(sys::MNASystem, n1::Int, n2::Int, V::Float64, Ivar::Int)
    # Stamp voltage source with current variable
    # n1 is + terminal, n2 is - terminal
    if n1 > 0
        sys.G[n1, Ivar] += 1.0      # Current flows into n1
        sys.G[Ivar, n1] += 1.0      # V(n1) appears in V equation
    end
    if n2 > 0
        sys.G[n2, Ivar] -= 1.0      # Current flows out of n2
        sys.G[Ivar, n2] -= 1.0      # -V(n2) appears in V equation
    end
    sys.b[Ivar] = V                  # V(n1) - V(n2) = V
end
```

### Mapping from Current Code to MNA Stamps

| Current Code | MNA Equivalent |
|--------------|----------------|
| `Net(name)` | Allocate node index, create KCL row |
| `variable(scope(:I))` in branch! | Allocate current variable column |
| `equation!(val, scope)` | Add `val` to appropriate row |
| `ddt(V)*C` in capacitor | Stamp C matrix |
| Resistor `V/R` | Stamp G matrix |
| Voltage source | Stamp G matrix + b vector |

### Minimal Modification Approach

**Phase 1: Add MNA collection mode (non-invasive)**

1. Create `MNACollector` struct that accumulates stamps
2. Add flag to `SimSpec` to enable collection mode
3. In `Net` constructor, optionally allocate node index
4. In `branch!`, optionally record branch info
5. In device functions, call stamp functions when in collection mode

**Phase 2: Generate matrices from collected data**

1. After circuit trace, build sparse G, C, b from stamps
2. Create `MNASystem` compatible with SciML ODE interface:
   ```julia
   function (sys::MNASystem)(du, u, p, t)
       # du = -C⁻¹ * G * u + C⁻¹ * b
       # Or use mass matrix form:
       # C * du = -G * u + b
   end
   ```

**Phase 3: Switch solve path**

Replace:
```julia
sys = CircuitIRODESystem(circuit)
prob = ODEProblem(sys, u0, tspan, circuit)
```

With:
```julia
mna = build_mna_system(circuit)
prob = ODEProblem(mna, u0, tspan)
```

### Key Insight: Where to Intercept

The cleanest interception point is the `Net` and `branch!` infrastructure in `simulate_ir.jl`:

```julia
# Current code:
struct Net{T} <: AbstractNet
    V::T           # DAECompiler variable
    kcl!::equation # DAECompiler equation
    multiplier::Float64
end

# Modified code for MNA:
struct Net{T} <: AbstractNet
    V::T
    kcl!::equation
    multiplier::Float64
    node_idx::Int           # NEW: MNA node index
end

function Net(name, multiplier)
    if mna_mode[]
        node_idx = allocate_node!(mna_collector[], name)
        return MNANet(node_idx, multiplier)
    else
        # Original DAECompiler path
        ...
    end
end
```

Similarly for `branch!`:
```julia
function branch!(scope, net₊, net₋)
    if mna_mode[]
        # Record branch for MNA
        return (net₊.V - net₋.V, mna_current_var)
    else
        # Original DAECompiler path
        ...
    end
end
```

---

## Recommended Next Steps

1. ~~Study how DAECompiler.IRODESystem analyzes the circuit function~~
2. ~~Identify the exact point where variables and equations are enumerated~~
3. ~~Design MNA matrix structure that captures the same information~~
4. Implement shim that collects stamps during trace execution
5. Test with simple RLC circuit before tackling VA models
