# Internal Node Support Implementation Plan

## Executive Summary

This document presents a comprehensive plan for implementing internal node support in Verilog-A models within the MNA backend. Internal nodes are circuit nodes declared within a module that are not accessible from outside (not terminals). They are essential for realistic device models like BJTs with parasitic resistances, diodes with series resistance, and MOSFETs with internal access nodes.

---

## 1. Problem Statement

### Current Limitations

The current `make_mna_device()` in `vasim.jl` handles only terminal nodes:
- Nodes are passed as arguments to `stamp!(..., p, n, ...)`
- Internal nodes declared via `electrical a_int;` are parsed but not allocated
- No mechanism to expand the solution vector for internal node voltages

### Example: Diode with Series Resistance

```verilog
module DiodeRs(a, c);
    parameter real Is = 1e-14;
    parameter real Rs = 10.0;
    inout a, c;
    electrical a, c, a_int;  // a_int is INTERNAL
    analog begin
        I(a, a_int) <+ V(a, a_int) / Rs;      // Series resistance
        I(a_int, c) <+ Is*(exp(V(a_int,c)/0.026) - 1);  // Junction
    end
endmodule
```

Currently this fails because `a_int` is not allocated as a circuit node.

---

## 2. Research Findings

### 2.1 OpenVAF/OSDI Approach

OpenVAF distinguishes between:
- **Terminals** (`num_terminals`): External ports visible to the circuit
- **Nodes** (`num_nodes`): Total nodes including internal ones

Key concepts from `sim_back/src/topology.rs`:
- **Implicit equations**: Internal nodes create implicit algebraic constraints
- **Node collapse**: Pairs of nodes can be collapsed (e.g., when Rs=0, `a` and `a_int` become the same)
- **Contribution structure**: Separate resistive/reactive contributions tracked per branch

The OSDI descriptor has:
```rust
pub struct OsdiDescriptor {
    pub num_nodes: u32,        // Total nodes (terminals + internal)
    pub num_terminals: u32,    // External terminals only
    pub num_collapsible: u32,  // Pairs that can collapse
    pub collapsible: Vec<OsdiNodePair>,
    // ...
}
```

### 2.2 DAECompiler (Old) Approach

From `vasim.jl` lines 806-812, internal nodes were handled via:
```julia
internal_nodeset = map(enumerate(internal_nodes)) do (n, id)
    quote
        $id = variable(DScope(dscope, Symbol("V($id)")))
    end
end
```

Each internal node created a DAECompiler variable, and KCL equations were added automatically.

### 2.3 SciML/DifferentialEquations.jl Approach

From SciMLDocs analysis:
- **Algebraic variables** are marked with `0` in mass matrix diagonal
- **Structural elimination**: Internal variables can be eliminated and computed on-demand
- **Index reduction**: Higher-index DAEs require special handling

For circuit simulation:
- Internal nodes are algebraic (their voltage is determined by KCL)
- Mass matrix marks them as `M[i,i] = 0` (no capacitive contribution)
- They add rows to the system for KCL equations

---

## 3. Architectural Options

### Option A: Static Internal Node Allocation

**Approach**: Allocate internal node indices at device stamp time, similar to how current variables are allocated for voltage sources.

```julia
function stamp!(dev::DiodeRs, ctx::MNAContext, a::Int, c::Int; kwargs...)
    # Allocate internal node
    a_int = alloc_internal_node!(ctx, Symbol("$(typeof(dev).name).a_int"))

    # Now stamp contributions using a, a_int, c
    stamp_Rs!(ctx, a, a_int, dev.Rs)
    stamp_diode!(ctx, a_int, c, dev.Is, dev.Vt)
end
```

**Pros**:
- Simple, matches existing `alloc_current!` pattern
- Internal node indices are known at stamp time
- Works with existing Newton iteration

**Cons**:
- Each device instance allocates separate internal nodes
- No sharing between instances (correct behavior)
- Matrix grows with internal nodes

### Option B: Hierarchical Node Management

**Approach**: Generate unique internal node names per device instance using hierarchical naming.

```julia
function stamp!(dev::DiodeRs, ctx::MNAContext, a::Int, c::Int;
                instance_name::Symbol=:unnamed, kwargs...)
    # Use instance-qualified name for internal node
    a_int_name = Symbol(instance_name, ".a_int")
    a_int = get_node!(ctx, a_int_name)
    # ...
end
```

**Pros**:
- Explicit naming prevents conflicts
- Easy debugging (node names are meaningful)
- Compatible with SPICE hierarchical naming

**Cons**:
- Requires passing instance names through call chain
- More complex signature

### Option C: Device-Local Node Vector

**Approach**: Each device maintains its own internal node allocation during stamp.

```julia
struct DiodeRsStamper
    dev::DiodeRs
    internal_nodes::Vector{Int}  # Indices allocated at stamp time
end

function stamp!(s::DiodeRsStamper, ctx::MNAContext, a::Int, c::Int; kwargs...)
    a_int = s.internal_nodes[1]  # Pre-allocated
    # ...
end
```

**Pros**:
- Clean separation of device data and circuit topology
- Supports complex devices with many internal nodes

**Cons**:
- Requires two-phase initialization
- More complex API

### Recommendation: Option A with Enhancement

Use Option A (static allocation) with enhancement for instance naming:

```julia
function alloc_internal_node!(ctx::MNAContext, name::Symbol)::Int
    ctx.n_nodes += 1
    push!(ctx.node_names, name)
    ctx.node_to_idx[name] = ctx.n_nodes

    # Extend b vector
    if length(ctx.b) < system_size(ctx)
        resize!(ctx.b, system_size(ctx))
        ctx.b[end] = 0.0
    end

    return ctx.n_nodes
end
```

This matches the existing pattern and is simple to implement.

---

## 4. Detailed Implementation Plan

### Phase 6.1: Internal Node Allocation Infrastructure

**Files**: `src/mna/context.jl`

Add to MNAContext:
```julia
# Track internal vs terminal nodes for debugging
internal_node_flags::BitVector  # true if node is internal

function alloc_internal_node!(ctx::MNAContext, name::Symbol)::Int
    idx = get_node!(ctx, name)
    # Mark as internal
    if idx > length(ctx.internal_node_flags)
        resize!(ctx.internal_node_flags, idx)
    end
    ctx.internal_node_flags[idx] = true
    return idx
end
```

**Tests**:
```julia
@testset "Internal node allocation" begin
    ctx = MNAContext()
    a = get_node!(ctx, :a)
    a_int = alloc_internal_node!(ctx, :a_int)
    @test a_int > a  # Internal nodes come after
    @test ctx.internal_node_flags[a_int] == true
    @test ctx.internal_node_flags[a] == false
end
```

### Phase 6.2: Codegen for Internal Nodes

**Files**: `src/vasim.jl` - Modify `make_mna_device()` and `generate_mna_stamp_method_nterm()`

#### 6.2.1 Collect Internal Nodes

In `make_mna_device()`, internal nodes are already collected:
```julia
internal_nodes = Vector{Symbol}()
for net in item.net_names
    id = Symbol(assemble_id_string(net.item))
    if !(id in ps)
        push!(internal_nodes, id)
    end
end
```

#### 6.2.2 Generate Allocation Code

In `generate_mna_stamp_method_nterm()`, add internal node allocation:

```julia
# Generate internal node allocation at top of stamp! method
internal_alloc = Expr(:block)
for (i, inode) in enumerate(internal_nodes)
    inode_var = Symbol("_inode_", inode)
    push!(internal_alloc.args,
        :($inode_var = CedarSim.MNA.alloc_internal_node!(ctx,
            Symbol(string(typeof(dev).name), ".", $(QuoteNode(inode)))))
    )
end
```

#### 6.2.3 Expand Dual Creation for Internal Nodes

Currently duals are created only for terminal nodes. Expand to include internal:

```julia
# All nodes: terminals + internal
all_nodes = [port_args..., internal_nodes...]
n_all = length(all_nodes)

# Create duals for all node voltages
for i in 1:n_all
    if i <= n_ports
        # Terminal: passed as argument
        V_expr = :($(node_params[i]) == 0 ? 0.0 : x[$(node_params[i])])
    else
        # Internal: allocated at stamp time
        inode_var = Symbol("_inode_", internal_nodes[i - n_ports])
        V_expr = :(x[$inode_var])
    end
    # Create dual with partial for this node
    ...
end
```

### Phase 6.3: Solution Vector Management

**Files**: `src/mna/solve.jl`

The solution vector must accommodate internal nodes. Since they're allocated during stamping, the system size is only known after circuit traversal.

```julia
function solve_dc(builder, params, spec; maxiter=100, tol=1e-9)
    # Initial stamp to determine system size
    ctx = builder(params, spec; x=Float64[])
    n = system_size(ctx)

    # Initialize solution
    x = zeros(n)

    # Newton iteration
    for iter in 1:maxiter
        clear_stamps!(ctx)
        ctx = builder(params, spec; x=x)
        sys = assemble!(ctx)

        # Solve linear system
        dx = sys.G \ (sys.b - sys.G * x)
        x .+= dx

        if norm(dx) < tol
            break
        end
    end

    return DCSolution(x, ctx)
end
```

### Phase 6.4: Transient Analysis with Internal Nodes

For transient analysis, internal nodes participate in the DAE:

```
C * dx/dt + G * x = b
```

Internal nodes typically have no capacitance (pure algebraic constraints from KCL). The mass matrix `C` will have zero rows for internal nodes, making them algebraic variables.

**Key insight**: This naturally fits the SciML DAE pattern:
- Mass matrix diagonal: `1.0` for capacitive nodes, `0.0` for algebraic (internal) nodes
- DAE solvers like `IDA` or `Rodas5P` handle this correctly

### Phase 6.5: Node Collapse Optimization (Optional)

OpenVAF includes node collapse for when internal nodes are trivially connected (e.g., Rs=0):

```julia
# At stamp time, check if resistance is zero
if dev.Rs ≈ 0.0
    # Don't allocate internal node, use terminal directly
    a_int = a  # Collapse a_int to a
else
    a_int = alloc_internal_node!(ctx, ...)
end
```

This optimization reduces system size but adds complexity. Defer to later.

---

## 5. Test Cases

### 5.1 Diode with Series Resistance

```julia
@testset "Diode with series resistance" begin
    va"""
    module VADiodeRs(a, c);
        parameter real Is = 1e-14;
        parameter real Rs = 10.0;
        inout a, c;
        electrical a, c, a_int;
        analog begin
            I(a, a_int) <+ V(a, a_int) / Rs;
            I(a_int, c) <+ Is*(exp(V(a_int,c)/0.026) - 1.0);
        end
    endmodule
    """

    function diode_rs_circuit(params, spec; x=Float64[])
        ctx = MNAContext()
        anode = get_node!(ctx, :anode)
        stamp!(VoltageSource(0.7; name=:V1), ctx, anode, 0)
        stamp!(VADiodeRs(Is=1e-14, Rs=10.0), ctx, anode, 0; x=x)
        return ctx
    end

    sol = solve_dc(diode_rs_circuit, (;), MNASpec())

    # With Rs=10Ω, voltage drop across Rs reduces junction voltage
    # V_junction < 0.7V, so current is lower than ideal diode
    I = -current(sol, :I_V1)
    @test I < 1e-14 * (exp(0.7/0.026) - 1)  # Less than ideal
    @test I > 0  # But still forward biased
end
```

### 5.2 BJT with Internal Nodes

```julia
@testset "Simple BJT structure" begin
    # Simplified BJT with internal base node
    va"""
    module VABJT(c, b, e);
        parameter real Is = 1e-15;
        parameter real Rb = 100.0;
        parameter real Bf = 100.0;
        inout c, b, e;
        electrical c, b, e, b_int;
        analog begin
            I(b, b_int) <+ V(b, b_int) / Rb;
            I(b_int, e) <+ Is*(exp(V(b_int,e)/0.026) - 1.0);
            I(c, e) <+ Bf * Is*(exp(V(b_int,e)/0.026) - 1.0);
        end
    endmodule
    """

    # Common emitter test
    # ...
end
```

### 5.3 Multiple Internal Nodes

```julia
@testset "Multiple internal nodes" begin
    # Device with two internal nodes
    va"""
    module TwoInternalNodes(p, n);
        parameter real R1 = 100.0;
        parameter real R2 = 200.0;
        parameter real R3 = 300.0;
        inout p, n;
        electrical p, n, int1, int2;
        analog begin
            I(p, int1) <+ V(p, int1) / R1;
            I(int1, int2) <+ V(int1, int2) / R2;
            I(int2, n) <+ V(int2, n) / R3;
        end
    endmodule
    """

    # Should behave as R1 + R2 + R3 in series
    # ...
end
```

---

## 6. Architecture Decisions

### 6.1 Internal Node Indexing

Internal nodes are indexed after terminal nodes in the solution vector:

```
x = [V_terminal_1, V_terminal_2, ..., V_internal_1, V_internal_2, ..., I_1, I_2, ...]
     |<-- terminals -->|             |<-- internal -->|             |<-- currents -->|
```

This matches the existing pattern where current variables come after nodes.

### 6.2 Dual Partial Ordering

For AD with internal nodes, partials must cover all nodes:

```julia
# For device with terminals (p, n) and internal (a_int):
# Create duals with partials for all 3 voltages
V_p = Dual{Nothing}(x[p], (1.0, 0.0, 0.0))      # ∂/∂V_p
V_n = Dual{Nothing}(x[n], (0.0, 1.0, 0.0))      # ∂/∂V_n
V_a_int = Dual{Nothing}(x[a_int], (0.0, 0.0, 1.0))  # ∂/∂V_a_int
```

This allows automatic Jacobian extraction for all nodes including internal.

### 6.3 Instance Naming

Each device instance needs unique internal node names:

```julia
# For: X1 a b DiodeRs Rs=10
# Internal node named: :X1.a_int

# For: X2 c d DiodeRs Rs=20
# Internal node named: :X2.a_int
```

The SPICE codegen already tracks instance names, so this can be passed through.

### 6.4 Ground Handling

Internal nodes cannot be ground (0). If a contribution references ground:

```verilog
I(a_int, 0) <+ ...  // a_int to ground
```

The codegen handles `0` specially (skip stamping to ground row/column).

---

## 7. Implementation Checklist

### Phase 6.1: Infrastructure (Est. ~100 LOC)
- [ ] Add `alloc_internal_node!()` to `context.jl`
- [ ] Add `internal_node_flags` tracking to MNAContext
- [ ] Add tests for internal node allocation
- [ ] Export new functions

### Phase 6.2: Codegen (Est. ~200 LOC)
- [ ] Modify `make_mna_device()` to pass internal nodes to generator
- [ ] Modify `generate_mna_stamp_method_nterm()` to:
  - [ ] Generate allocation code for internal nodes
  - [ ] Create duals with partials for all nodes (terminals + internal)
  - [ ] Handle contributions referencing internal nodes
- [ ] Add instance name parameter to stamp! signature
- [ ] Add tests for generated code structure

### Phase 6.3: Newton Solver Updates (Est. ~50 LOC)
- [ ] Ensure `solve_dc()` handles dynamic system size
- [ ] Add internal node values to solution accessors
- [ ] Add tests for DC convergence with internal nodes

### Phase 6.4: Transient Updates (Est. ~50 LOC)
- [ ] Ensure mass matrix handles algebraic (internal) nodes
- [ ] Test transient with internal nodes
- [ ] Verify DAE solver compatibility

### Phase 6.5: Integration Tests (Est. ~200 LOC)
- [ ] DiodeRs test (internal node for series resistance)
- [ ] BJT simplified test (internal base node)
- [ ] Multiple internal nodes test
- [ ] Hierarchical naming test
- [ ] VADistiller model compatibility tests

---

## 8. Future Enhancements

### 8.1 Node Collapse
When parameters make internal nodes trivially connected:
```julia
if dev.Rs ≈ 0.0
    # Collapse a_int to a
end
```

### 8.2 Observable Internal Nodes
Allow accessing internal node voltages in output:
```julia
sol[:X1.a_int]  # Get internal node voltage
```

### 8.3 Operating Point Analysis
Report internal node operating points for debugging:
```julia
print_operating_point(sol)
# Node    Voltage
# vcc     5.000
# out     2.500
# X1.a_int 4.950  # Internal
```

### 8.4 Sparse Pattern Optimization
Pre-compute Jacobian sparsity pattern including internal nodes for faster Newton.

---

## 9. References

1. **OpenVAF topology.rs**: Branch and contribution handling
2. **OpenVAF node_collapse.rs**: Node collapse optimization
3. **OpenVAF osdi_0_4.rs**: OSDI interface with num_nodes vs num_terminals
4. **SciMLDocs**: DAE handling, mass matrices, algebraic variables
5. **MNA Design Docs**: `doc/mna_design.md`, `doc/mna_architecture.md`
6. **VA Test File**: `test/mna/vadistiller.jl` showing needed features

---

## 10. Appendix: Full Example Generated Code

For a diode with series resistance, the generated stamp! method would look like:

```julia
function CedarSim.MNA.stamp!(dev::VADiodeRs, ctx::CedarSim.MNA.MNAContext,
                             _node_a::Int, _node_c::Int;
                             t::Real=0.0, mode::Symbol=:dcop, x::AbstractVector=Float64[],
                             spec::CedarSim.MNA.MNASpec=CedarSim.MNA.MNASpec(),
                             instance_name::Symbol=:unnamed)
    # Extract parameters
    Is = undefault(dev.Is)
    Rs = undefault(dev.Rs)

    # Allocate internal node
    _inode_a_int = CedarSim.MNA.alloc_internal_node!(ctx,
        Symbol(string(instance_name), ".a_int"))

    # Get operating point voltages
    V_1 = _node_a == 0 ? 0.0 : (isempty(x) ? 0.0 : x[_node_a])
    V_2 = _node_c == 0 ? 0.0 : (isempty(x) ? 0.0 : x[_node_c])
    V_3 = isempty(x) ? 0.0 : x[_inode_a_int]

    # Create duals with partials for all 3 nodes
    a = Dual{Nothing}(V_1, (1.0, 0.0, 0.0))
    c = Dual{Nothing}(V_2, (0.0, 1.0, 0.0))
    a_int = Dual{Nothing}(V_3, (0.0, 0.0, 1.0))

    # Contribution 1: I(a, a_int) <+ V(a, a_int) / Rs
    I_branch1 = (a - a_int) / Rs
    # Extract and stamp...

    # Contribution 2: I(a_int, c) <+ Is*(exp(V(a_int,c)/0.026) - 1)
    I_branch2 = Is * (exp((a_int - c) / 0.026) - 1.0)
    # Extract and stamp...

    return nothing
end
```

This structure preserves constant folding for device parameters while supporting internal node voltage differentiation.
