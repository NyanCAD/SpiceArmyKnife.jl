# MNA API Analysis: Current Code vs MNA Requirements

This document analyzes the current generated code patterns and determines
what API changes are needed to support direct MNA matrix stamping.

---

## Part 1: Concrete Examples of Generated Code

### 1.1 SPICE Netlist Example

**Input (SPICE):**
```spice
* Simple RC circuit
V1 vcc 0 DC 5
R1 vcc out 1k
C1 out 0 1u
```

**Generated Julia Code (from spc/codegen.jl):**
```julia
function (circuit)(🔍)
    # Create nets
    vcc = net(:vcc)
    out = net(:out)
    var"0" = net(Symbol("0"))

    # Ground the reference node
    Gnd()(var"0")

    # Instantiate devices
    Named(spicecall(vsource; dc=5.0), "V1")(vcc, var"0")
    Named(spicecall(resistor; r=1000.0), "R1")(vcc, out)
    Named(spicecall(capacitor; c=1e-6), "C1")(out, var"0")
end
```

**What each device call expands to:**

```julia
# net(:vcc) expands to (from simulate_ir.jl):
function Net(name)
    V = variable(name)           # DAECompiler: create voltage variable
    kcl! = equation(name)        # DAECompiler: create KCL equation slot
    dVdt = ddt(V)                # Pre-compute derivative
    return Net{typeof(dVdt)}(V, kcl!, 1.0)
end

# Resistor call: Named(SimpleResistor(r=1000), "R1")(vcc, out)
# Expands to (from simpledevices.jl):
function (R::SimpleResistor)(A, B; dscope=...)
    res = undefault(R.r)  # 1000.0
    branch!(dscope, A, B) do V, I
        I - V/res         # Returns residual: I - V/R = 0
    end
end

# branch! expands to (from simulate_ir.jl):
function branch!(scope, net₊, net₋)
    I = variable(scope(:I))      # Create branch current variable
    kcl!(net₊, -I)               # Add -I to KCL of positive node
    kcl!(net₋,  I)               # Add +I to KCL of negative node
    V = net₊.V - net₋.V          # Compute voltage across branch
    observed!(V, scope(:V))
    return (V, I)
end

# With callback form:
function branch!(f, scope, net₊, net₋)
    (V, I) = branch!(scope, net₊, net₋)
    equation!(f(V, I))           # Add equation: f(V,I) = 0
end

# Capacitor: I - ddt(V)*C = 0
function (C::SimpleCapacitor)(A, B; dscope=...)
    branch!(dscope, A, B) do V, I
        I - ddt(V)*C.capacitance  # I = C*dV/dt
    end
end

# Voltage source: V - Vdc = 0
function (VS::VoltageSource)(A, B; dscope=...)
    branch!(dscope, A, B) do V, I
        V - VS.dc                 # V = Vdc (constrains voltage)
    end
end

# Ground: V = 0
function (::Gnd)(A; dscope=...)
    kcl!(A, variable())           # Add dummy variable to KCL
    equation!(A.V - 0, scope)     # V = 0
end
```

### 1.2 Verilog-A Example

**Input (resistor.va):**
```verilog
module BasicVAResistor(p, n);
inout p, n;
electrical p, n;
parameter real R=1 exclude 0;

analog begin
    I(p,n) <+ V(p,n)/R;
end
endmodule
```

**Generated Julia Code (from vasim.jl make_spice_device):**
```julia
@kwdef struct BasicVAResistor <: VAModel
    R::DefaultOr{Float64} = 1.0
end

function (self::BasicVAResistor)(port_p, port_n; dscope=GenScope(debug_scope[], :BasicVAResistor))
    # Port voltage assignment
    p = port_p.V
    n = port_n.V

    # Create internal current variable for this branch
    I_p_n = DAECompiler.variable(DScope(dscope, Symbol("I(p, n)")))

    # Get parameter value
    R = undefault(getfield(self, :R))

    # Branch state tracking
    branch_state_p_n = CURRENT   # Contribution type
    branch_value_p_n = 0.0       # Accumulated contribution

    # === Analog block translation ===
    # I(p,n) <+ V(p,n)/R  becomes:
    branch_value_p_n += (p - n) / R

    # === KCL equations (external ports) ===
    kcl!(port_p, -I_p_n)  # Current flows out of p
    kcl!(port_n,  I_p_n)  # Current flows into n

    # === Branch equation ===
    # I(p,n) = accumulated current contributions
    DAECompiler.equation!(I_p_n - branch_value_p_n, DScope(dscope, Symbol("Branch(p, n)")))

    return ()
end
```

### 1.3 Verilog-A with ddt (Capacitor)

**Input (hypothetical capacitor.va):**
```verilog
module VACapacitor(p, n);
inout p, n;
electrical p, n;
parameter real C=1p;

analog begin
    I(p,n) <+ C * ddt(V(p,n));
end
endmodule
```

**Generated Julia Code:**
```julia
@kwdef struct VACapacitor <: VAModel
    C::DefaultOr{Float64} = 1e-12
end

function (self::VACapacitor)(port_p, port_n; dscope=...)
    p = port_p.V
    n = port_n.V

    I_p_n = DAECompiler.variable(DScope(dscope, Symbol("I(p, n)")))
    C = undefault(getfield(self, :C))

    branch_state_p_n = CURRENT
    branch_value_p_n = 0.0

    # I(p,n) <+ C * ddt(V(p,n))  becomes:
    branch_value_p_n += C * ddt(p - n)   # ddt from DAECompiler

    kcl!(port_p, -I_p_n)
    kcl!(port_n,  I_p_n)

    DAECompiler.equation!(I_p_n - branch_value_p_n, ...)

    return ()
end
```

### 1.4 Verilog-A with Nonlinear Charge (ddt(q))

**Input (NLVCR.va with charge):**
```verilog
module NLCap(p, n);
inout p, n;
electrical p, n;
parameter real C0=1p;

real q;

analog begin
    q = C0 * V(p,n) * V(p,n);  // q = C0 * V^2 (nonlinear)
    I(p,n) <+ ddt(q);
end
endmodule
```

**Generated Julia Code:**
```julia
function (self::NLCap)(port_p, port_n; dscope=...)
    p = port_p.V
    n = port_n.V

    I_p_n = DAECompiler.variable(...)
    C0 = undefault(getfield(self, :C0))

    branch_state_p_n = CURRENT
    branch_value_p_n = 0.0

    # Local variable
    q = 0.0

    # q = C0 * V(p,n)^2
    q = C0 * (p - n)^2

    # I(p,n) <+ ddt(q)
    branch_value_p_n += ddt(q)

    kcl!(port_p, -I_p_n)
    kcl!(port_n,  I_p_n)

    DAECompiler.equation!(I_p_n - branch_value_p_n, ...)

    return ()
end
```

---

## Part 2: Analysis - Can Current API Map to MNA?

### 2.1 What MNA Needs

For MNA, we need to build matrices:
```
G*x + C*dx/dt = b

where:
  x = [V_nodes..., I_branches...]  (node voltages + voltage source currents)
  G = conductance/Jacobian matrix
  C = capacitance matrix
  b = source vector
```

For each device, we need to extract:
1. **G stamps**: ∂(residual)/∂V for each node
2. **C stamps**: coefficients of ddt() terms
3. **b stamps**: constant source contributions

### 2.2 The Fundamental Mismatch

**Current API returns residuals:**
```julia
# Resistor returns: I - V/R = 0
branch!(A, B) do V, I
    I - V/R
end
```

**MNA needs Jacobian entries:**
```julia
# Resistor stamps:
# G[node_a, node_a] += 1/R
# G[node_a, node_b] -= 1/R
# G[node_b, node_a] -= 1/R
# G[node_b, node_b] += 1/R
```

**The problem:** From the residual `I - V/R`, we can't directly extract that:
- The `1/R` coefficient belongs in the G matrix
- This is a conductance, not a current source

### 2.3 The branch! Creates Unnecessary Variables

**Current behavior:**
```julia
function branch!(scope, net₊, net₋)
    I = variable(scope(:I))  # Creates current variable for EVERY branch
    ...
end
```

**MNA requirement:**
- Resistors/capacitors: NO current variable (stamped directly into G/C)
- Voltage sources: YES current variable (MNA extended formulation)
- Inductors: YES current variable (flux linkage equation)
- Current sources: NO current variable (stamps into b vector)

### 2.4 ddt() Is Opaque

**Current code:**
```julia
I - ddt(V)*C  # C*dV/dt contribution
```

**Problem:** `ddt(V)` returns a value at runtime. We can't extract `C` separately to stamp into the capacitance matrix without:
1. AD (what DAECompiler does)
2. Explicit separation of C and V

### 2.5 Nonlinear ddt(q) Is Even Harder

```julia
q = C0 * V^2    # Nonlinear charge
I = ddt(q)      # I = d(C0*V^2)/dt = 2*C0*V*dV/dt
```

The capacitance is now `∂q/∂V = 2*C0*V`, which is voltage-dependent.
For Newton iteration, we need:
- q(V) - the charge function
- C(V) = ∂q/∂V - the incremental capacitance

---

## Part 3: Proposed New API

### 3.1 Key Insight: Separate Contribution Types

Instead of returning a single residual, devices should provide separate:
1. **Resistive contribution** `f(x)` - stamps into G matrix
2. **Reactive contribution** `q(x)` - stamps into C matrix via ∂q/∂x
3. **Source contribution** - stamps into b vector

### 3.2 New Device Interface

```julia
abstract type MNAContribution end

struct ConductanceStamp <: MNAContribution
    node1::Int      # Row
    node2::Int      # Column (or 0 for diagonal only)
    value::Float64  # Conductance value
end

struct CapacitanceStamp <: MNAContribution
    node1::Int
    node2::Int
    value::Float64
end

struct CurrentSourceStamp <: MNAContribution
    node::Int
    value::Float64
end

struct VoltageSourceStamp <: MNAContribution
    node_pos::Int
    node_neg::Int
    current_var::Int  # Index of current variable
    voltage::Float64
end
```

### 3.3 New Device Implementations

**Resistor:**
```julia
function stamp!(R::SimpleResistor, ctx::MNAContext, A::MNANode, B::MNANode)
    G = 1.0 / R.r
    # Stamp conductance matrix
    stamp_conductance!(ctx, A.idx, A.idx,  G)
    stamp_conductance!(ctx, A.idx, B.idx, -G)
    stamp_conductance!(ctx, B.idx, A.idx, -G)
    stamp_conductance!(ctx, B.idx, B.idx,  G)
end
```

**Linear Capacitor:**
```julia
function stamp!(C::SimpleCapacitor, ctx::MNAContext, A::MNANode, B::MNANode)
    cap = C.capacitance
    # Stamp capacitance matrix (same pattern as conductance)
    stamp_capacitance!(ctx, A.idx, A.idx,  cap)
    stamp_capacitance!(ctx, A.idx, B.idx, -cap)
    stamp_capacitance!(ctx, B.idx, A.idx, -cap)
    stamp_capacitance!(ctx, B.idx, B.idx,  cap)
end
```

**Voltage Source:**
```julia
function stamp!(VS::VoltageSource, ctx::MNAContext, A::MNANode, B::MNANode)
    # Allocate current variable
    I_idx = allocate_current!(ctx, :V1)

    # Stamp G matrix for voltage source
    # KCL rows: I appears in node equations
    stamp_conductance!(ctx, A.idx, I_idx, 1.0)   # Current into A
    stamp_conductance!(ctx, B.idx, I_idx, -1.0)  # Current out of B

    # V equation row: V_A - V_B = Vdc
    stamp_conductance!(ctx, I_idx, A.idx, 1.0)
    stamp_conductance!(ctx, I_idx, B.idx, -1.0)
    stamp_source!(ctx, I_idx, VS.dc)
end
```

**Inductor:**
```julia
function stamp!(L::SimpleInductor, ctx::MNAContext, A::MNANode, B::MNANode)
    # Allocate current variable (inductor current is a state)
    I_idx = allocate_current!(ctx, :L1)

    # KCL rows: I appears in node equations
    stamp_conductance!(ctx, A.idx, I_idx, 1.0)
    stamp_conductance!(ctx, B.idx, I_idx, -1.0)

    # V = L*dI/dt  →  V - L*dI/dt = 0  →  V appears in G, L appears in C
    stamp_conductance!(ctx, I_idx, A.idx, 1.0)
    stamp_conductance!(ctx, I_idx, B.idx, -1.0)
    stamp_capacitance!(ctx, I_idx, I_idx, -L.inductance)  # Note: -L because residual form
end
```

### 3.4 Handling Nonlinear Devices

For nonlinear devices, we can't just stamp constants. We need:

```julia
function evaluate!(diode::SimpleDiode, ctx::MNAContext, A::MNANode, B::MNANode, x::Vector)
    V = x[A.idx] - x[B.idx]

    # Compute current and conductance
    I_d = diode.IS * (exp(V / (diode.N * Vt)) - 1)
    G_d = diode.IS / (diode.N * Vt) * exp(V / (diode.N * Vt))  # dI/dV

    # Equivalent current source for Newton
    I_eq = I_d - G_d * V

    # Stamp linearized model
    stamp_conductance!(ctx, A.idx, A.idx,  G_d)
    stamp_conductance!(ctx, A.idx, B.idx, -G_d)
    stamp_conductance!(ctx, B.idx, A.idx, -G_d)
    stamp_conductance!(ctx, B.idx, B.idx,  G_d)

    stamp_source!(ctx, A.idx, -I_eq)
    stamp_source!(ctx, B.idx,  I_eq)
end
```

### 3.5 Handling ddt(q) in Verilog-A

For `I(p,n) <+ ddt(q)` where `q = q(V)`:

```julia
function evaluate!(device::NLCap, ctx::MNAContext, P::MNANode, N::MNANode, x::Vector, dx::Vector)
    V = x[P.idx] - x[N.idx]
    dV = dx[P.idx] - dx[N.idx]  # Time derivative from solver

    # Compute charge and capacitance
    q = device.C0 * V^2           # q(V)
    C_incr = 2 * device.C0 * V    # dq/dV = incremental capacitance

    # For transient: I = dq/dt = C_incr * dV/dt
    # For Newton, we linearize around operating point

    # Stamp incremental capacitance
    stamp_capacitance!(ctx, P.idx, P.idx,  C_incr)
    stamp_capacitance!(ctx, P.idx, N.idx, -C_incr)
    stamp_capacitance!(ctx, N.idx, P.idx, -C_incr)
    stamp_capacitance!(ctx, N.idx, N.idx,  C_incr)

    # For charge-based formulation, we might also need to track q directly
end
```

---

## Part 4: Migration Path - Keeping Both APIs

### 4.1 The Problem with Changing branch!

The current `branch!` API is deeply embedded:
- Every device in simpledevices.jl uses it
- All Verilog-A generated code uses it
- Test cases depend on it

Changing it would break everything.

### 4.2 Proposed Solution: Dual-Mode Execution

**Option A: Mode flag in context**

```julia
const mna_mode = ScopedValue{Bool}(false)
const mna_ctx = ScopedValue{Union{Nothing, MNAContext}}(nothing)

function branch!(f, scope, net₊, net₋)
    if mna_mode[]
        # MNA path: collect stamps
        ctx = mna_ctx[]
        (V, I) = (net₊.V - net₋.V, 0.0)  # Dummy values for evaluation

        # Call f to get residual, then extract stamps via AD
        residual = f(V, I)

        # Use ForwardDiff to get Jacobian
        # ... stamp extraction logic ...
    else
        # Original DAECompiler path
        I = variable(scope(:I))
        kcl!(net₊, -I)
        kcl!(net₋,  I)
        V = net₊.V - net₋.V
        observed!(V, scope(:V))
        equation!(f(V, I))
        return (V, I)
    end
end
```

**Option B: New stamp_branch! for MNA-aware devices**

Keep `branch!` unchanged, add new interface:

```julia
# New MNA-specific interface
function stamp_branch!(ctx::MNAContext, device, A::MNANode, B::MNANode)
    # Device-specific stamping
    stamp!(device, ctx, A, B)
end

# Wrapper that detects which interface to use
function invoke_device(device, A, B; dscope=...)
    if supports_mna_stamping(device)
        stamp_branch!(mna_ctx[], device, A, B)
    else
        # Fall back to residual-based (requires AD extraction)
        device(A, B; dscope)
    end
end
```

### 4.3 Extracting MNA from Residual Form

For devices that only provide residual form, we can extract MNA stamps using AD:

```julia
function extract_mna_stamps(f, V_val, I_val)
    # Use ForwardDiff to get Jacobian
    function residual(x)
        V, I = x[1], x[2]
        f(V, I)
    end

    x0 = [V_val, I_val]
    J = ForwardDiff.jacobian(residual, x0)

    # J[1,1] = ∂residual/∂V  (goes in G matrix)
    # J[1,2] = ∂residual/∂I  (determines if current var needed)

    return (drdV = J[1,1], drdI = J[1,2], r0 = residual(x0))
end
```

For ddt terms, we need to detect them:

```julia
# Using a tagged type to track ddt
struct DDTMarker{T}
    value::T
end

function ddt_for_extraction(x)
    DDTMarker(x)
end

# Then in residual evaluation, check for DDTMarker
# to identify capacitance contributions
```

---

## Part 5: Tricky Cases

### 5.1 Voltage Sources (V = Vdc)

**Residual form:** `V - Vdc = 0` where I is unconstrained

**MNA requirement:** Need explicit current variable because:
- Voltage is constrained (not a free variable at nodes)
- Current must flow to satisfy KCL
- Creates an additional equation and variable

**Current code correctly creates current variable via branch!**

**MNA stamping:**
```julia
# For V1: V(A) - V(B) = Vdc
# With current I_V1 as additional unknown

# G matrix stamps:
# Row A (KCL):    ... +I_V1
# Row B (KCL):    ... -I_V1
# Row I_V1 (Veq): V(A) - V(B) = Vdc
#                 → G[I_V1, A] = 1, G[I_V1, B] = -1

# b vector:
# b[I_V1] = Vdc
```

### 5.2 Inductors (V = L*dI/dt)

**Residual form:** `V - L*ddt(I) = 0`

**MNA requirement:**
- Current I is a state variable (not just algebraic)
- Equation: V = L*dI/dt
- In matrix form: G has voltage terms, C has -L on diagonal for current

**MNA stamping:**
```julia
# Row I_L (inductor equation): V(A) - V(B) - L*dI/dt = 0
# G[I_L, A] = 1
# G[I_L, B] = -1
# C[I_L, I_L] = -L  (or +L depending on sign convention)
```

### 5.3 Nonlinear ddt(q) where q = q(V)

**Verilog-A:** `I(p,n) <+ ddt(q)` with `q = q(V)`

**The challenge:** `I = dq/dt = (dq/dV) * (dV/dt) = C(V) * dV/dt`

**For MNA:**
1. Compute `q(V)` and `C(V) = dq/dV` at operating point
2. Stamp `C(V)` into capacitance matrix
3. For Newton convergence, may need charge-based formulation:
   - State: q (charge)
   - Equation: q - q(V) = 0 (algebraic)
   - Plus: I = dq/dt (from KCL)

**Code pattern:**
```julia
function stamp_nonlinear_cap!(ctx, P, N, q_func, x)
    V = x[P.idx] - x[N.idx]

    # Get charge and incremental capacitance via AD
    q = q_func(V)
    C_incr = ForwardDiff.derivative(q_func, V)

    # Stamp incremental capacitance
    stamp_capacitance!(ctx, P.idx, P.idx,  C_incr)
    stamp_capacitance!(ctx, P.idx, N.idx, -C_incr)
    stamp_capacitance!(ctx, N.idx, P.idx, -C_incr)
    stamp_capacitance!(ctx, N.idx, N.idx,  C_incr)
end
```

### 5.4 VCCS (Voltage-Controlled Current Source)

**Verilog-A:** `I(out+,out-) <+ gm * V(in+,in-)`

**MNA stamping:**
```julia
# Current gm*V(in) injected at out nodes
# G[out+, in+] += gm
# G[out+, in-] -= gm
# G[out-, in+] -= gm
# G[out-, in-] += gm
```

**Note:** This is a 4-terminal device - current in output depends on voltage at input. The off-diagonal stamps connect input and output.

### 5.5 Current Sources

**Simple:** `I(p,n) <+ Idc`

**MNA stamping:**
```julia
# Just stamps into b vector
# b[P] -= Idc  (current flows out of P)
# b[N] += Idc  (current flows into N)
```

No current variable needed, no G entries.

---

## Part 6: Conclusions and Recommendations

### 6.1 The Current API Cannot Directly Map to MNA

The residual-based `branch!` API fundamentally mismatches MNA's stamp-based approach:
1. It creates current variables for all branches (MNA only needs them for V-sources, L)
2. It returns residuals instead of Jacobian entries
3. It doesn't separate resistive from reactive contributions

### 6.2 Recommended Approach

1. **Keep the existing API** for backward compatibility
2. **Add AD-based stamp extraction** for existing devices:
   - Evaluate residual at operating point
   - Use ForwardDiff to get Jacobian
   - Detect ddt terms via marker type
   - Build MNA stamps from Jacobian structure

3. **Add new stamp! interface** for performance-critical devices:
   - Explicit MNA stamping
   - No AD overhead
   - Can be added incrementally

4. **Modify Verilog-A codegen** to optionally emit stamp-based code:
   - Parse contributions to identify stamp patterns
   - Generate direct stamping for simple cases
   - Fall back to AD extraction for complex cases

### 6.3 The Key Insight

**We don't need to change the API** - we need to add a layer that extracts MNA structure from the existing residual-based evaluation:

```julia
function build_mna_system(circuit)
    ctx = MNAContext()

    # Trace circuit to discover structure
    with(mna_mode => true, mna_ctx => ctx) do
        circuit()
    end

    # Now ctx contains:
    # - List of nodes with indices
    # - List of branch currents (V-sources, inductors only)
    # - G, C, b matrix entries (from AD extraction)

    return finalize_mna_system(ctx)
end
```

This way, existing device code works unchanged, and we extract MNA structure through a combination of tracing and automatic differentiation.
