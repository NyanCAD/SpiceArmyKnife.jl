# MNA Codegen Analysis: When Can We Stamp Directly?

## The Key Question

For a Verilog-A contribution like `I(p,n) <+ V(p,n)/R`, can we:
1. Recognize it's a simple conductance at codegen time
2. Generate direct stamp code (no current variable)
3. Avoid the residual/AD path entirely

**Answer: Yes, for many common cases.**

---

## Contribution Classification

### Case 1: Linear Current Contribution (Direct G Stamp)

**Pattern:** `I(a,b) <+ linear_expr(V(...))`

**Examples:**
```verilog
I(p,n) <+ V(p,n)/R;           // Resistor
I(p,n) <+ gm * V(c,e);        // VCCS (transconductance)
I(out,0) <+ V(in,0) * gain;   // Voltage-controlled current
```

**Recognition:** RHS is linear combination of V() terms with constant coefficients.

**Codegen:** Extract coefficients, generate stamp code:
```julia
# For I(p,n) <+ V(p,n)/R
function stamp!(self::BasicVAResistor, ctx, p, n)
    G = 1.0 / self.R
    stamp_G!(ctx, p, p,  G)
    stamp_G!(ctx, p, n, -G)
    stamp_G!(ctx, n, p, -G)
    stamp_G!(ctx, n, n,  G)
end
```

**No current variable needed.**

---

### Case 2: Linear Capacitive Contribution (Direct C Stamp)

**Pattern:** `I(a,b) <+ C * ddt(V(...))`  or `I(a,b) <+ ddt(C * V(...))`

**Examples:**
```verilog
I(p,n) <+ C * ddt(V(p,n));    // Linear capacitor
I(p,n) <+ ddt(C * V(p,n));    // Equivalent
```

**Recognition:** RHS is `ddt(linear_in_V)` or `constant * ddt(V)`.

**Codegen:**
```julia
function stamp!(self::VACapacitor, ctx, p, n)
    C = self.C
    stamp_C!(ctx, p, p,  C)
    stamp_C!(ctx, p, n, -C)
    stamp_C!(ctx, n, p, -C)
    stamp_C!(ctx, n, n,  C)
end
```

**No current variable needed.**

---

### Case 3: Constant Current Source (Direct b Stamp)

**Pattern:** `I(a,b) <+ constant`

**Examples:**
```verilog
I(p,n) <+ Idc;                // DC current source
I(p,n) <+ 1m;                 // 1mA source
```

**Codegen:**
```julia
function stamp!(self::VACurrentSource, ctx, p, n)
    I = self.Idc
    stamp_b!(ctx, p, -I)  # Current flows out of p
    stamp_b!(ctx, n,  I)  # Current flows into n
end
```

**No current variable needed.**

---

### Case 4: Voltage Contribution (NEEDS Current Variable)

**Pattern:** `V(a,b) <+ expr`

**Examples:**
```verilog
V(p,n) <+ Vdc;                // Voltage source
V(p,n) <+ gain * V(c,d);      // VCVS
```

**Why current variable needed:** Voltage is constrained, but KCL still needs current to balance. The current is unknown and must be solved for.

**Codegen:**
```julia
function stamp!(self::VAVoltageSource, ctx, p, n)
    # Allocate current variable
    I_idx = alloc_current!(ctx, self.name)

    # KCL: current flows through
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # Voltage equation: V(p) - V(n) = Vdc
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)
    stamp_b!(ctx, I_idx, self.Vdc)
end
```

---

### Case 5: Inductor (NEEDS Current Variable)

**Pattern:** `V(a,b) <+ L * ddt(I(a,b))`

**Why current variable needed:** Current is a state variable (has ddt).

**Codegen:**
```julia
function stamp!(self::VAInductor, ctx, p, n)
    I_idx = alloc_current!(ctx, self.name)

    # KCL
    stamp_G!(ctx, p, I_idx,  1.0)
    stamp_G!(ctx, n, I_idx, -1.0)

    # V = L * dI/dt  →  in residual form: V - L*dI/dt = 0
    stamp_G!(ctx, I_idx, p,  1.0)
    stamp_G!(ctx, I_idx, n, -1.0)
    stamp_C!(ctx, I_idx, I_idx, -self.L)  # The dI/dt term
end
```

---

### Case 6: Nonlinear Current (Runtime AD)

**Pattern:** `I(a,b) <+ nonlinear_expr(V(...))`

**Examples:**
```verilog
I(p,n) <+ Is * (exp(V(p,n)/Vt) - 1);     // Diode
I(d,s) <+ Kp * (Vgs - Vth)^2;            // MOSFET
```

**Cannot stamp at codegen time** - need to linearize at operating point.

**Codegen:** Generate evaluation code that computes Jacobian via AD:
```julia
function evaluate!(self::VADiode, ctx, p, n, x)
    V = x[p] - x[n]

    # Compute current and conductance
    I = self.Is * (exp(V / self.Vt) - 1)
    G = self.Is / self.Vt * exp(V / self.Vt)  # dI/dV

    # Equivalent current source: Ieq = I - G*V
    Ieq = I - G * V

    # Stamp linearized model
    stamp_G!(ctx, p, p,  G)
    stamp_G!(ctx, p, n, -G)
    stamp_G!(ctx, n, p, -G)
    stamp_G!(ctx, n, n,  G)
    stamp_b!(ctx, p, -Ieq)
    stamp_b!(ctx, n,  Ieq)
end
```

---

### Case 7: Nonlinear Charge ddt(q(V)) (Runtime AD)

**Pattern:** `I(a,b) <+ ddt(q)` where `q = nonlinear(V)`

**Examples:**
```verilog
q = C0 * V(p,n) * V(p,n);    // Nonlinear charge
I(p,n) <+ ddt(q);
```

**At runtime:** Compute `C_incr = dq/dV` via AD, stamp into C matrix.

```julia
function evaluate!(self::VANonlinearCap, ctx, p, n, x)
    V = x[p] - x[n]

    # Compute charge and incremental capacitance
    q = self.C0 * V^2
    C_incr = 2 * self.C0 * V  # dq/dV

    stamp_C!(ctx, p, p,  C_incr)
    stamp_C!(ctx, p, n, -C_incr)
    stamp_C!(ctx, n, p, -C_incr)
    stamp_C!(ctx, n, n,  C_incr)
end
```

---

## Codegen Strategy for vasim.jl

### Step 1: Classify Each Contribution

When processing `ContributionStatement` in vasim.jl:

```julia
function classify_contribution(contrib::ContributionStatement)
    lhs_kind = Symbol(contrib.lvalue.id)  # :I or :V
    rhs = contrib.assign_expr

    if lhs_kind == :V
        return :voltage_source  # Always needs current var
    end

    # Analyze RHS
    if is_constant(rhs)
        return :current_source  # Just stamp b
    elseif is_linear_in_V(rhs) && !contains_ddt(rhs)
        return :conductance     # Stamp G directly
    elseif is_linear_ddt_V(rhs)
        return :capacitance     # Stamp C directly
    else
        return :nonlinear       # Runtime AD needed
    end
end
```

### Step 2: Generate Appropriate Code

```julia
function emit_contribution(kind, contrib, to_julia)
    if kind == :conductance
        # Extract linear coefficients and generate stamp code
        coeffs = extract_linear_coeffs(contrib.rhs)
        emit_stamp_G(coeffs)
    elseif kind == :capacitance
        coeffs = extract_ddt_coeffs(contrib.rhs)
        emit_stamp_C(coeffs)
    elseif kind == :current_source
        val = eval_constant(contrib.rhs)
        emit_stamp_b(val)
    elseif kind == :voltage_source
        emit_vsource_stamps(contrib)
    else  # :nonlinear
        emit_runtime_eval(contrib)
    end
end
```

### Step 3: Example Generated Code

**Input:**
```verilog
module BasicVAResistor(p, n);
inout p, n;
electrical p, n;
parameter real R=1;
analog begin
    I(p,n) <+ V(p,n)/R;
end
endmodule
```

**Generated (NEW - direct stamping):**
```julia
@kwdef struct BasicVAResistor <: MNADevice
    R::Float64 = 1.0
end

function stamp!(self::BasicVAResistor, ctx::MNAContext, p::Int, n::Int)
    G = 1.0 / self.R

    # 2x2 conductance stamp
    stamp_G!(ctx, p, p,  G)
    stamp_G!(ctx, p, n, -G)
    stamp_G!(ctx, n, p, -G)
    stamp_G!(ctx, n, n,  G)
end
```

**No current variable. No runtime evaluation. Just stamps.**

---

## What About simpledevices.jl?

The same principle applies. Instead of:

```julia
# OLD: residual-based
function (R::SimpleResistor)(A, B; dscope=...)
    branch!(dscope, A, B) do V, I
        I - V/R.r
    end
end
```

We write:

```julia
# NEW: direct stamping
function stamp!(R::SimpleResistor, ctx::MNAContext, A::Int, B::Int)
    G = 1.0 / R.r
    stamp_G!(ctx, A, A,  G)
    stamp_G!(ctx, A, B, -G)
    stamp_G!(ctx, B, A, -G)
    stamp_G!(ctx, B, B,  G)
end
```

---

## Summary: When Do We Need Current Variables?

| Contribution Type | Current Variable? | Why |
|-------------------|-------------------|-----|
| `I(a,b) <+ linear(V)` | NO | Stamps directly into G |
| `I(a,b) <+ ddt(linear(V))` | NO | Stamps directly into C |
| `I(a,b) <+ constant` | NO | Stamps directly into b |
| `I(a,b) <+ nonlinear(V)` | NO | Linearize + stamp G, b |
| `V(a,b) <+ anything` | YES | Voltage constraint needs I |
| `V(a,b) <+ L*ddt(I)` | YES | Inductor, I is state |
| `I(a,b) <+ f(I(a,b))` | YES | I appears on RHS |

**The key insight:** Most devices are current contributions that can be stamped directly. Only voltage constraints and inductors need explicit current variables.

---

## Implementation Plan

1. **Modify vasim.jl** to classify contributions and emit stamp code for simple cases
2. **Rewrite simpledevices.jl** to use stamp! interface instead of branch!
3. **New MNA context** that collects stamps during circuit trace
4. **Build matrices** from collected stamps
5. **Solve** using SciML (ODEProblem with mass matrix or DAEProblem)

For nonlinear devices, the stamp! function takes the solution vector x and computes linearized stamps at each Newton iteration.
