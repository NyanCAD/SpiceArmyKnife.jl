# Path to Zero-Allocation Verilog-A Models

## Executive Summary

The current VA integration (`vasim.jl`) uses ForwardDiff to automatically extract Jacobians from Verilog-A contribution statements. This approach allocates ~2KB per Newton iteration, but **ForwardDiff Duals should inline to simple arithmetic** when properly specialized.

The key insight is that Duals with compile-time known sizes are stack-allocated and should be zero-allocation. The allocations likely come from:
1. Runtime type dispatch (`if x isa Dual{Tag}`)
2. Dynamic tuple sizes for partials
3. Closure variable capture

This document outlines how to make Dual-based evaluation zero-allocation while preserving correctness and GPU compatibility.

**Note**: Enzyme is an alternative AD approach but may have GPU compatibility issues. Sticking with ForwardDiff Duals is simpler and proven to work with StaticArrays.

## Current Architecture Analysis

### How VA Models Work Now

```
VA Source → VerilogAParser → AST → make_mna_device() → stamp!(dev, ctx, nodes...; x, t, spec)
                                                              ↓
                                        [Each Newton iteration]
                                              ↓
                              1. Extract parameters from dev struct
                              2. Allocate internal nodes (idempotent)
                              3. Create Dual{JacobianTag} for each node voltage
                              4. Evaluate analog block with duals
                              5. Wrap ddt() values in Dual{ContributionTag}
                              6. Extract partials → stamp G, C, b
```

### Allocation Sources

| Source | Location | Bytes/call | Cause |
|--------|----------|------------|-------|
| Dual creation | vasim.jl:1584-1590 | ~128 | `Dual{JacobianTag}(V, partials...)` |
| Partials tuple | vasim.jl:1588 | ~64 | `ntuple(...)` for partials |
| `va_ddt()` | contrib.jl:96-114 | ~32 | `Dual{ContributionTag}(...)` |
| COO push! | context.jl | ~48 | First call only, then idempotent |
| Intermediate allocations | varies | ~200+ | Expressions creating temporaries |

**Total: ~500+ bytes per contribution evaluation, ~2KB per stamp! call for complex devices**

### Generated stamp! Method Structure

From `generate_mna_stamp_method_nterm()` (vasim.jl:1729-1771):

```julia
function stamp!(dev::DeviceName, ctx, node1, node2, ...; x, t, spec)
    # 1. Parameter extraction (no allocation)
    param1 = undefault(dev.param1)

    # 2. Internal node allocation (idempotent)
    _node_internal = alloc_internal_node!(ctx, :name)

    # 3. Voltage extraction (no allocation)
    V_1 = x[node1]

    # 4. Dual creation (ALLOCATES!)
    node1 = Dual{JacobianTag}(V_1, (1.0, 0.0, ...))  # ← Problem

    # 5. Evaluate contributions (may allocate via va_ddt)
    I_branch = expr_using_duals  # ← Problem if uses ddt()

    # 6. Extract Jacobian (type dispatch)
    if I_branch isa Dual{ContributionTag}  # ← Runtime check
        ...
    end

    # 7. Stamp into COO (first call allocates, subsequent don't)
    stamp_G!(ctx, p, k, dI_dVk)
end
```

## Zero-Allocation Architecture

### Key Insight: Duals Inline to Arithmetic

When ForwardDiff Duals have compile-time known sizes, they inline to simple scalar operations:

```julia
# This Dual code:
V = Dual{Tag}(1.0, (1.0, 0.0))  # V with ∂V/∂V₁=1, ∂V/∂V₂=0
I = V / R

# Compiles to equivalent of:
I_val = V_val / R
I_partial_1 = V_partial_1 / R  # = 1/R
I_partial_2 = V_partial_2 / R  # = 0
```

No heap allocation needed - just scalar arithmetic on the stack.

### Design Principles

1. **Keep ForwardDiff Duals** - they work, just need proper specialization
2. **Compile-time known sizes** - use `Val{N}` to fix partials tuple length
3. **Eliminate runtime type dispatch** - use `@generated` or multiple dispatch
4. **Pre-computed COO indices** - separate structure discovery from value evaluation

### Why NOT Symbolic Derivatives

Extracting symbolic derivatives from Verilog-A AST would require:
- Pattern matching for all VA math functions
- Handling nested expressions, conditionals, loops
- Special cases for ddt(), ddx(), idt()
- Massive complexity for marginal benefit

ForwardDiff already does this correctly. We just need to make it zero-allocation.

### Type Hierarchy

```julia
# For CPU simulation (sparse matrices)
struct VACompiledDevice{F,P,S}
    # Stamp structure (discovered at first call, frozen after)
    stamp_pattern::S            # COO indices for G, C, b

    # Evaluation function (generated, specialized)
    eval_fn::F                  # (params, spec, x, t) -> (G_vals, C_vals, b_vals)

    # Device data
    params::P
end

# For GPU ensemble (StaticArrays)
struct VAStaticDevice{N,T,K,F,P}
    # Pre-computed stamp pattern as tuples
    G_indices::NTuple{K_G, Tuple{Int,Int}}
    C_indices::NTuple{K_C, Tuple{Int,Int}}
    b_indices::NTuple{K_b, Int}

    # Specialized evaluation function
    eval_fn::F                  # (params, x, t) -> values tuple

    # Device parameters (as tuple for type stability)
    params::P
end
```

### Evaluation Without Duals

Instead of runtime AD, generate specialized evaluation code at compile time:

```julia
# Current (allocating):
function eval_contribution(x)
    V1 = Dual{JacobianTag}(x[1], 1.0, 0.0)  # Allocates
    V2 = Dual{JacobianTag}(x[2], 0.0, 1.0)  # Allocates
    I = V1 / R + C * va_ddt(V1 - V2)        # More allocations
    # ... extract from duals
end

# Zero-allocation:
@generated function eval_contribution_static(x::SVector{2,T}, params) where T
    quote
        V1, V2 = x[1], x[2]
        Vpn = V1 - V2

        # Resistive: I_R = Vpn / R
        I_R = Vpn / params.R
        dI_R_dV1 = 1 / params.R
        dI_R_dV2 = -1 / params.R

        # Reactive: I_C = C * ddt(Vpn) → stamps C into C matrix
        # Value contribution is 0 (capacitor current at DC)
        # Jacobian contribution is C
        dq_dV1 = params.C
        dq_dV2 = -params.C

        # Return all values (no Dual allocation)
        (I = I_R,
         dI_dV = (dI_R_dV1, dI_R_dV2),
         q = zero(T),
         dq_dV = (dq_dV1, dq_dV2))
    end
end
```

### Stamp Pattern Discovery

Run the model once to discover the COO pattern, then freeze it:

```julia
struct StampPattern
    # G matrix stamps: (row, col, evaluator_index)
    G_stamps::Vector{Tuple{Int,Int,Int}}

    # C matrix stamps: (row, col, evaluator_index)
    C_stamps::Vector{Tuple{Int,Int,Int}}

    # b vector stamps: (row, evaluator_index)
    b_stamps::Vector{Tuple{Int,Int}}
end

function compile_va_device(dev, ctx, nodes...; x)
    # Run stamp! once to discover pattern
    stamp!(dev, ctx, nodes...; x=x)

    # Extract pattern from ctx's COO arrays
    pattern = StampPattern(...)

    # Generate specialized evaluation function
    eval_fn = generate_eval_function(dev, pattern)

    return VACompiledDevice(pattern, eval_fn, dev_params)
end
```

## Implementation Phases

### Phase 1: Fix Dual Size at Compile Time

The generated stamp! method creates Duals with runtime-determined size. Fix this:

```julia
# Current (vasim.jl:1588) - partials tuple size known but not propagated:
partials_tuple = Expr(:tuple, [k == i ? 1.0 : 0.0 for k in 1:n_all_nodes]...)
$node_sym = Dual{JacobianTag}($(Symbol("V_", i)), $partials_tuple...)

# Better - use Partials{N} explicitly:
# This ensures ForwardDiff knows the size at compile time
$node_sym = Dual{JacobianTag,Float64,$n_all_nodes}(
    $(Symbol("V_", i)),
    Partials{$n_all_nodes,Float64}($partials_tuple))
```

### Phase 2: Aggressive Inlining for Constant Folding

The current code uses runtime `isa` checks (vasim.jl:1434-1461):

```julia
# Current - runtime dispatch:
if I_branch isa ForwardDiff.Dual{ContributionTag}
    # has ddt
elseif I_branch isa ForwardDiff.Dual
    # pure resistive
else
    # scalar
end
```

**Key insight**: With aggressive inlining, these branches can be **constant-folded** by the compiler because the type of `I_branch` is known at compile time!

**Fix 1**: Ensure `@inline` on all VA evaluation functions:
```julia
@inline function va_ddt(x::Real)
    Dual{ContributionTag}(zero(x), x)
end

@inline function evaluate_branch(...)
    # The type of I_branch is inferrable from the contribution expression
    I_branch = expr...
    # If expr uses va_ddt, type is Dual{ContributionTag,...}
    # If not, type is Dual{JacobianTag,...} or scalar
    # Compiler knows this → constant folds the isa check
end
```

**Fix 2**: Use `@generated` to specialize at codegen time:
```julia
# Track at codegen time whether device has reactive components
has_ddt = any(expr -> contains_ddt(expr), contributions)

# Generate specialized code path (no runtime check needed)
if has_ddt
    # Generate code that expects ContributionTag outer Dual
else
    # Generate code that expects JacobianTag Dual or scalar
end
```

**Fix 3**: Mark stamp! functions for forced inlining:
```julia
# In generated stamp! method
@inline function stamp!(dev::$symname, ctx, nodes...; x, t, spec)
    # With @inline, the entire evaluation chain can be inlined
    # Type inference propagates through, branches constant-fold
    ...
end
```

When the entire call chain is inlined:
1. Dual types are known at compile time
2. `if x isa SomeType` evaluates to `if true` or `if false`
3. Dead branches are eliminated
4. Result: just scalar arithmetic

### Phase 3: Pre-allocate COO Structure

Separate structure discovery from value updates:

```julia
struct VACompiledCircuit{N,T}
    # COO indices (discovered once, frozen)
    G_I::Vector{Int}
    G_J::Vector{Int}
    G_V::Vector{T}  # Pre-allocated, updated in-place

    C_I::Vector{Int}
    C_J::Vector{Int}
    C_V::Vector{T}

    b_I::Vector{Int}
    b_V::Vector{T}

    # Evaluation function (specialized for this device)
    eval_fn::Function
end

function rebuild_values!(vc::VACompiledCircuit, x, t, params)
    # Reset values (no allocation)
    fill!(vc.G_V, zero(T))
    fill!(vc.C_V, zero(T))
    fill!(vc.b_V, zero(T))

    # Evaluate with Duals and write to pre-allocated arrays
    vc.eval_fn(vc.G_V, vc.C_V, vc.b_V, x, t, params)
end
```

### Phase 4: StaticArrays for GPU Ensemble

For small circuits with known size N, use StaticArrays throughout:

```julia
struct VAStaticCircuit{N,T,F}
    # Device evaluator (generated, takes SVector returns SMatrix/SVector)
    eval_fn::F
    params::NamedTuple
end

# Evaluation with Duals but zero allocation (all stack):
@inline function evaluate_static(
    vc::VAStaticCircuit{N,T,F},
    u::SVector{N,T}, t
) where {N,T,F}
    # Create Duals with StaticArrays partials
    u_duals = create_jacobian_duals(u)  # Returns SVector of Duals

    # Evaluate device (all on stack)
    G, C, b = vc.eval_fn(u_duals, t, vc.params)

    return G, C, b  # SMatrix, SMatrix, SVector
end

# The key: create_jacobian_duals uses Partials backed by SVector
@generated function create_jacobian_duals(u::SVector{N,T}) where {N,T}
    exprs = [:(Dual{JacobianTag,T,N}(u[$i],
               Partials{N,T}(ntuple(j -> j == $i ? one(T) : zero(T), Val($N)))))
             for i in 1:N]
    :(SVector{$N}($(exprs...)))
end
```

### Phase 5: Verify Zero Allocation

Create tests that verify the full evaluation path is allocation-free:

```julia
@testset "VA model zero allocation" begin
    # Compile VA resistor to static form
    va_circuit = compile_va_static(va_resistor, params, Val(2))

    u = @SVector [1.0, 0.0]

    # Warmup
    G, C, b = evaluate_static(va_circuit, u, 0.0)

    # Test
    function test_va_alloc(vc, n)
        for _ in 1:n
            u = @SVector [1.0, 0.0]
            G, C, b = evaluate_static(vc, u, 0.0)
        end
    end
    test_va_alloc(va_circuit, 10)

    allocs = @allocated test_va_alloc(va_circuit, 100)
    @test allocs == 0
end
```

## Migration Path

### Step 1: Add `compile_va_circuit()` (Parallel to existing API)

```julia
# New API for compiled circuits
compiled = compile_va_circuit(builder, params, spec, n)

# Returns CompiledVACircuit that can be used with:
# - StaticCircuit for GPU ensemble
# - PrecompiledCircuit for CPU with reduced allocation
```

### Step 2: Optimize Common Device Types

Priority order:
1. **Resistor, Capacitor, Inductor** - Trivial analytic Jacobians
2. **Diode** - Simple exponential pattern
3. **MOSFET Level 1** - Quadratic I-V, well-known Jacobian
4. **BSIM4** - Complex but high-value target

### Step 3: Integrate with Existing Tests

```julia
# Verify compiled versions match original
function test_compiled_equivalence(va_code)
    original = load_va_model(va_code)
    compiled = compile_va_model(va_code)

    for x in test_points
        r_orig = evaluate_original(original, x)
        r_comp = evaluate_compiled(compiled, x)
        @test r_orig ≈ r_comp
    end
end
```

## Expected Results

| Metric | Current | After Phase 1-2 | After Phase 3-4 |
|--------|---------|-----------------|-----------------|
| Bytes/iteration (simple) | ~500 | ~100 | 0 |
| Bytes/iteration (BSIM4) | ~5000 | ~500 | ~50 (cache only) |
| GPU ensemble support | ❌ | ❌ | ✅ |
| StaticArrays support | ❌ | ❌ | ✅ |

## Files to Modify/Create

| File | Changes |
|------|---------|
| `src/mna/va_compiled.jl` | **NEW**: VACompiledDevice, pattern extraction |
| `src/mna/va_static.jl` | **NEW**: VAStaticDevice for GPU |
| `src/vasim.jl` | Add symbolic Jacobian extraction |
| `src/mna/compiled.jl` | Integration with StaticCircuit |
| `test/mna/va_compiled.jl` | **NEW**: Zero-allocation tests |

## Open Questions

1. **How to handle ddx()?** The `ddx()` VA function computes partial derivatives - may need special handling.

2. **Conditional contributions**: VA `if` blocks that add/remove contributions require careful pattern handling.

3. **Internal node aliasing**: Short-circuit detection currently modifies structure at runtime.

4. **Time-dependent sources**: PWL, SIN sources change `b` vector - may need separate handling.

## Conclusion

Zero-allocation VA models are achievable by **fixing how Duals are used**, not replacing them:

1. **Fix Dual sizes at compile time** - Use `Dual{Tag,T,N}` with explicit N
2. **Eliminate runtime type dispatch** - Generate specialized code paths at codegen time
3. **Pre-allocate COO structure** - Separate structure discovery from value updates
4. **Use StaticArrays for partials** - Stack-allocated Partials{N,T}

ForwardDiff Duals inline to simple arithmetic when properly specialized. No need for:
- Symbolic derivative extraction from VA AST
- Analytical Jacobian tables
- Complex pattern matching

The migration can be done incrementally by modifying `generate_mna_stamp_method_nterm()` in vasim.jl to emit more specialized code.
