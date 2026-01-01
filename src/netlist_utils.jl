# This file contains utility functions when writing netlists using the operadic
# julia embedding.
# Note: Most of this functionality has been deprecated with the move to MNA.
# The parallel/series operators are kept for backwards compatibility.

export ∥, ⋯, parallel, series

## Diagrammatic Composition
"""
    ∥(...)
    parallel(args...)

The parallel diagrammatic composition operator `∥` (\\parallel or using
the ASCII name `parallel`) may be used for the parallel composition of
subcircuits or devices.

# Examples

```julia
# Two resistors in parallel.
const R500 = R(1k) ∥ R(1k)

# Array of LEDs for output signals
∥(Fix1.(LED, outputs)...)(G)
```
"""
function ∥(devices...)
    (args...)->foreach(devices) do dev
        dev(args...)
    end
end
const parallel = ∥

"""
    ⋯(...)
    series(args...)

The sequential diagrammatic composition operator `⋯` (\\cdots or using
the ASCII name `series`) may be used to sequentially compose circuit elements.

# Examples

```julia
# Two resistors in series.
const R2k = R(1k) ⋯ R(1k)

# Inverter chain
inverter(vdd, vss) = (in, out)->(pmos(vdd, in, out, vdd); nmos(out, in, vss, vss))
inverter_chain(n) = (vdd, vss, in, out)->mapreduce(_->inverter(vdd, vss), ⋯, 1:n)(in, out)
```
"""
function series(dev1, devs...)
    (l, r)->dev1(l, foldr(devs; init=r) do (dev, nr)
        # This would need Net() which is removed - series requires explicit nets now
        error("series operator requires explicit net creation in MNA mode")
    end)
end
const ⋯ = series
