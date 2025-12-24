#==============================================================================#
# MNA: Modified Nodal Analysis Module
#
# This module provides the core MNA infrastructure for circuit simulation:
# - Context and stamping primitives (context.jl)
# - Sparse matrix assembly (build.jl)
# - Basic device stamps (devices.jl)
# - Analysis solvers: DC, AC, transient (solve.jl)
#
# The MNA formulation solves:
#     G*x + C*dx/dt = b
#
# where:
#     x = [V₁, V₂, ..., Vₙ, I₁, I₂, ..., Iₘ]ᵀ
#     G = conductance matrix (resistive part)
#     C = capacitance matrix (reactive part)
#     b = source vector (independent sources)
#
# Usage Example:
# ```julia
# using CedarSim.MNA
#
# # Create context and stamp circuit
# ctx = MNAContext()
# vcc = get_node!(ctx, :vcc)
# out = get_node!(ctx, :out)
#
# stamp!(VoltageSource(5.0), ctx, vcc, 0)
# stamp!(Resistor(1000.0), ctx, vcc, out)
# stamp!(Resistor(1000.0), ctx, out, 0)
#
# # Solve DC
# sol = solve_dc(ctx)
# @assert sol.x[out] ≈ 2.5  # Voltage divider
#
# # Or assemble for later use
# sys = assemble!(ctx)
# sol = solve_dc(sys)
# ```
#==============================================================================#

module MNA

using Printf
using ForwardDiff

# Context and stamping primitives
include("context.jl")

# Sparse matrix assembly
include("build.jl")

# Basic device stamps
include("devices.jl")

# Analysis solvers
include("solve.jl")

# VA contribution function support (Phase 5)
include("contrib.jl")

end # module MNA
