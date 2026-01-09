#!/usr/bin/env julia
#==============================================================================#
# VACASK Benchmark Runner
#
# Runs all VACASK benchmarks and outputs results as markdown suitable for
# GitHub Actions Job Summaries.
#
# Usage:
#   julia --project=. benchmarks/vacask/run_benchmarks.jl [output_file]
#
# If output_file is provided, markdown is written there.
# Otherwise, it's written to stdout.
#==============================================================================#

using Pkg
Pkg.instantiate()

using Printf
using Statistics
using BenchmarkTools
using SciMLBase: ReturnCode
using Sundials: IDA
using OrdinaryDiffEq: FBDF, Rodas5P

const BENCHMARK_DIR = @__DIR__

# Solvers to test
const SOLVERS = [
    ("IDA", () -> IDA(max_error_test_failures=20)),
    ("FBDF", () -> FBDF()),
    ("Rodas5P", () -> Rodas5P()),
]

# Results storage
struct BenchmarkResult
    name::String
    solver::String
    status::Symbol  # :success, :failed, :skipped
    median_time::Float64  # seconds
    min_time::Float64
    max_time::Float64
    memory::Float64  # MB
    allocs::Int
    timepoints::Int
    rejected::Int
    error_msg::String
end

BenchmarkResult(name, solver, status, error_msg="") = BenchmarkResult(name, solver, status, NaN, NaN, NaN, NaN, 0, 0, 0, error_msg)

function format_time(seconds::Float64)
    if isnan(seconds)
        return "-"
    elseif seconds < 1e-3
        return @sprintf("%.2f µs", seconds * 1e6)
    elseif seconds < 1
        return @sprintf("%.2f ms", seconds * 1e3)
    else
        return @sprintf("%.2f s", seconds)
    end
end

function format_memory(mb::Float64)
    if isnan(mb)
        return "-"
    elseif mb < 1
        return @sprintf("%.1f KB", mb * 1024)
    elseif mb < 1024
        return @sprintf("%.1f MB", mb)
    else
        return @sprintf("%.2f GB", mb / 1024)
    end
end

#==============================================================================#
# Generate Markdown Report
#==============================================================================#
function generate_markdown(results::Vector{BenchmarkResult})
    io = IOBuffer()

    println(io, "# VACASK Benchmark Results")
    println(io)
    println(io, "Benchmarks run on Julia $(VERSION)")
    println(io)

    # Summary table with all solvers
    println(io, "## Summary")
    println(io)
    println(io, "| Benchmark | Solver | Status | Median Time | Timepoints | Rejected | Memory |")
    println(io, "|-----------|--------|--------|-------------|------------|----------|--------|")

    for r in results
        status_emoji = r.status == :success ? "✅" : r.status == :skipped ? "⏭️" : "❌"
        rejected_str = r.rejected >= 0 ? string(r.rejected) : "-"
        println(io, "| $(r.name) | $(r.solver) | $(status_emoji) | $(format_time(r.median_time)) | $(r.timepoints > 0 ? r.timepoints : "-") | $(rejected_str) | $(format_memory(r.memory)) |")
    end
    println(io)

    # Detailed results grouped by benchmark
    println(io, "## Detailed Results")
    println(io)

    # Group results by benchmark name
    benchmarks = unique(r.name for r in results)
    for bench_name in benchmarks
        println(io, "### $(bench_name)")
        println(io)

        bench_results = filter(r -> r.name == bench_name, results)
        successful = filter(r -> r.status == :success, bench_results)

        if !isempty(successful)
            println(io, "| Solver | Median | Min | Max | Rejected | Memory | Notes |")
            println(io, "|--------|--------|-----|-----|----------|--------|-------|")
            for r in successful
                notes = isempty(r.error_msg) ? "" : r.error_msg
                println(io, "| $(r.solver) | $(format_time(r.median_time)) | $(format_time(r.min_time)) | $(format_time(r.max_time)) | $(r.rejected) | $(format_memory(r.memory)) | $(notes) |")
            end
            println(io)

            # Show fastest
            fastest = argmin(r -> r.median_time, successful)
            println(io, "> 🏆 Fastest: **$(fastest.solver)** ($(format_time(fastest.median_time)))")
            println(io)
        end

        # Show failures
        failed = filter(r -> r.status == :failed, bench_results)
        for r in failed
            println(io, "> ❌ $(r.solver) failed: $(r.error_msg)")
        end
        println(io)
    end

    return String(take!(io))
end

#==============================================================================#
# Run individual benchmarks
#==============================================================================#

function run_benchmark_with_solver(name, script_path, solver_name, solver_fn)
    println("Running $name with $solver_name...")
    try
        # Include the benchmark script (only once per script)
        if !isdefined(Main, Symbol("__included_$(hash(script_path))"))
            include(script_path)
            Core.eval(Main, Expr(:(=), Symbol("__included_$(hash(script_path))"), true))
        end

        solver = solver_fn()
        bench, sol = Base.invokelatest(run_benchmark, solver)
        if bench === nothing || sol === nothing
            return BenchmarkResult(name, solver_name, :failed, "Benchmark returned nothing")
        end

        # Check if simulation reached the end time
        tspan_end = sol.prob.tspan[2]
        reached_end = isapprox(sol.t[end], tspan_end; rtol=1e-6)

        # Warn about non-success retcode but don't fail if we reached the end
        warning = ""
        if sol.retcode != ReturnCode.Success
            warning = " ($(sol.retcode))"
            @warn "$name with $solver_name: $(sol.retcode)"
        end

        if !reached_end
            return BenchmarkResult(name, solver_name, :failed,
                "Stopped at t=$(sol.t[end]), expected $(tspan_end)")
        end

        # Extract rejected steps from solver stats
        rejected = hasproperty(sol.stats, :nreject) ? sol.stats.nreject : 0

        return BenchmarkResult(
            name, solver_name, :success,
            median(bench.times) / 1e9, minimum(bench.times) / 1e9, maximum(bench.times) / 1e9,
            bench.memory / 1e6, bench.allocs, length(sol.t), rejected, warning
        )
    catch e
        return BenchmarkResult(name, solver_name, :failed, sprint(showerror, e))
    end
end

function run_benchmark_all_solvers(name, script_path)
    results = BenchmarkResult[]
    for (solver_name, solver_fn) in SOLVERS
        push!(results, run_benchmark_with_solver(name, script_path, solver_name, solver_fn))
    end
    return results
end

#==============================================================================#
# Main
#==============================================================================#
function main()
    println("=" ^ 60)
    println("VACASK Benchmark Suite")
    println("=" ^ 60)
    println()

    results = BenchmarkResult[]

    # RC Circuit - all solvers
    append!(results, run_benchmark_all_solvers(
        "RC Circuit",
        joinpath(BENCHMARK_DIR, "rc", "cedarsim", "runme.jl")
    ))

    # Graetz Bridge - all solvers
    append!(results, run_benchmark_all_solvers(
        "Graetz Bridge",
        joinpath(BENCHMARK_DIR, "graetz", "cedarsim", "runme.jl")
    ))

    # Voltage Multiplier - all solvers
    append!(results, run_benchmark_all_solvers(
        "Voltage Multiplier",
        joinpath(BENCHMARK_DIR, "mul", "cedarsim", "runme.jl")
    ))

    # # Ring Oscillator - all solvers
    # append!(results, run_benchmark_all_solvers(
    #     "Ring Oscillator",
    #     joinpath(BENCHMARK_DIR, "ring", "cedarsim", "runme.jl")
    # ))

    # # C6288 Multiplier - all solvers
    # append!(results, run_benchmark_all_solvers(
    #     "C6288 Multiplier",
    #     joinpath(BENCHMARK_DIR, "c6288", "cedarsim", "runme.jl")
    # ))

    println()
    println("=" ^ 60)
    println("Generating report...")
    println("=" ^ 60)

    markdown = generate_markdown(results)

    # Write to file or stdout
    if length(ARGS) >= 1
        output_file = ARGS[1]
        open(output_file, "w") do f
            write(f, markdown)
        end
        println("Report written to: $output_file")
    else
        println()
        println(markdown)
    end

    # Return success - skipped benchmarks don't cause failure
    return 0
end

exit(main())
