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

const BENCHMARK_DIR = @__DIR__

# Results storage
struct BenchmarkResult
    name::String
    status::Symbol  # :success, :failed, :skipped
    median_time::Float64  # seconds
    min_time::Float64
    max_time::Float64
    memory::Float64  # MB
    allocs::Int
    timepoints::Int
    error_msg::String
end

BenchmarkResult(name, status, error_msg="") = BenchmarkResult(name, status, NaN, NaN, NaN, NaN, 0, 0, error_msg)

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

    # Summary table
    println(io, "## Summary")
    println(io)
    println(io, "| Benchmark | Status | Median Time | Timepoints | Memory |")
    println(io, "|-----------|--------|-------------|------------|--------|")

    for r in results
        status_emoji = r.status == :success ? "✅" : r.status == :skipped ? "⏭️" : "❌"
        println(io, "| $(r.name) | $(status_emoji) $(r.status) | $(format_time(r.median_time)) | $(r.timepoints > 0 ? r.timepoints : "-") | $(format_memory(r.memory)) |")
    end
    println(io)

    # Detailed results
    println(io, "## Detailed Results")
    println(io)

    for r in results
        println(io, "### $(r.name)")
        println(io)

        if r.status == :success
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")
            println(io, "| Median Time | $(format_time(r.median_time)) |")
            println(io, "| Min Time | $(format_time(r.min_time)) |")
            println(io, "| Max Time | $(format_time(r.max_time)) |")
            println(io, "| Timepoints | $(r.timepoints) |")
            println(io, "| Memory | $(format_memory(r.memory)) |")
            println(io, "| Allocations | $(r.allocs) |")
        elseif r.status == :skipped
            println(io, "> ⏭️ Skipped: $(r.error_msg)")
        else
            println(io, "> ❌ Failed: $(r.error_msg)")
        end
        println(io)
    end

    return String(take!(io))
end

#==============================================================================#
# Run individual benchmarks
#==============================================================================#

function run_rc_benchmark()
    println("Running RC Circuit benchmark...")
    try
        include(joinpath(BENCHMARK_DIR, "rc", "cedarsim", "runme.jl"))
        bench, sol = Base.invokelatest(run_benchmark; warmup=true)
        if bench === nothing || sol === nothing
            return BenchmarkResult("RC Circuit", :failed, "Benchmark returned nothing")
        end
        return BenchmarkResult(
            "RC Circuit", :success,
            median(bench.times) / 1e9, minimum(bench.times) / 1e9, maximum(bench.times) / 1e9,
            bench.memory / 1e6, bench.allocs, length(sol.t), ""
        )
    catch e
        return BenchmarkResult("RC Circuit", :failed, sprint(showerror, e))
    end
end

function run_graetz_benchmark()
    println("Running Graetz Bridge benchmark...")
    try
        include(joinpath(BENCHMARK_DIR, "graetz", "cedarsim", "runme.jl"))
        bench, sol = Base.invokelatest(run_benchmark; warmup=true)
        if bench === nothing || sol === nothing
            return BenchmarkResult("Graetz Bridge", :failed, "Benchmark returned nothing")
        end
        return BenchmarkResult(
            "Graetz Bridge", :success,
            median(bench.times) / 1e9, minimum(bench.times) / 1e9, maximum(bench.times) / 1e9,
            bench.memory / 1e6, bench.allocs, length(sol.t), ""
        )
    catch e
        return BenchmarkResult("Graetz Bridge", :failed, sprint(showerror, e))
    end
end

function run_mul_benchmark()
    println("Running Voltage Multiplier benchmark...")
    try
        include(joinpath(BENCHMARK_DIR, "mul", "cedarsim", "runme.jl"))
        bench, sol = Base.invokelatest(run_benchmark; warmup=true)
        if bench === nothing || sol === nothing
            return BenchmarkResult("Voltage Multiplier", :failed, "Benchmark returned nothing")
        end
        return BenchmarkResult(
            "Voltage Multiplier", :success,
            median(bench.times) / 1e9, minimum(bench.times) / 1e9, maximum(bench.times) / 1e9,
            bench.memory / 1e6, bench.allocs, length(sol.t), ""
        )
    catch e
        return BenchmarkResult("Voltage Multiplier", :failed, sprint(showerror, e))
    end
end

function run_ring_benchmark()
    println("Running Ring Oscillator benchmark...")
    try
        include(joinpath(BENCHMARK_DIR, "ring", "cedarsim", "runme.jl"))
        bench, sol = Base.invokelatest(run_benchmark; warmup=true)
        if bench === nothing || sol === nothing
            return BenchmarkResult("Ring Oscillator", :failed, "Benchmark returned nothing")
        end
        return BenchmarkResult(
            "Ring Oscillator", :success,
            median(bench.times) / 1e9, minimum(bench.times) / 1e9, maximum(bench.times) / 1e9,
            bench.memory / 1e6, bench.allocs, length(sol.t), ""
        )
    catch e
        return BenchmarkResult("Ring Oscillator", :failed, sprint(showerror, e))
    end
end

function run_c6288_benchmark()
    println("Running C6288 Multiplier benchmark...")
    try
        include(joinpath(BENCHMARK_DIR, "c6288", "cedarsim", "runme.jl"))
        bench, sol = Base.invokelatest(run_benchmark; warmup=true)
        if bench === nothing || sol === nothing
            return BenchmarkResult("C6288 Multiplier", :failed, "Benchmark returned nothing")
        end
        return BenchmarkResult(
            "C6288 Multiplier", :success,
            median(bench.times) / 1e9, minimum(bench.times) / 1e9, maximum(bench.times) / 1e9,
            bench.memory / 1e6, bench.allocs, length(sol.t), ""
        )
    catch e
        return BenchmarkResult("C6288 Multiplier", :failed, sprint(showerror, e))
    end
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

    push!(results, run_rc_benchmark())
    push!(results, run_graetz_benchmark())
    push!(results, run_mul_benchmark())

    # PSP103-based benchmarks are disabled until PSP103 model initialization is fixed
    # See benchmarks/vacask/cedarsim/STATUS.md for details
    # push!(results, run_ring_benchmark())
    # push!(results, run_c6288_benchmark())
    push!(results, BenchmarkResult("Ring Oscillator", :skipped, "PSP103 model initialization not yet supported"))
    push!(results, BenchmarkResult("C6288 Multiplier", :skipped, "PSP103 model initialization not yet supported"))

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
