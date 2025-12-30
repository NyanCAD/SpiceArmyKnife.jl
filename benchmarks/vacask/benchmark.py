#!/usr/bin/python3
import sys, os, time, subprocess, shutil, importlib, platform
import numpy as np

# Simple benchmarking framework
# 
# Command line syntax:
# python3 benchmark.py [options] <benchmark case> <path to simulator> [<args>]
# 
# Benchmark case is of the form <case name>/<simulator>. 
#
# Creates a work directory and copies the case into it. 
# Default name of the work directory is rundir. 
# This can be changed with the -wd option. 
# Before running the case the run() function from the prepare.py 
# file is called. A dictionary with the following members is
# passed as the only argument:
# - source_dir .. full path to the problem directory
# - run_openvaf .. a function that invokes OpenVAF (run_openvaf())
#   The function has 3 arguments:
#   - name of the .va file to compile
#   - name of the output .osdi file
#   - optional list of arguments
#
# The run function can set the pre_options and post_options 
# members to a list of options that will be added to the <args> 
# specified on the command line. 
#
# If everything is OK, run() returns True. 
#
# Then initial runs are performed (to fill the caches). 
# The number of initial runs can be set with the -nd option
# (0 by default). 
# Initial runs are followed by timed runs. The number of timed 
# runs can be set with the -n option. 
#
# When all runs are finished a summary is printed and 
# the work directory is removed. If the -k option is 
# specified, the work directory is kept. 
#
# The script searches for OpenVAF-reloaded in the following locations:
# - file specified by the SIM_OPENVAF environmental variable
# - directory specified by the OPENVAF_DIR environmental variable
# - ../../build.VACASK/Debug/simulator
# - ../../build.VACASK/Release/simulator
# - system PATH

# Defaults
count = 1 
ignored_count = 0
workdir = "rundir"
keep = False
print_help = False
openvaf = None
pre_options = []
post_options = []

def cleanup(wd):
    if os.path.isdir(wd):
        shutil.rmtree(wd)

def prepare(wd, subdir):
    cleanup(wd)
    shutil.copytree(subdir, wd)

def find_openvaf():
    openvaf = None

    system = platform.system()
    if system=="Windows":
        openvaf_bin = "openvaf-r.exe"
    else:
        openvaf_bin = "openvaf-r"
    
    # Check SIM_OPENVAF env variable
    if openvaf is None:
        d = os.environ.get("SIM_OPENVAF")
        if d is not None:
            fc = d
            if os.path.isfile(fc):
                openvaf = fc
    
    # Check OPENVAF_DIR env variabe + binary name
    if openvaf is None:
        d = os.environ.get("OPENVAF_DIR")
        if d is not None:
            fc = os.path.join(d, openvaf_bin)
            if os.path.isdir(d) and os.path.isfile(fc):
                openvaf = fc

    # Check ../../build.VACASK/Debug/simulator
    if openvaf is None:
        d = "../../build.VACASK/Debug/simulator"
        fc = os.path.join(d, openvaf_bin)
        if os.path.isdir(d) and os.path.isfile(fc):
            openvaf = fc

    # Check ../../build.VACASK/Release/simulator
    if openvaf is None:
        d = "../../build.VACASK/Release/simulator"
        fc = os.path.join(d, openvaf_bin)
        if os.path.isdir(d) and os.path.isfile(fc):
            openvaf = fc

    # Check system path
    if openvaf is None:
        openvaf = shutil.which(openvaf_bin)

    # Canonical path
    if openvaf is not None:
        openvaf = os.path.realpath(openvaf)

    return openvaf

def run_openvaf(file, output, args=[]):
    cmdline = [ openvaf ]
    if output is not None:
        cmdline += [ "-o", output ]
    cmdline += args
    cmdline += [ file ]
    retval = subprocess.run(cmdline)
    if retval.returncode != 0:
        print("Verilog-A compiler error.")
        return False
    
    return True

ndx = 1
while ndx<len(sys.argv):
    if sys.argv[ndx]=="-n":
        ndx += 1
        count = int(sys.argv[ndx])
    elif sys.argv[ndx]=="-nd":
        ndx += 1
        ignored_count = int(sys.argv[ndx])
    elif sys.argv[ndx]=="-wd":
        ndx += 1
        workdir = sys.argv[ndx]
    elif sys.argv[ndx] == "--keep" or sys.argv[ndx] == "-k":
        keep = True
    elif sys.argv[ndx] == "-h":
        print_help = True
        break
    else:
        break
    ndx += 1
    if ndx>=len(sys.argv):
        print("Need more arguments.\n")
        print_help = True
        break

# Get directory and command
if ndx+2>len(sys.argv):
    print("Need more arguments.\n")
    print_help = True
else:
    subdir = sys.argv[ndx]
    cmd = sys.argv[ndx+1]
    args = sys.argv[ndx+2:]

if print_help:
    print("Usage: python3 benchmark.py [options] directory command [arguments]")
    print()
    print("       Changes to a directory and runs a command with arguments.")
    print("       Prints timing summary.")
    print()
    print("Options")
    print("  -n <num>   .. number of repetitions")
    print("  -nd <num>  .. number of initial runs that are ignored")
    print("                0 by default")
    print("  -wd <name> .. directory where runs are performed")
    print("                rundir by default")
    print("  -k, --keep .. do not delete work directory")
    print("  -h         .. print help")
    sys.exit(0)

# Canonical path to command and workdir
workdir = os.path.realpath(workdir)

# Find binary
cmd_path = shutil.which(cmd)
if cmd_path:
    # Found in path
    cmd = cmd_path

# Use canonical path, assume relative to current directory
cmd = os.path.realpath(cmd)
subdir = os.path.realpath(subdir)

# Find openvaf
openvaf = find_openvaf()

# Sanity check
if not os.path.isfile(cmd):
    print("Executable not found:", cmd)
    sys.exit(1)
if not os.path.isdir(subdir):
    print("Problem not found:", subdir)
    sys.exit(1)

# Report
print("Problem:         ", subdir)
print("Work directory:  ", workdir, "(kept)" if keep else "")
print("Command:         ", cmd)
print("OpenVAF reloaded:", openvaf)

# Prepare work directory
prepare(workdir, subdir)

# Change directory
olddir = os.getcwd()
os.chdir(workdir)

# Preparations for run
if os.path.isfile("./prepare.py"):
    print("Preparing for run...")
    spec = importlib.util.spec_from_file_location("prepare", "./prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    sys.modules["prepare"] = prepare
    spec.loader.exec_module(prepare)

    settings = { 
        "source_dir": subdir, 
        "run_openvaf": run_openvaf 
    }
    retval = prepare.run(settings)
    if not retval:
        print("Preparations failed.")
        sys.exit(1)

    # Get custom options
    pre_options = settings.get("pre_options", [])
    post_options = settings.get("post_options", [])

# Run benchmark
times = []
for cnt in range(count+ignored_count):
    if cnt<ignored_count:
        print("\n\nStartup run", cnt+1, "(ignored)")    
    else:
        print("\n\nRun", cnt-ignored_count+1)

    # Command line
    cmdline = [cmd] + pre_options + args + post_options + ["runme.sim"]
    print(" ".join(cmdline))
    print()

    # Mark time
    t0 = time.perf_counter()

    # Run
    retval = subprocess.run(cmdline)

    # Measure time
    t1 = time.perf_counter()
    dt = t1-t0
    print()
    print("Time:", dt)
    if cnt>=ignored_count:
        times.append(dt)
        
    # Abort on error
    if retval.returncode != 0:
        print("Run failed.")
        os.chdir(olddir)
        if not keep:
            cleanup(workdir)
        sys.exit(1)

print()
print()
print("Summary:")
if ignored_count>0:
    print("  Ignored runs:      ", ignored_count)
    print()
mean = np.mean(np.array(times))
print("  Number of runs:    ", len(times))
if len(times)>1:
    print("  Minimum runtime:   ", min(times))
    print("  Maximum runtime:   ", max(times))
    print("  Average runtime:   ", mean)
else:
    print("  Runtime:           ", mean)
if len(times)>1:
    std = np.std(np.array(times), ddof=1)
    print("  Standard deviation:", std)
    print("            relative:", std/mean)

os.chdir(olddir)
if not keep:
    cleanup(workdir)

