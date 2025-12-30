# Simulators

Benchmarks were performed with [Xyce 7.9](https://xyce.sandia.gov/), [Gnucap 20240220](http://gnucap.org), [Ngspice pre-master 45](https://ngspice.sourceforge.io/), and [VACASK](https://codeberg.org/arpadbuermen/VACASK). 

All the tests measure single threaded performance. Xyce, Ngspice, and VACASK all used KLU as the linear solver. Gnucap used its own solver. 

Xyce 7.9 was compiled without parallel computing support. Trilinos 16.0.0 was used. Since it has an obsession with taking time measurements this can result in significant slowdowns for small circuits. To disable time measurements the file `Xyce/src/UtilityPKG/N_UTL_StatMetricTraits.C` was modified by replacing `return Xyce::cpu_time();` and `return Xyce::wall_time();` with `return 0;` (thank goes to Eric Keiter for suggesting this). This modified version is named Xyce-fast. 

Gnucap was compiled with default options. Version 20240220 was the last stable release at the time of testing. We had convergence problems with version 20250525-dev on the graetz testbench so we stuck with the stable release. 

Ngspice was configured with 
```
configure --with-x --with-readline=yes --enable-osdi --disable-debug --disable-openmp
```
and then compiled. 

VACASK (version 0.2.2-80-ge15fc2a1) was compiled with default options. All benchmarks were run under Debian Linux. 

# Methodology

Each case was run 6 times. The first run was not timed. We assumed that the caches are filled after the first run so the performance variability in subsequent runs is smaller. No results were stored in files to prevent disk IO operations from affecting the results. Timing includes the whole run - loading of the input files, parsing, elaboration, and simulation. The tables list the average values across 5 runs. The standard deviation of timings was below 2% of the average. You can review the methodology by looking at [benchmark.py](benchmark.py). 

Ngspice, Xyce and Gnucap were using builtin models of circuit elements. VACASK used Ngspice models converted into Verilog-A by [Verilog-A Distiller](https://codeberg.org/arpadbuermen/VADistiller). The simplified noise model was used and output variables that induce extra internal nodes were left out (model variant `sn`) to make models as close as possible to the ones provided by Ngspice in terms of complexity. The CMOS test problems used the PSP103.4 MOSFET model. 

For each test problem the number of timepoints and the number of Newton-Raphson (NR) iterations (residual evaluations for Xyce) is listed. In some cases the number of rejected timepoints is also given. All benchmarks were run on a computer with an AMD Threadripper 7970 processor. 

# Results

## RC circuit excited by a pulse train (rc)
This is a simple circuit with 2 elements (resistor, capacitor) excited by a voltage pulse train. The timestep was forced artificially to a small value so that roughly 1000000 time steps are computed. This is a linear circuit with no rejected timepoints. 

|Simulator  |Time (s)        |Timepoints |Iterations |
|-----------|----------------|-----------|-----------|
|Xyce       | 9.39           |1011527    |2029571    |
|Xyce-fast  | 4.12           |1011527    |2029571    |
|Gnucap     | 8.54           |1006982    |2018061    |
|Ngspice    | 1.31           |1006013    |2012031    |
|VACASK     | 0.94           |1005006    |2010014    |

## Full wave rectifier with smoothing and load (graetz)
A full wave rectifier (4 diodes) with a capacitor and a resistor as load excited by a sinusoidal voltage. Two 1GOhm resistors are used for setting up a DC path to ground. The timestep was forced artificially to a small value so that roughly 1000000 time steps are computed. Second order Gear integration was used. The diode model has transit time (tt) set to 0 because Gnucap exhibits convergence problem when reverse recovery effects are modelled. 

|Simulator  |Time (s)        |Timepoints |Rejected |Iterations |
|-----------|----------------|-----------|---------|-----------|
|Xyce       |10.60           |1000002    |0        |2000014    |
|Xyce-fast  | 5.34           |1000002    |0        |2000014    |
|Gnucap     |15.16           |1000026    |12       |3459503    |
|Ngspice    | 2.21           |1000008    |0        |2000024    |
|VACASK     | 1.89           |1000003    |0        |2000277    |

## Diode voltage multiplier (mul)
A voltage multiplier (4 diodes, 4 capacitors) with a series resistor at its input excited by a sinusoidal voltage. The timestep was forced artificially to a small value so that roughly 500000 time steps are computed. Second order Gear integration was used. The diode model has transit time (tt) set to 0 because Gnucap exhibits convergence problem when reverse recovery effects are modelled. 

|Simulator  |Time (s)        |Timepoints |Rejected |Iterations |
|-----------|----------------|-----------|---------|-----------|
|Xyce       | 5.51           |502341     |1270     |1041833    |
|Xyce-fast  | 2.78           |502341     |1270     |1041833    |
|Gnucap     | 9.94           |520797     |739      |2300992    |
|Ngspice    | 1.16           |500467     |957      |1019733    |
|VACASK     | 0.97           |500056     |3        |1001233    |

## 9 stage CMOS ring oscillator (ring)
This is a ring oscillator with 9 CMOS inverters (18 transistors) powered by 1.2V. The timestep was limited to 50ps. Xyce `timeint reltol` option was set to 5e-3 to make the number of computed timepoints roughly equal to that of VACASK. 

|Simulator  |Time (s)        |Timepoints |Rejected |Iterations |
|-----------|----------------|-----------|---------|-----------|
|Xyce       | 3.33           |27310      |0        |95462      |
|Xyce-fast  | 3.10           |27310      |0        |95462      |
|Ngspice    | 1.60           |20556      |1037     |80018      |
|VACASK     | 1.18           |26066      |0        |81875      |

## 16x16 CMOS multiplier (c6288)
A medium size digital circuit with 10112 transistors and 25380 nodes. It computes the product of 0xFFFF with itself. The timestep was not limited. The simulation is purely analog (i.e. transistor-level). Due to its size we do not test it with Xyce-fast since the timing overheads are small compared to simulation time. VACASK `tran_lteratio` option was set to 1.5 to roughly match the number of computed timepoints to those of Ngspice. For the same reason Xyce `timeint reltol` option was set 2.5e-4. 

|Simulator  |Time (s)        |Timepoints |Rejected |Iterations |
|-----------|----------------|-----------|---------|-----------|
|Xyce       | 151.57         |1013       |37       |3559       |
|Ngspice    |  71.81         |1020       |1        |3474       |
|VACASK     |  57.98         |1021       |7        |3487       |

### Effect of residual tolerance checks
Residual tolerance check (rtc) makes sure KCL equations are satisfied within a prescribed tolerance. It can induce extra NR iterations and slow down the simulation, but on the other hand makes the simulation results more accurate. By default residual tolerance check is enabled (option `nr_residualcheck` set to 1). This check is not performed in Ngspice. The table lists the results of the c6288 benchmark. 

|Simulator            |Time (s)        |Timepoints |Rejected |Iterations |
|---------------------|----------------|-----------|---------|-----------|
|Ngspice              | 71.81          |1020       |1        |3474       |
|VACASK, rtc (default)| 57.98          |1021       |7        |3487       |
|VACASK, no rtc       | 48.34          |1024       |8        |3090       |

When residual tolerance check is disabled VACASK is faster, but the results are less accurate. 

### Effect of continuation bypass 
Continuation bypass (cb) is a technique where the simulator avoids the evaluation of the circuit in the first iteration of the NR algorithm if the result of the previous run is used as the starting point, e.g. in 
* a transient analysis after an accepted timepoint, 
* a continuation method after a converged NR run for a slightly different homotopy parameter value, and 
* a parametric sweep after a converged NR run for a slightly different swept parameter value. 

Since in continuation mode NR algorithm requires only a few iterations to converge, omitting the circuit evaluation in the first iteration saves a lot of time. Continuation bypass cannot be applied for circuit elements that are not time-invariant (most device models are time invariant). It is enabled by default (option `nr_contbypass` set to 1). 

The following table was obtained with residual tolerance check disabled (`nr_residualcheck` set to 0) in both VACASK runs of the c6288 benchmark. Disabling residual tolerance check and continuation bypass brings VACASK simulation algorithms roughly to the level of the ones in Ngspice with elements bypass disabled (--enable-nobypass configuration option). The nobypass option has no effect on the c6288 benchmark because element bypass in Ngspice is not implemented for OSDI devices. 

|Simulator            |Time (s)        |Timepoints |Rejected |Iterations |Time/iter (ms)|
|---------------------|----------------|-----------|---------|-----------|--------------|
|Ngspice              | 71.81          |1020       |1        |3474       |20.7          |
|Ngspice, nobypass    | 71.80          |1020       |1        |3474       |20.7          |
|VACASK, no cb        | 63.19          |1024       |8        |3090       |20.4          |
|VACASK, cb (default) | 48.34          |1024       |8        |3090       |15.6          |

Disabling continuation bypass slows down the simulation. 

### Effect of inactive element bypass 
Evaluation of inactive elements can be bypassed if the input quantities (in most cases voltages at the element's terminals) do not change significantly and the residuals contributed by the element have stabilized within a certain tolerance. When the input quantities change sufficiently the element is evaluated again. This technique is enabled by default in SPICE. To disable it you must configure Ngspice with the `--nobypass` option. Because it can reduce the accuracy, cause convergence problems, and sometimes even slow down the simulation (checking if the element has converged takes time) it is by default disabled in VACASK (`nr_bypass` is set to 0). 

The `nr_convtol` and `nr_bypasstol` specify the convergence tolerances for the residuals and the input quantities (relative to the NR convergence tolerance). Inactive element bypass (ieb) can be enabled by setting the `nr_bypass` option to 1. When enabled it is recommended to disable residual tolerance checks by setting `nr_residualcheck` to 0 because they often cancel out the gains obtained due to inactive element bypass. When `nr_residualcheck` is enabled full precision is forced (inactive element bypass is disabled) once the convergence criteria for the unknowns are met, but the convergence criteria for the residuals are not. 

The following table was obtained on the c6288 benchmark with residual tolerance check disabled (`nr_residualcheck` set to 0) in both VACASK runs. `nr_convtol` and `nr_bypasstol` were both set to 1. 

|Simulator                |Time (s)        |Timepoints |Rejected |Iterations |
|-------------------------|----------------|-----------|---------|-----------|
|Ngspice                  | 71.81          |1020       |1        |3474       |
|VACASK, no ieb (default) | 48.34          |1024       |8        |3090       |
|VACASK, ieb              | 46.74          |1024       |10       |3101       |

Inactive element bypass is effective only for large circuits when a significant part of elements is inactive most of the time. 