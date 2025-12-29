9 stage ring oscillator with PSP103 MOSFETs

* NMOS wrapper - uses PSP103VA directly (uppercase params to match VA model)
.subckt nmos d g s b w=1u l=0.2u ld=0.5u ls=0.5u
  xm d g s b PSP103VA TYPE=1 W={w} L={l}
.ends

* PMOS wrapper - uses PSP103VA directly with TYPE=-1
.subckt pmos d g s b w=1u l=0.2u ld=0.5u ls=0.5u
  xm d g s b PSP103VA TYPE=-1 W={w} L={l}
.ends

* Inverter subcircuit
.subckt inverter in out vdd vss w=1u l=0.2u pfact=2
  xmp out in vdd vdd pmos w={w*pfact} l={l}
  xmn out in vss vss nmos w={w} l={l}
.ends

* Current pulse to kick-start oscillation
i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n

* 9-stage ring oscillator
xu1 1 2 vdd 0 inverter w={10u} l={1u}
xu2 2 3 vdd 0 inverter w={10u} l={1u}
xu3 3 4 vdd 0 inverter w={10u} l={1u}
xu4 4 5 vdd 0 inverter w={10u} l={1u}
xu5 5 6 vdd 0 inverter w={10u} l={1u}
xu6 6 7 vdd 0 inverter w={10u} l={1u}
xu7 7 8 vdd 0 inverter w={10u} l={1u}
xu8 8 9 vdd 0 inverter w={10u} l={1u}
xu9 9 1 vdd 0 inverter w={10u} l={1u}

* Supply voltage
vdd vdd 0 1.2

.end
