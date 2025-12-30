C6288 16x16 multiplier with PSP103 MOSFETs

* NMOS wrapper - uses PSP103VA directly (uppercase params to match VA model)
.subckt nmos d g s b w=1u l=0.2u ld=0.5u ls=0.5u
  xm d g s b PSP103VA TYPE=1 W={w} L={l}
.ends

* PMOS wrapper - uses PSP103VA directly with TYPE=-1
.subckt pmos d g s b w=1u l=0.2u ld=0.5u ls=0.5u
  xm d g s b PSP103VA TYPE=-1 W={w} L={l}
.ends

.include "multiplier.inc"

vdd vdd 0 1.2
vss vss 0 0

x1  a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15
+   b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15
+   p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15
+   p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31
+ c6288

.subckt v01 out ref
  vdrv int 0 pulse 0 1.2 0.1n 0.1n
  rdrv int out r=1
.ends

  xa0  a0  0 v01
  xa1  a1  0 v01
  xa2  a2  0 v01
  xa3  a3  0 v01
  xa4  a4  0 v01
  xa5  a5  0 v01
  xa6  a6  0 v01
  xa7  a7  0 v01
  xa8  a8  0 v01
  xa9  a9  0 v01
  xa10 a10 0 v01
  xa11 a11 0 v01
  xa12 a12 0 v01
  xa13 a13 0 v01
  xa14 a14 0 v01
  xa15 a15 0 v01

  xvb0  b0  0 v01
  xvb1  b1  0 v01
  xvb2  b2  0 v01
  xvb3  b3  0 v01
  xvb4  b4  0 v01
  xvb5  b5  0 v01
  xvb6  b6  0 v01
  xvb7  b7  0 v01
  xvb8  b8  0 v01
  xvb9  b9  0 v01
  xvb10 b10 0 v01
  xvb11 b11 0 v01
  xvb12 b12 0 v01
  xvb13 b13 0 v01
  xvb14 b14 0 v01
  xvb15 b15 0 v01

.end
