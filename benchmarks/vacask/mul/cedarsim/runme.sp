Diode cascade (Voltage multiplier)
* Note: Using phase=90 to start at peak (dV/dt=0) for better Newton convergence
* at t=0. The cascaded diode topology has convergence issues with phase=0.

vs a 0 dc=0 sin 0 50 100k 0 0 90
r1 a 1 0.01
c1 1 2 100n
xd1 0 1 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
c2 0 10 100n
xd2 1 10 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
c3 1 2 100n
xd3 10 2 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
c4 10 20 100n
xd4 2 20 sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45

.end
