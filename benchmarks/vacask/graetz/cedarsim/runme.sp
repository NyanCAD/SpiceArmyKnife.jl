Full-wave rectifier with smoothing and load

vs inp inn 0 sin 0.0 20 50.0

xd1 inp outp sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
xd2 outn inp sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
xd3 inn outp sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
xd4 outn inn sp_diode is=76.9p rs=42.0m cjo=26.5p m=0.333 n=1.45
cl outp outn 100u
rl outp outn 1k
rgnd1 inn 0 1meg
rgnd2 outn 0 1meg

.end
