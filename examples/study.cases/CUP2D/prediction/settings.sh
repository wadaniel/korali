#!/bin/bash
# Defaults for Options
BPDX=${BPDX:-32}
BPDY=${BPDY:-16}
NAGENTS=420 # BPDX*BPDY - 2*BPDX - 2*(BPDY-2)
LEVELS=${LEVELS:-1}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-1}
CFL=${CFL:-0.1}
PT=${PT:-1e-10}
PTR=${PTR:-0}
# Defaults for Objects
XPOS=${XPOS:-0.25}
XVEL=${XVEL:-0.2}
RADIUS=${RADIUS:-0.1}
## to compare against "High-resolution simulations of the flow around an impulsively started cylinder using vortex methods" By P. KOUMOUTSAKOST AND A. LEONARD ##
# Re=40 <-> NU=0.01
NU=${NU:-0.01}
OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0 -nu $NU -tend 0 -muteAll 0 -verbose 0 -poissonTol $PT -poissonTolRel $PTR -bAdaptChiGradient 1"
OBJECTS="disk radius=$RADIUS xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=5"
