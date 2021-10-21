#!/bin/bash

# Defaults for Options
BPDX=${BPDX:-64}
BPDY=${BPDY:-64}
LEVELS=${LEVELS:-1}
RTOL=${RTOL-0.1}
CTOL=${CTOL-0.01}
EXTENT=${EXTENT:-1.4}
CFL=${CFL:-0.22}
# Defaults for Objects
XPOS=${XPOS:-0.2}

YPOS1=${YPOS:-0.6}
YPOS2=${YPOS2:-0.8}

XVEL=${XVEL:-0.15}

MAAXIS=${MAAXIS:-0.0405}
MIAXIS=${MIAXIS:-0.0135}
NU=${NU:-0.0001215}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -tdump 0.1 -nu $NU -tend 0 -muteAll 0 -verbose 1"
OBJECTS="windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS1 bForced=1 bFixed=1 xvel=$XVEL tAccel=0 bBlockAng=0
windmill semiAxisX=$MAAXIS semiAxisY=$MIAXIS xpos=$XPOS ypos=$YPOS2 bForced=1 xvel=$XVEL tAccel=0 bBlockAng=0"