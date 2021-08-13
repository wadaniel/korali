#!/bin/bash

# Defaults for Options
BPDX=${BPDX:-2}
BPDY=${BPDY:-1}
LEVELS=${LEVELS:-8}
RTOL=${RTOL-2}
CTOL=${CTOL-1}
EXTENT=${EXTENT:-4}
CFL=${CFL:-0.4}
PT=${PT:-1e-5}
PTR=${PTR:-1e-2}
PR=${PR:-0}

# Defaults for fish
LENGTH=${LENGTH:-0.2}
PERIOD=${PERIOD:-1}
PID=${PID:-0}
PIDPOS=${PIDPOS:-0}

# L=0.1 stefanfish Re=1'000 <-> NU=0.00004
NU=${NU:-0.00004}

OPTIONS="-bpdx $BPDX -bpdy $BPDY -levelMax $LEVELS -levelStart 5  -Rtol $RTOL -Ctol $CTOL -extent $EXTENT -CFL $CFL -poissonTol $PT -poissonTolRel $PTR -maxPoissonRestarts $PR -bAdaptChiGradient 0 -tdump 0.1 -nu $NU -tend 0 -muteAll 0 -verbose 0"

### for L=0.2 and extentx=extenty=2, 4 swimmers
# OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# "

### for L=0.2 and extentx=extenty=2, 9 swimmers
OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=0.80 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.20 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
"

### for L=0.2 and extentx=extenty=2, 16 swimmers
# OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=0.80 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.20 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.70 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.30 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=0.80 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.20 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.40 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# "

### for L=0.2 and extentx=extenty=2, 25 swimmers
# OBJECTS="stefanfish L=$LENGTH T=$PERIOD xpos=0.60 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=0.90 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=0.80 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.20 ypos=1.20 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.70 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.50 ypos=1.30 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=0.60 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=0.80 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.20 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=1.80 ypos=1.40 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=0.70 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.10 ypos=1.30 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.40 ypos=0.80 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.40 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.40 ypos=1.20 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.70 ypos=0.90 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=2.70 ypos=1.10 bFixed=1 pidpos=$PIDPOS pid=$PID
# stefanfish L=$LENGTH T=$PERIOD xpos=3.00 ypos=1.00 bFixed=1 pidpos=$PIDPOS pid=$PID
# "