#!/bin/bash

NU=${NU:-0.00001} #Re=4000

FACTORY=" StefanFish L=0.2 T=1 xpos=1.00 ypos=1.00 zpos=1.00 bFixFrameOfRef=0 heightProfile=danio widthProfile=stefan bFixToPlanar=1
"
OPTIONS=
OPTIONS+=" -bpdx 2 -bpdy 2 -bpdz 2 -extentx 2.0 -levelMax 6 -levelStart 3 "
OPTIONS+=" -Rtol 10000.00 -Ctol 100.00"
OPTIONS+=" -tdump 0.1 -tend 0 "
OPTIONS+=" -CFL 0.5 -lambda 1e6 -nu ${NU}"
OPTIONS+=" -poissonTol 1e-6 -poissonTolRel 1e-4 "
