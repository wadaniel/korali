# Defaults for Options
ENV=${ENV:-HalfCheetah-v4}  # environment
EBRU=${EBRU:-100}       # experiences between reward updates
DBS=${DBS:-10}          # demonstration batch size
BBS=${BBS:-50}          # background batch size
BSS=${BSS:-1000}        # background sample size
EXP=${EXP:-5000000}     # number of experiences
RNN=${RNN:-32}          # reward neural net size
POL=${POL:-Linear}      # demo policy type
RUN=${RUN:-100}         # run tag
