# Defaults for Options
ENV=${ENV:-HalfCheetah-v4}  # environment
EBRU=${EBRU:-5000}       # experiences between reward updates
DBS=${DBS:-10}          # demonstration batch size
BBS=${BBS:-50}          # background batch size
BSS=${BSS:-500}         # background sample size
EXP=${EXP:-5000000}     # number of experiences
RNN=${RNN:-8}           # reward neural net size
POL=${POL:-Linear}      # demo policy type
DAT=${DAT:-1000}        # number of obs
RUN=${RUN:-100}         # run tag
