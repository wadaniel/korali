#! /usr/bin/env python3
from subprocess import call

r = call(["make", "-j4"])
if r!=0:
  exit(r)

r = call(["mpirun", "-n", "9", "./run-cmaes", "4"])
if r!=0:
  exit(r)

r = call(["mpirun", "-n", "9", "./run-tmcmc", "4"])
if r!=0:
  exit(r)

exit(0)
