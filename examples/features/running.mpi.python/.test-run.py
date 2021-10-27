#! /usr/bin/env python3
from subprocess import call

r = call(["mpirun", "-n", "9", "python3", "./run-cmaes.py"])
if r!=0:
  exit(r)

exit(0)
