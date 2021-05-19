#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "run-tmcmc-exponential.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-tmcmc-gaussian.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-tmcmc-gaussian5d.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-tmcmc-laplace.py"])
if r!=0:
  exit(r)

exit(0)
