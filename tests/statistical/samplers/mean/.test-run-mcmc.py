#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "run-mcmc-exponential.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-mcmc-gaussian.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-mcmc-gaussian5d.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-mcmc-laplace.py"])
if r!=0:
  exit(r)

exit(0)
