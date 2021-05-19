#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "run-nested-exponential.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-nested-gaussian.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-nested-gaussian5d.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-nested-laplace.py"])
if r!=0:
  exit(r)

exit(0)
