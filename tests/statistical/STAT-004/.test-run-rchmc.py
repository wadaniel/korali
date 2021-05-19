#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "run-rchmc-gaussian.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-rchmc-laplace.py"])
if r!=0:
  exit(r)

exit(0)
