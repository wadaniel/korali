#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "run-hmc-gaussian.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-hmc-laplace.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-hmc-nuts-gaussian.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-hmc-nuts-laplace.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-hmc-riemannian-laplace.py"])
if r!=0:
  exit(r)

exit(0)
