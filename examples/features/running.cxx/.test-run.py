#! /usr/bin/env python3
from subprocess import call

r = call(["make", "-j4"])
if r!=0:
  exit(r)

r = call(["./run-cmaes"])
if r!=0:
  exit(r)

r = call(["./run-cmaes-direct"])
if r!=0:
  exit(r)

r = call(["./run-tmcmc"])
if r!=0:
  exit(r)

exit(0)
