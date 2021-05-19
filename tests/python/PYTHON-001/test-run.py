#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "-m", "korali.plotter", "--help"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plotter", "--dir", "cmaes", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plotter", "--dir", "dea", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plotter", "--dir", "mocmaes", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plotter", "--dir", "lmcmaes", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plotter", "--dir", "tmcmc", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plotter", "--dir", "mcmc", "--test"])
if r!=0:
  exit(r)

exit(0)
