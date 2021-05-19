#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "-m", "korali.plotter", "--help"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.plotter", "--dir", "optimization", "--test"])
if r!=0:
  exit(r)

#r = call(["python3", "-m", "korali.plotter", "--dir", "optimization", "--test"])
#if r!=0:
#  exit(r)

exit(0)
