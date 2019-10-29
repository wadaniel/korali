#!/usr/bin/env python

# log Gaussian with means +- 2.0

def lgaussian2d( s ):
  x0 = s["Parameters"][0]
  x1 = s["Parameters"][1]
  r = -0.5*( (x0 - 2.0)**2 + (x1 + 2.0)**2 )
  print("x0: {0} x1: {1} r: {2}".format(x0, x1, r))
  s["Evaluation"] = r
