#!/usr/bin/env python

# log Gaussian with means +- 2.0

def lgaussian2d( s ):
  x0 = s["Parameters"][0]
  x1 = s["Parameters"][1]
  r = -0.5*( (x0 - 2.0)**2 + (x1 + 2.0)**2 )
  s["Evaluation"] = r

def lgaussian2dCustom( s ):
  x0 = s["Parameters"][0]
  x1 = s["Parameters"][1]
  r = -0.5*( (x0 - 2.0)**2 + (x1 + 2.0)**2 )
  s["logLikelihood"] = r


