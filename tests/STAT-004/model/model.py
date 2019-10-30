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


# log exponential with mean 4
def lexponential( s ):
  lam = 1./4.
  x0 = s["Parameters"][0]
  x1 = s["Parameters"][1]
  r = 2*math.log(lam)-lam*x0 - lam*x1
  s["Evaluation"] = r

def lexponentialCustom( s ):
  lam = 1./4.
  x0 = s["Parameters"][0]
  x1 = s["Parameters"][1]
  r = 2*math.log(lam)-lam*x0 - lam*x1
  s["logLikelihood"] = r

# log exponential with mean 4 and scale 1 (var 2)
def llaplace( s ):
    mu = 4.0
    scale = 1.0
    r = -math.log(2.0*scale)-math.abs(x0 - mu)/scale
    s["Evaluation"] = r

def llaplaceCustom( s ):
    mu = 4.0
    scale = 1.0
    r = -math.log(2.0*scale)-math.abs(x0 - mu)/scale
    s["logLikelihood"] = r







