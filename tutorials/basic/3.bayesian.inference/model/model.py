#!/usr/bin/env python

# This is a linear regression model with two params (slope and intercept)

import numpy as np


def model( s, x ): 
  a   = s["Parameters"][0]
  b   = s["Parameters"][1]
  sig = s["Parameters"][2]
  
  fval = [ ]
  sdev = [ ]
  for i in range(len(x)):
    fval.append(a*x[i] + b)
    sdev.append(sig)

  s["Reference Evaluations"] = fval
  s["Standard Deviation"]    = sdev

def modelWithDerivatives( s, x ): 
  a   = s["Parameters"][0]
  b   = s["Parameters"][1]
  sig = s["Parameters"][2]
  
  fval = [ ]
  sdev = [ ]
  
  fvalgrad = [ ]
  sdevgrad = [ ]
  for i in range(len(x)):
    fval.append(a*x[i] + b)
    sdev.append(sig)
    
    fvalgrad.append([x[i], 1.0, 0.0])
    sdevgrad.append([0.0, 0.0, 1.0])

  s["Reference Evaluations"] = fval
  s["Standard Deviation"]    = sdev

  s["Gradient Mean"]               = fvalgrad
  s["Gradient Standard Deviation"] = sdevgrad


def getReferenceData():
 y=[]
 y.append(3.21);
 y.append(4.14);
 y.append(4.94);
 y.append(6.06);
 y.append(6.84);
 return y

def getReferencePoints():
 x=[] 
 x.append(1.0);
 x.append(2.0);
 x.append(3.0);
 x.append(4.0);
 x.append(5.0);
 return x
