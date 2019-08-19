#!/usr/bin/env python

# This is the negative square -0.5*(x^2)
# Proportional to log Normal with 0 mean and 1 var
def model( s ):
  x = s.getVariable(0)
  r = -0.5*x*x
  s.setResult(r)

