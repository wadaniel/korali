#!/usr/bin/env python

# This is the negative square -(x^2)
def evaluateModel( s ):
  x = s.getVariable(0)
  r = -0.5*x*x
  s.addResult(r)

