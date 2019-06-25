#!/usr/bin/env python

# This is the negative square -(x^2)
def evaluateModel( s ):
  x = s.getVariable(0)
  r = -x*x
  s.addResult(r)

