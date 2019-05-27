#!/usr/bin/env python

def evaluateModel( s ):
  x = s.getParameter(0)
  r = -x*x
  s.addResult(r)

