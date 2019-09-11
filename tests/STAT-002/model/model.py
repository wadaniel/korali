#!/usr/bin/env python

# This is the negative square -(x^2)
def model( s ):
  r = -0.5*s[0]*s[0]
  s.setResult(r)

