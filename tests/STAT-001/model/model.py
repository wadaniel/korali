#!/usr/bin/env python

# This is the negative square -0.5*(x^2)
# Proportional to log Normal with 0 mean and 1 var
def model( s ):
  r = -0.5*s[0]*s[0]
  s.setResult(r)

