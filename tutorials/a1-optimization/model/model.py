#!/usr/bin/env python

# This is the negative square -(x^2)
def model(k):
  x = k[0]
  r = -0.5*x*x
  k.setResult(r)

