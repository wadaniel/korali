#!/usr/bin/env python

# This is the negative square -(x^2)
def model(x):
  v = x["Parameters"][0]
  r = -0.5*v*v
  x["Evaluation"] = r

