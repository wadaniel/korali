#!/usr/bin/env python

# This is the negative square -(x^2)
def model(p):
  x = p["Parameters"][0]
  p["Evaluation"] = -0.5*x*x

# This is the negative square -(x^2)
def model_with_gradient(p):
  x = p["Parameters"];
  p["Evaluation"] = 0
  for k in range( len(x) ):
    p["Evaluation"]  += -0.5*x[k]*x[k]
    p["Gradient"][k]["Value"] = -x[k]
