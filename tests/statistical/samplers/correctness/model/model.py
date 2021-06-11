#!/usr/bin/env python


# This is the negative square -0.5*(x^2)
# Proportional to log Normal with 0 mean and 1 var
def model(s):
  v = s["Parameters"][0]
  r = -0.5 * v * v
  s["F(x)"] = r
  s["logP(x)"] = r
  s["logLikelihood"] = r
  s["logLikelihood Gradient"] = [0.0]
  s["grad(logP(x))"] = [-v]
  s["H(logP(x))"] = [[-1.0]]

def model2(s):
  x = s["Parameters"][0]
  y = s["Parameters"][1]
  r = -0.5 * x * x -0.5 * y * y
  s["F(x)"] = r
  s["logP(x)"] = r
  s["logLikelihood"] = r
  s["logLikelihood Gradient"] = [0.0]
  s["grad(logP(x))"] = [-x-y]
  s["H(logP(x))"] = [[-1.0]]