#!/usr/bin/env python

# This is the negative square -(x^2)
def model(x):
  v = x["Parameters"][0]
  x["Evaluation"] = -0.5*v*v

# This is the negative square -(x^2)
def model_with_gradient(x):
  v = x["Parameters"][0]
  x["Evaluation"] = -0.5*v*v
  x["Gradient Evaluation"] = -v
