#!/usr/bin/env python
import numpy as np

# Helper
def assertclose(expected, value, atol):
    assert np.isclose(expected, value, atol = atol), "Value {0} "\
        "deviates from expected value {1}".format(value, expected)

# Function to minimize
def minmodel1(p):
  x = p["Parameters"][0]
  p["Evaluation"] = (x-2.0)*(x-2.0)+10 # fopt = 10.0

# Function to minimize
def minmodel2(p):
  x = p["Parameters"][0]
  p["Evaluation"] = (x-2.0)*(x-2.0)-1e9 # fopt = -1e9

# Function to maximize
def maxmodel1(p):
  x = p["Parameters"][0]
  p["Evaluation"] = -(x-2.0)*(x-2.0)-10 # fopt = -10

# Function to maximize
def maxmodel2(p):
  x = p["Parameters"][0]
  p["Evaluation"] = -(x-2.0)*(x-2.0)+1e9 # fopt = 1e9


