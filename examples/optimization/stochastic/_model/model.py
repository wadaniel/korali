#!/usr/bin/env python

# 1-d problem
def model(p):
  x = p["Parameters"][0]
  p["F(x)"] = -0.5 * x * x

# multi dimensional problem (sphere)
def negative_sphere(p):
    x = p["Parameters"]
    dim = len(x)
    res = 0.
    grad = [0.]*dim
    for i in range(dim):
        res += x[i]**2
        grad[i] = -x[i]

    p["F(x)"] = -0.5*res
    p["Gradient"] = grad

# multi dimensional problem (rosenbrock)
def negative_rosenbrock(p):
    x = p["Parameters"]
    dim = len(x)
    res = 0.
    for i in range(dim-1):
        res += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2

    p["F(x)"] = -res
