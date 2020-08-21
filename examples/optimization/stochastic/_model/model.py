#!/usr/bin/env python

# Single function evaluation
def model(p):
  x = p["Parameters"][0]
  p["F(x)"] = -0.5 * x * x

def negative_sphere(p):
    x = p["Parameters"]
    dim = len(x)
    res = 0.
    for i in range(dim):
        res += x[i]**2

    p["F(x)"] = -res

def negative_rosenbrock(p):
    x = p["Parameters"]
    dim = len(x)
    res = 0.
    for i in range(dim-1):
        res += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2

    p["F(x)"] = -res
