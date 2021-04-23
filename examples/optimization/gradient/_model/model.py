#!/usr/bin/env python
import sys
import numpy as np

# negative sphere
def negative_sphere(p):
  X = p["Parameters"]
  gradient = []
  evaluation = 0
  for x in X:
    evaluation += -0.5 * x * x
    gradient.append(-x)

  p["F(x)"] = evaluation
  p["Gradient"] = gradient

# negative rosenbrock
def negative_rosenbrock(p):
    x = p["Parameters"]
    dim = len(x)
    res = 0.
    grad = [0.]*dim
    a = 10
    b = 1
    for i in range(dim-1):
        res -= a*(x[i+1]-x[i]**2)**2+(b-x[i])**2
        grad[i] += 2*(b-x[i]) + 2*a*(x[i+1]-x[i]**2)*2*x[i]
        grad[i+1] -= 2*a*(x[i+1]-x[i]**2)

    p["F(x)"] = res
    p["Gradient"] = grad

# negative himmelblau function
def negative_himmelblau(p):
    x = p["Parameters"]
    if (len(x) != 2): 
        print("Himmelblau function requires two parameter.") 
        sys.exit()

    p["F(x)"] = -(x[0]*x[0] + x[1] - 11)**2 - (x[0]+x[1]*x[1]-7)**2
    
    grad = [0, 0]
    grad[0] = -4*(x[0]*x[0] + x[1] - 11)*x[0] - 2*(x[0]+x[1]*x[1]-7)
    grad[1] = -2*(x[0]*x[0] + x[1] - 11) - 4*(x[0]+x[1]*x[1]-7)*x[1]

    p["Gradient"] = grad
