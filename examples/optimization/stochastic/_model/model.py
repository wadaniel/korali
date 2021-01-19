#!/usr/bin/env python
import numpy as np

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
    grad = [0.]*dim
    for i in range(dim-1):
        res += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
        grad[i] += 2.*(1-x[i]) + 200.*(x[i+1]-x[i])
        grad[i+1] -= 200.*(x[i+1]-x[i])

    p["F(x)"] = -res
    p["Gradient"] = grad

# multi dimensional problem (ackley)
def negative_ackley(p):
    x = p["Parameters"]
    a = 20.
    b = 0.2
    c = 2.*np.pi
    dim = len(x)

    sum1 = 0.
    sum2 = 0.
    for i in range(dim):
        sum1 += x[i]*x[i]
        sum2 += np.cos(c*x[i])

    sum1 /= dim
    sum2 /= dim
    r1 = a*np.exp(-b*np.sqrt(sum1))
    r2 = np.exp(sum2)

    p["F(x)"] = r1 + r2 - a - np.exp(1)
    
    grad = [0.]*dim
    for i in range(dim):
      grad[i] = r1*-1*b*0.5/np.sqrt(sum1)*1.0/dim*2.0*x[i]
      grad[i] -= r2*1.0/dim*np.sin(c*x[i])*c

    p["Gradient"] = grad
