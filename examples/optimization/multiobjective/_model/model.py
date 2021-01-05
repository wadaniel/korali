#!/usr/bin/env python
import numpy as np

# multi dimensional problem  with two objectives (rosenbrock and sphere)
def negative_rosenbrock_and_sphere(p):
    x = p["Parameters"]
    dim = len(x)
    resOne = 0.
    for i in range(dim-1):
        resOne += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
 
    resTwo = 0.
    for i in range(dim):
        resTwo += x[i]**2

    p["F(x)"] = [-resOne, -resTwo]


# multi dimensional problem  with three objectives (rosenbrock and two spheres)
def negative_rosenbrock_and_two_spheres(p):
    x = p["Parameters"]
    dim = len(x)
    resOne = 0.
    for i in range(dim-1):
        resOne += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
 
    resTwo = 0.
    for i in range(dim):
        resTwo += x[i]**2
 
    resThree = 0.
    for i in range(dim):
        resThree += (x[i]-2)**2

    p["F(x)"] = [-resOne, -resTwo, -resThree]
