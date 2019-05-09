#!/usr/bin/env python
import math

# Minus Rosenbrock
def m_rosenbrock(x):

    result = 0.0
    for i in range(x.getParameterCount()):
        result += 100.0*pow(x.getParameter(i+1)-x.getParameter(i),2) + pow(1.0 - x.getParameter(i),2)

    x.addResult(-result)
