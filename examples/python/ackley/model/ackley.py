#!/usr/bin/env python
import math

def ackley(x):
 a = 20.0
 b = 0.2
 c = 2.*math.pi
 for j in range(70):
  s1 = 0.0
  s2 = 0.0
  for i in range(4):
   s1 += x.getParameter(i)*x.getParameter(i)
   s2 += math.cos(c*x.getParameter(i))

 result = -(-a*math.exp(-b*math.sqrt(s1/4)) - math.exp(s2/4) + a + math.exp(1.))
 x.addResult(result)