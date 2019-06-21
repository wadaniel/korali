#!/usr/bin/env python

def evaluateModel( s, x ):
 for i in range(len(x)):
  th0 = s.getVariable(0)
  th1 = s.getVariable(1)
  r  =  th0*x[i] + th1
  s.addResult(r)

def getReferenceData():
 x=[]; y=[]
 x.append(1.0); y.append(3.2069);
 x.append(2.0); y.append(4.1454);
 x.append(3.0); y.append(4.9393);
 x.append(4.0); y.append(6.0588);
 x.append(5.0); y.append(6.8425);
 return x, y
