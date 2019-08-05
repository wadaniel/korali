#!/usr/bin/env python

# This is a linear regression model with two params (slope and intercept)

def model( s, x ):
 for i in range(len(x)):
  th0 = s.getVariable(0)
  th1 = s.getVariable(1)
  r  =  th0*x[i] + th1
  s.addReferenceEvaluation(r)

def getReferenceData():
 y=[]
 y.append(3.2069);
 y.append(4.1454);
 y.append(4.9393);
 y.append(6.0588);
 y.append(6.8425);
 return y

def getReferencePoints():
 x=[] 
 x.append(1.0);
 x.append(2.0);
 x.append(3.0);
 x.append(4.0);
 x.append(5.0);
 return x
