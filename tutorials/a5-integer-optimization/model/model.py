#!/usr/bin/env python

# Minus g09

def model(d):
  npar = 10
  res = 0.0
  v = d["Parameters"]
  
  for i in range(npar):
    if( i == 0 or i == 1 or i == 3 or i == 6):
        res += pow( 10, 6.0*i/npar) * round(v[i]) * round(v[i])
    else:
        res += pow( 10, 6.0*i/npar) * v[i] * v[i]

  d["Evaluation"] = -res;
