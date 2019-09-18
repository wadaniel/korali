#!/usr/bin/env python

# Minus g09

def model(d):
  npar = 10
  res = 0.0
  for i in range(npar):
    if( i == 0 or i == 1 or i == 3 or i == 6):
        res += pow( 10, 6.0*i/npar) * round(d[i]) * round(d[i])
    else:
        res += pow( 10, 6.0*i/npar) * d[i] * d[i]

  d.setResult(-res);
