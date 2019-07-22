#!/usr/bin/env python

# Minus g09

def model(d):
  pars = d.getVariables();

  npar = 10
  res = 0.0
  for i in range(npar):
    if( i == 0 or i == 1 or i == 3 or i == 6):
        res += pow( 10, 6.0*i/npar) * round(pars[i]) * round(pars[i])
    else:
        res += pow( 10, 6.0*i/npar) * pars[i] * pars[i]

  d.addResult(-res);
