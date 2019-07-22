#!/usr/bin/env python

# Minus g09

def model(d):
  pars = d.getVariables();

  res = (pars[0] - 10.0)**2 + 5.0 * (pars[1] - 12.0)**2           \
        + pars[2]**4  + 3.0 * (pars[3] - 11.0)**2                 \
        + 10.0 * pars[4]**6 + 7.0 * pars[5]**2 + pars[6]**4.      \
        - 4.0 * pars[5] * pars[6] - 10.0 * pars[5] - 8.0 * pars[6];

  d.addResult(-res);
