#!/usr/bin/env python

# Minus g09

def g09( pars, result):
  nPars = pars.size();
  if( nPars != 7 )
    print("g09: num pars must be 7\n");
    exit();

  res = (pars[0] - 10.0)**2 + 5.0 * (pars[1] - 12.0)**2           \
        + pars[2]**4  + 3.0 * (pars[3] - 11.0)**2                 \
        + 10.0 * pars[4]**6 + 7.0 * pars[5]**2 + pars[6]**4.      \
        - 4.0 * pars[5] * pars[6] - 10.0 * pars[5] - 8.0 * pars[6];

  result.addResult(-res);



def g1( x, N ):
  return  -127.0 + 2 * x[0] * x[0] + 3.0 * pow(x[1], 4) + x[2] \
           + 4.0 * x[3] * x[3] + 5.0 * x[4];


def g2( x, N ):
   return -282.0 + 7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2] * x[2] \
          + x[3] - x[4];
}

def g3( x, N ):
{
   return -196.0 + 23.0 * x[0] + x[1] * x[1] + 6.0 * x[5] * x[5] \
          - 8.0 * x[6];
}

double g4( x, N ):
{
  return  4.0 * x[0] * x[0] + x[1] * x[1] - 3.0 * x[0] * x[1] \
          + 2.0 * x[2] * x[2] + 5.0 * x[5] - 11.0 * x[6];
}
