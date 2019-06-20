#include "rosenbrock.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Minus g09
void g09(std::vector<double>& pars, std::vector<double>& result)
{

  size_t nPars = pars.size();
  if (nPars != 7) { printf("g09: num pars must be 7\n"); exit(-1); }

  double res = pow(pars[0] - 10.0,2) + 5.0 * pow(pars[1] - 12.0,2)
              + pow(pars[2], 4.) + 3.0 * pow(pars[3] - 11.0,2)
              + 10.0 * pow(pars[4], 6) + 7.0 * pow(pars[5],2) + pow(pars[6], 4.)
              - 4.0 * pars[5] * pars[6] - 10.0 * pars[5] - 8.0 * pars[6];

  result.push_back(-res);
}

double g1( std::vector<double>& x )
{
   return  -127.0 + 2 * x[0] * x[0] + 3.0 * pow(x[1], 4) + x[2]
       + 4.0 * x[3] * x[3] + 5.0 * x[4];
}

double g2( std::vector<double>& x )
{
   return -282.0 + 7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2] * x[2]
       + x[3] - x[4];
}

double g3( std::vector<double>& x )
{
   return -196.0 + 23.0 * x[0] + x[1] * x[1] + 6.0 * x[5] * x[5]
       - 8.0 * x[6];
}

double g4( std::vector<double>& x )
{
  return  4.0 * x[0] * x[0] + x[1] * x[1] - 3.0 * x[0] * x[1]
      + 2.0 * x[2] * x[2] + 5.0 * x[5] - 11.0 * x[6];
}

// Minus g09, including constraints
void g09_mod(std::vector<double>& pars, std::vector<double>& result)
{
    if( g1( pars ) > 0)      result.push_back(-1e9);
    else if ( g2(pars) > 0 ) result.push_back(-1e9);
    else if ( g3(pars) > 0 ) result.push_back(-1e9);
    else if ( g4(pars) > 0 ) result.push_back(-1e9);
    else                          g09(pars, result);
}
