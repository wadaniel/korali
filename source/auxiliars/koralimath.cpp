#include "koralimath.hpp"
#include <cmath>
#include <algorithm>


// Computes:
//  log sum_{i=1}^N x_i
// using the log-sum-exp trick:
// https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
//
// Input: vector of log(x_i)
//
double logSumExp( std::vector<double> logValues )
{
  double maxLogValues =  *std::max_element(std::begin(logValues), std::end(logValues));

  if( std::isinf(maxLogValues)==true ){
    if( maxLogValues<0 )  return -Korali::Inf;
    else                  return  Korali::Inf;
  }

  double sumExpValues = 0.0;
  for (auto& v : logValues) sumExpValues += exp( v - maxLogValues );

  return maxLogValues + log(sumExpValues);
}
