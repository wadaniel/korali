#include "koralimath.hpp"
#include <math.h>

// Computes:
//  log sum_{i=1}^N x_i
// using the log-sum-exp trick:
// https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
//
// Input: vector of log(x_i)
//
double logSumExp( vector<double> logValues )
{
  double maxLogValues =  *max_element(begin(logValues), end(logValues));
  double sumExpValues = 0.0;
  for (auto& v : logValues) sumExpValues += exp( v - maxLogValues );

  return maxLogValues + log(sumExpValues);
}
