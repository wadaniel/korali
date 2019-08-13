#include "koralimath.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

// Computes:
//  log sum_{i=1}^N x_i
// using the log-sum-exp trick:
// https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
//
// Input: vector of log(x_i)
//
double logSumExp( vector<double> logValues )
{
  double maxLogValues =  *std::max_element(begin(logValues), end(logValues));

  if( isinf(maxLogValues)==true ){
    if(maxLogValues<0) return 0;
    else return INFINITY;
  }

  double sumExpValues = 0.0;
  for (auto& v : logValues) sumExpValues += exp( v - maxLogValues );

  return maxLogValues + log(sumExpValues);
}
