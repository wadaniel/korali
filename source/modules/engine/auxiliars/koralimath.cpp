#include "koralimath.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>

// Computes: log sum_{i=1}^N x_i
// using the log-sum-exp trick:
// https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
// Input: vector of log(x_i)
double Korali::logSumExp( std::vector<double> logValues )
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


std::string Korali::getTimestamp()
{
 time_t rawtime;
 time (&rawtime);
 std::string curTime(ctime(&rawtime));
 return curTime.substr(0, curTime.size()-1);
}

size_t getTimehash()
{
 return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

