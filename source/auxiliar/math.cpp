#include "math.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>

// Computes: log sum_{i=1}^N x_i
// using the log-sum-exp trick:
// https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
// Input: vector of log(x_i)
double korali::logSumExp(const std::vector<double>& logValues )
{
  double maxLogValues = *std::max_element(std::begin(logValues), std::end(logValues));

  if( std::isinf(maxLogValues)==true ){
    if( maxLogValues<0 )  return -korali::Inf;
    else                  return  korali::Inf;
  }

  double sumExpValues = 0.0;
  for (size_t i = 0; i < logValues.size(); i++) sumExpValues += exp(logValues[i] - maxLogValues);

  return maxLogValues + log(sumExpValues);
}


bool korali::isanynan( const std::vector<double>& x )
{
  for( size_t j=0; j<x.size(); j++)
    if( std::isnan( x[j] ) == true ) return true;

  return false;
}


double korali::vectorNorm( const std::vector<double>& x )
{
  double norm=0.;
  for( size_t i=0; i<x.size(); i++ ) norm += x[i]*x[i];
  return sqrt(norm);
}


double korali::vectorDistance( const std::vector<double>& x, const std::vector<double>& y )
{
  double norm=0.;
  for( size_t i=0; i<x.size(); i++ ){
    double z = x[i]-y[i];
    norm += z*z;
  }
  return sqrt(norm);
}


std::string korali::getTimestamp()
{
  time_t rawtime;
  time (&rawtime);
  std::string curTime(ctime(&rawtime));
  return curTime.substr(0, curTime.size()-1);
}


size_t korali::getTimehash()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
