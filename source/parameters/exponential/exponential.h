#ifndef _KORALI_EXPONENTIAL_H_
#define _KORALI_EXPONENTIAL_H_

#include "parameters/base/base.h"

namespace Korali::Parameter
{

class Exponential : public Korali::Parameter::Base
{
 private:
  double _loc;
  double _mean;

 public:

  Exponential(nlohmann::json& js, size_t seed);
  Exponential(double mean, double loc, size_t seed = 0);
  ~Exponential();

  double getDensity(double x);
  double getDensityLog(double x);
  double getRandomNumber();

  void printDetails() override;

  // Serialization Methods
  nlohmann::json getConfiguration();
  void setConfiguration(nlohmann::json& js);
};

} // namespace Korali

#endif // _KORALI_EXPONENTIAL_H_
