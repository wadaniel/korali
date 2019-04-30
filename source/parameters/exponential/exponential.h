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

  double getDensity(double x) override;
  double getDensityLog(double x) override;
  double getRandomNumber() override;

  void printDetails() override;

  // Serialization Methods
  nlohmann::json getConfiguration() override;
  void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_EXPONENTIAL_H_
