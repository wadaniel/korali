#ifndef _KORALI_CAUCHY_H_
#define _KORALI_CAUCHY_H_

#include "parameters/base/base.h"

namespace Korali::Parameter
{

class Cauchy : public Korali::Parameter::Base
{
 private:
  double _aux;
  double _loc;
  double _scale;

 public:

  Cauchy(nlohmann::json& js, size_t seed);
  Cauchy(double scale, double loc, size_t seed = 0);
  ~Cauchy();

  double getDensity(double x) override;
  double getDensityLog(double x) override;
  double getRandomNumber() override;

  void printDetails() override;
  void initialize();

  // Serialization Methods
  nlohmann::json getConfiguration() override;
  void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_CAUCHY_H_
