#ifndef _KORALI_UNIFORM_H_
#define _KORALI_UNIFORM_H_

#include "parameters/base/base.h"

namespace Korali::Parameter
{

class Uniform : public Korali::Parameter::Base
{
 private:
   double _aux;

 public:

	 Uniform(nlohmann::json& js, size_t seed);
	 Uniform(double _lowerBound, double _upperBound, size_t seed = 0);
	 ~Uniform();

  double getDensity(double x);
  double getDensityLog(double x);
  double getRandomNumber();

  // Constructor / Destructor
  void printDetails() override;
  void initialize();

  // Serialization Methods
  nlohmann::json getConfiguration();
  void setConfiguration(nlohmann::json& js);
};

} // namespace Korali

#endif // _KORALI_DISTRIBUTION_H_
