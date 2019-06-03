#ifndef _KORALI_VARIABLE_UNIFORM_H_
#define _KORALI_VARIABLE_UNIFORM_H_

#include "variables/base.h"

namespace Korali { namespace Variable {

class Uniform : public Base
{
 private:
  double _aux;

 public:

  Uniform(nlohmann::json& js, size_t seed);
  Uniform(double _lowerBound, double _upperBound, size_t seed = 0);
  ~Uniform();

  double getDensity(double x) override;
  double getLogDensity(double x) override;
  double getRandomNumber() override;

  // Constructor / Destructor
  void printDetails() override;
  void initialize();

  // Serialization Methods
  nlohmann::json getConfiguration() override;
  void setConfiguration(nlohmann::json& js) override;
};

} } // namespace Korali::Variable

#endif // _KORALI_DISTRIBUTION_H_
