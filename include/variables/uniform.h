#ifndef _KORALI_VARIABLE_UNIFORM_H_
#define _KORALI_VARIABLE_UNIFORM_H_

#include "variables/base.h"

namespace Korali { namespace Variable {

class Uniform : public Base
{
 private:
  double _aux;
  double _minimum;
  double _maximum;

 public:

  Uniform(nlohmann::json& js, size_t seed);
  Uniform(double _minimum, double _maximum, size_t seed = 0);
  ~Uniform();

  double getDensity(double x) override;
  double getLogDensity(double x) override;
  double getRandomNumber() override;

  // Constructor / Destructor
  void printDetails() override;
  void initialize() override;

  // Serialization Methods
  nlohmann::json getConfiguration() override;
  void setConfiguration(nlohmann::json& js) override;
};

} } // namespace Korali::Variable

#endif // _KORALI_DISTRIBUTION_H_
