#ifndef _KORALI_VARIABLE_CAUCHY_H_
#define _KORALI_VARIABLE_CAUCHY_H_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include "variables/base.h"

namespace Korali { namespace Variable {

class Cauchy : public Base
{
 private:
	size_t _seed;
	gsl_rng* _range;
  double _aux;
  double _loc;
  double _scale;

 public:

  Cauchy(nlohmann::json& js, size_t seed);
  Cauchy(double scale, double loc, size_t seed = 0);
  ~Cauchy();

  double getDensity(double x) override;
  double getLogDensity(double x) override;
  double getRandomNumber() override;

  void printDetails() override;
  void initialize() override;

  // Serialization Methods
  nlohmann::json getConfiguration() override;
  void setConfiguration(nlohmann::json& js) override;
};

} } // namespace Korali::Variable

#endif // _KORALI_VARIABLE_CAUCHY_H_
