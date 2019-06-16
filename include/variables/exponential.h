#ifndef _KORALI_VARIABLE_EXPONENTIAL_H_
#define _KORALI_VARIABLE_EXPONENTIAL_H_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include "variables/base.h"

namespace Korali { namespace Variable {

class Exponential : public Base
{
 private:
	size_t _seed;
	gsl_rng* _range;
  double _loc;
  double _mean;

 public:

  Exponential(nlohmann::json& js, size_t seed);
  Exponential(double mean, double loc, size_t seed = 0);
  ~Exponential();

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

#endif // _KORALI_VARIABLE_EXPONENTIAL_H_
