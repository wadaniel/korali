#ifndef _KORALI_VARIABLE_GAUSSIAN_H_
#define _KORALI_VARIABLE_GAUSSIAN_H_

#include "variables/base.h"

namespace Korali { namespace Variable {

class Gaussian : public Base
{
 private:
  double _mean;
  double _sigma;
  double _aux;

 public:

  // Constructor / Destructor
  Gaussian(nlohmann::json& js, size_t seed = 0);
  Gaussian(double mean, double sigma, size_t seed = 0);
  ~Gaussian();

  double getDensity(double x) override;
  double getLogDensity(double x) override;
  double getRandomNumber() override;
  static double logLikelihood(double sigma, size_t nData, double* x, double* u);
  void setDistribution(double mean, double sigma) { _mean = mean; _sigma = sigma; };

  void printDetails() override;
  void initialize();

  // Serialization Methods
  nlohmann::json getConfiguration() override;
  void setConfiguration(nlohmann::json& js) override;
};

} } // namespace Korali::Variable

#endif // _KORALI_VARIABLE_GAUSSIAN_H_
