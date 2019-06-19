#ifndef _KORALI_VARIABLE_H_
#define _KORALI_VARIABLE_H_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <string>
#include "json/json.hpp"

enum KoraliDistributionType {KoraliCauchyDistribution, KoraliDefaultDistribution, KoraliExponentialDistribution, KoraliGammaDistribution, KoraliGaussianDistribution, KoraliLaplaceDistribution, KoraliUniformDistribution};

namespace Korali {

class Variable
{
 private:

 size_t _seed;
 gsl_rng* _range;
 double _aux;
 double _a;
 double _b;

 public:

 std::string _name;
 KoraliDistributionType _distributionType;

 // Constructor / Destructor
 Variable(std::string name = "");
 ~Variable();

 double getDensity(double x);
 double getLogDensity(double x);
 double getRandomNumber();
 void printDetails();
 void initialize();

 // Serialization Methods
  void setDistribution(nlohmann::json& js);
  void getDistribution(nlohmann::json& js);
};

} // namespace Korali

#endif // _KORALI_VARIABLE_H_
