#ifndef _KORALI_VARIABLE_HPP_
#define _KORALI_VARIABLE_HPP_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <string>
#include "json.hpp"

enum KoraliDistributionType {KoraliCauchyDistribution, KoraliDefaultDistribution, KoraliExponentialDistribution, KoraliGammaDistribution, KoraliGaussianDistribution, KoraliLaplaceDistribution, KoraliUniformDistribution, KoraliGeometricDistribution};

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

 bool _isLogSpace;
 std::string _name;
 KoraliDistributionType _distributionType;

 // Constructor / Destructor
 Variable();
 ~Variable();

 double getDensity(double x);
 double getLogDensity(double x);
 double getRandomNumber();
 void printDetails();
 void initialize();
 void getConfiguration(nlohmann::json& js);
 void setConfiguration(nlohmann::json& js);
 void setDistribution(nlohmann::json& js);
 void getDistribution(nlohmann::json& js);
};

} // namespace Korali

#endif // _KORALI_VARIABLE_HPP_
