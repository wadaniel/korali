#ifndef _KORALI_VARIABLE_HPP_
#define _KORALI_VARIABLE_HPP_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <string>
#include "json.hpp"

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
 std::string _distributionType;

 // Constructor / Destructor
 Variable();
 ~Variable();

 double getDensity(double x);
 double getLogDensity(double x);
 double getRandomNumber();
 void printDetails();
 void initialize();
 void setDistributionType(std::string distributionType);
 void setProperty(std::string propertyName, double value);
 void getConfiguration(nlohmann::json& js);
 void setConfiguration(nlohmann::json& js);
 void setDistribution(nlohmann::json& js);
 void getDistribution(nlohmann::json& js);
 void setSolverSettings(nlohmann::json& js);
 void getSolverSettings(nlohmann::json& js);
 };

} // namespace Korali

#endif // _KORALI_VARIABLE_HPP_
