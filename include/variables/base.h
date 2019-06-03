#ifndef _KORALI_VARIABLE_BASE_H_
#define _KORALI_VARIABLE_BASE_H_

#include <string>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include "json/json.hpp"

enum variableType { KORALI_COMPUTATIONAL, KORALI_STATISTICAL };

namespace Korali { namespace Variable {

class Base
{
 public:
 gsl_rng* _range;
 variableType _type;

 std::string _name;
 double _lowerBound;
 double _upperBound;

 // These are CMA-ES Specific, but could be used for other methods in the future
 double _initialValue;
 double _initialStdDev;
 double _minStdDevChange;

 // Constructor / Destructor
 Base(nlohmann::json& js, size_t seed);
 Base(size_t seed);
 ~Base();

 virtual double getDensity(double x) = 0;
 virtual double getLogDensity(double x) = 0;
 virtual double getRandomNumber() = 0;
 virtual void printDetails() = 0;

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

} } // namespace Korali::Variable

#endif // _KORALI_VARIABLE_BASE_H_
