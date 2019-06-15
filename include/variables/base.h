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
 bool _hasDistribution;

 // Constructor / Destructor
 Base(nlohmann::json& js, size_t seed);
 Base(size_t seed);
 ~Base();

 virtual double getDensity(double x) { fprintf(stderr, "[Korali] Error: Problem requires that variable '%s' to have a defined distribution.\n", _name.c_str()); exit(-1); return 0.0; };
 virtual double getLogDensity(double x) { fprintf(stderr, "[Korali] Error: Problem requires that variable '%s' to have a defined distribution.\n", _name.c_str()); exit(-1); return 0.0; };
 virtual double getRandomNumber() { fprintf(stderr, "[Korali] Error: Problem requires that variable '%s' to have a defined distribution.\n", _name.c_str()); exit(-1); return 0.0; };
 virtual void printDetails() { return; };
 virtual void initialize() { };

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

} } // namespace Korali::Variable

#endif // _KORALI_VARIABLE_BASE_H_
