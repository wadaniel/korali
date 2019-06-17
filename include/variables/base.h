#ifndef _KORALI_VARIABLE_BASE_H_
#define _KORALI_VARIABLE_BASE_H_

#include <string>
#include "json/json.hpp"

namespace Korali { namespace Variable {

class Base
{
 public:

  std::string _name;

 virtual double getDensity(double x) = 0;
 virtual double getLogDensity(double x) = 0;
 virtual double getRandomNumber() = 0;
 virtual void printDetails() = 0;
 virtual void initialize() = 0;

 // Serialization Methods
 virtual nlohmann::json getConfiguration() = 0;
 virtual void setConfiguration(nlohmann::json& js) = 0;
};

} } // namespace Korali::Variable

#endif // _KORALI_VARIABLE_BASE_H_
