#ifndef _KORALI_VARIABLE_DEFAULT_H_
#define _KORALI_VARIABLE_DEFAULT_H_

#include <string>
#include "json/json.hpp"
#include "variables/base.h"

namespace Korali { namespace Variable {

class Default : public Base
{
 public:

 // Constructor / Destructor
 Default(nlohmann::json& js);
 Default();
 ~Default();

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

#endif // _KORALI_VARIABLE_DEFAULT_H_
