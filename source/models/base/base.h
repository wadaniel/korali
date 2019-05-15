#ifndef _KORALI_MODEL_BASE_
#define _KORALI_MODEL_BASE_

#include <stdlib.h>
#include "json/json.hpp"

namespace Korali::Model
{

class Base {
 public:

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

class Model;

} // namespace Korali

#endif // _KORALI_MODEL_BASE_
