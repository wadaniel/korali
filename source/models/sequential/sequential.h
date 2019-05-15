#ifndef _KORALI_MODEL_SEQUENTIAL_H_
#define _KORALI_MODEL_SEQUENTIAL_H_

#include "models/base/base.h"

namespace Korali::Model
{

class Sequential : public Base {
 public:

 // Constructor / Destructor
 Sequential(nlohmann::json& js);
 ~Sequential();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_MODEL_SEQUENTIAL_H_
