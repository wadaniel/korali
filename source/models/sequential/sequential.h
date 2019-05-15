#ifndef _KORALI_MODEL_SEQUENTIAL_H_
#define _KORALI_MODEL_SEQUENTIAL_H_

#include "models/base/base.h"

namespace Korali::Model
{

class Sequential : public Base {
 public:

 // Evaluation Methods
 virtual void evaluate(std::vector<double>& parameters, std::vector<double>& results);
 std::function<void(Korali::Model::Sequential&)> _method;

 // Constructor / Destructor
 Sequential(std::function<void(Korali::Model::Sequential&)> method);
 ~Sequential();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_MODEL_SEQUENTIAL_H_
