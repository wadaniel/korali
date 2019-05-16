#ifndef _KORALI_MODEL_SIMPLE_H_
#define _KORALI_MODEL_SIMPLE_H_

#include "models/base/base.h"

namespace Korali::Model
{

class Simple : public Base {
 public:

  // Team Creation and Distribution
  void initialize(size_t nRanks);
  size_t getTeamCount();
  size_t getRanksPerTeam();

 // Evaluation Methods
 virtual void evaluate(std::vector<double>& parameters, std::vector<double>& results);
 std::function<void(Korali::Model::Simple&)> _method;

 // Constructor / Destructor
 Simple(std::function<void(Korali::Model::Simple&)> method);
 ~Simple();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_MODEL_SIMPLE_H_
