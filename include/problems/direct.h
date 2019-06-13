#ifndef _KORALI_PROBLEM_DIRECT_H_
#define _KORALI_PROBLEM_DIRECT_H_

#include "problems/base.h"

namespace Korali { namespace Problem {

class Direct : public Base
{
 public:

 double evaluateFitness(Korali::ModelData& data) override;
 double evaluateLogPrior(double* sample) override;

 // Constructor / Destructor
 Direct(nlohmann::json& js);
 ~Direct();

 // Initialization Method
 void initialize() override;

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} } // namespace Korali::Problem

#endif // _KORALI_PROBLEM_DIRECT_H_
