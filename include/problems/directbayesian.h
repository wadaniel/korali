#ifndef _KORALI_PROBLEM_DIRECTBAYESIAN_H_
#define _KORALI_PROBLEM_DIRECTBAYESIAN_H_

#include "problems/base.h"

namespace Korali { namespace Problem {

class DirectBayesian : public Base
{
 public:

 double evaluateFitness(Korali::ModelData& data) override;
 double evaluateLogPrior(double* sample) override;

 // Constructor / Destructor
 DirectBayesian(nlohmann::json& js);
 ~DirectBayesian();

 // Initialization Method
 void initialize() override;

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} } // namespace Korali::Problem

#endif // _KORALI_PROBLEM_DIRECTBAYESIAN_H_
