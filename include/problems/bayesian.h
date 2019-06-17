#ifndef _KORALI_PROBLEM_LIKELIHOOD_H_
#define _KORALI_PROBLEM_LIKELIHOOD_H_

#include "problems/base.h"

namespace Korali { namespace Problem {

class Bayesian : public Base
{
 public:

 // Reference Data
 double* _referenceData;
 double* fitnessArrayPointer;
 size_t _referenceDataSize;

 std::vector<size_t> _computationalVariableIndices;
 std::vector<size_t> _statisticalVariableIndices;

 void packVariables(double* sample, Korali::ModelData& data) override;
 double evaluateFitness(Korali::ModelData& data) override;
 double evaluateLogPrior(double* sample) override;

 // Initialization Method
 void initialize() override;

 // Constructor / Destructor
 Bayesian(nlohmann::json& js);
 ~Bayesian();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} } // namespace Korali::Problem


#endif // _KORALI_PROBLEM_LIKELIHOOD_H_
