#ifndef _KORALI_PROBLEM_DIRECTBAYESIAN_H_
#define _KORALI_PROBLEM_DIRECTBAYESIAN_H_

#include "problems/base.h"
#include <vector>

namespace Korali { namespace Problem {

class DirectBayesian : public Base
{
 public:

 void packVariables(double* sample, Korali::ModelData& data) override;
 double evaluateFitness(Korali::ModelData& data) override;
 double evaluateLogPrior(double* sample) override;

 double* _referenceData;
 double* fitnessArrayPointer;
 size_t _referenceDataSize;

 std::vector<size_t> _computationalVariableIndices;
 std::vector<size_t> _statisticalVariableIndices;

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
