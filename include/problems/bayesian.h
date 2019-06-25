#ifndef _KORALI_PROBLEM_LIKELIHOOD_H_
#define _KORALI_PROBLEM_LIKELIHOOD_H_

#include "problems/base.h"

namespace Korali { namespace Problem {

enum likelihoodType {DirectLikelihood, ReferenceLikelihood};

class Bayesian : public Base
{
 public:

 // Reference Data
 double* _referenceData;
 double* fitnessArrayPointer;
 size_t _referenceDataSize;
 likelihoodType _likelihood;

 std::vector<size_t> _computationalVariableIndices;
 std::vector<size_t> _statisticalVariableIndices;

 void packVariables(double* sample, Korali::ModelData& data) override;
 double evaluateFitness(Korali::ModelData& data) override;
 double evaluateLogPrior(double* sample) override;

 // Constructor / Destructor
 Bayesian(nlohmann::json& js);
 ~Bayesian();

 // Serialization Methods
 void getConfiguration(nlohmann::json& js) override;
 void setConfiguration(nlohmann::json& js) override;
};

} } // namespace Korali::Problem


#endif // _KORALI_PROBLEM_LIKELIHOOD_H_
