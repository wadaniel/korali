#ifndef _KORALI_PROBLEM_BAYESIAN_H_
#define _KORALI_PROBLEM_BAYESIAN_H_

#include "problems/base.h"

namespace Korali { namespace Problem {

enum likelihoodType  {DirectLikelihood, ReferenceLikelihood};
enum likelihoodModel {AdditiveGaussian, MultiplicativeGaussian, MultiplicativeGaussianData};

class Bayesian : public Base
{
 public:

 // Reference Data
 std::vector<double> _referenceData;
 double* fitnessArrayPointer;
 size_t _referenceDataSize;
 likelihoodType  _likelihood;
 likelihoodModel _likelihoodModel;

 std::vector<size_t> _computationalVariableIndices;
 std::vector<size_t> _statisticalVariableIndices;

 void packVariables(double* sample, Korali::Model& data) override;
 double evaluateFitness(Korali::Model& data) override;
 double evaluateLogPrior(double* sample) override;

 void initialize() override;
 void finalize() override;

 // Serialization Methods
 void getConfiguration() override;
 void setConfiguration() override;

  private:

    double likelihoodGaussianAdditive(Korali::Model& data);
    double likelihoodGaussianMultiplicative(Korali::Model& data);
    double likelihoodGaussianMultiplicativeData(Korali::Model& data);

};

} } // namespace Korali::Problem


#endif // _KORALI_PROBLEM_BAYESIAN_H_
