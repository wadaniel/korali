#ifndef _KORALI_PROBLEM_LIKELIHOOD_H_
#define _KORALI_PROBLEM_LIKELIHOOD_H_

#include "problems/base/base.h"

namespace Korali::Problem
{

class Bayesian : public Korali::Problem::Base
{
 public:

 // Reference Data
 double* _referenceData;
 double* fitnessArrayPointer;

 size_t _sigmaPosition;
 size_t _referenceDataSize;
 double evaluateFitness(double* sample, MPI_Comm comm = MPI_COMM_WORLD) override;
 double evaluateLogPrior(double* sample) override;

 // Constructor / Destructor
 Bayesian(nlohmann::json& js);
 ~Bayesian();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali


#endif // _KORALI_PROBLEM_LIKELIHOOD_H_
