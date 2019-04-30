#ifndef _KORALI_LIKELIHOOD_H_
#define _KORALI_LIKELIHOOD_H_

#include "problems/base/base.h"

namespace Korali::Problem
{

class Likelihood : public Korali::Problem::Base
{
 public:

 // Reference Data
 double* _referenceData;
 double* fitnessArrayPointer;

 size_t _sigmaPosition;
 size_t _referenceDataSize;
 double evaluateFitness(double* sample) override;

 // Constructor / Destructor
 Likelihood(nlohmann::json& js);
 ~Likelihood();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali


#endif // _KORALI_LIKELIHOOD_H_
