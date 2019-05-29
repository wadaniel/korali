#ifndef _KORALI_PROBLEM_DIRECT_H_
#define _KORALI_PROBLEM_DIRECT_H_

#include "problems/base.h"

namespace Korali::Problem
{

class Direct : public Korali::Problem::Base
{
 public:

  double evaluateFitness(double* sample, bool isLeader
 #ifdef _KORALI_USE_MPI
 , MPI_Comm comm = MPI_COMM_NULL
 #endif
 ) override;

 double evaluateLogPrior(double* sample) override;

 // Constructor / Destructor
 Direct(nlohmann::json& js);
 ~Direct();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali


#endif // _KORALI_PROBLEM_DIRECT_H_
