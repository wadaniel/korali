#ifndef _KORALI_PROBLEM_BASE_H_
#define _KORALI_PROBLEM_BASE_H_

#include "auxiliar/json/json.hpp"
#include "variables/base/base.h"

namespace Korali::Problem
{

class Base
{
 public:

 virtual double evaluateFitness(double* sample, bool isLeader
 #ifdef _KORALI_USE_MPI
 , MPI_Comm comm = MPI_COMM_NULL
 #endif
 ) = 0;

 virtual double evaluateLogPrior(double* sample) = 0;

 size_t N; // Variable Count size_t N; // Variable Count
 size_t _statisticalParameterCount;
 size_t _computationalParameterCount;

 bool isSampleOutsideBounds(double* sample);
 std::vector<Korali::Variable::Base*> _parameters;

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

} // namespace Korali


#endif // _KORALI_PROBLEM_BASE_H_
