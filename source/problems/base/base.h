#ifndef _KORALI_BASEPROBLEM_H_
#define _KORALI_BASEPROBLEM_H_

#include "json/json.hpp"
#include "variables/base/base.h"

namespace Korali::Problem
{

class Base
{
 public:

 virtual double evaluateFitness(double* sample) = 0;
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


#endif // _KORALI_BASEPROBLEM_H_
