#ifndef _KORALI_PROBLEM_BASE_H_
#define _KORALI_PROBLEM_BASE_H_

#include "json/json.hpp"
#include "variables/base.h"
#include "conduits/base.h"

namespace Korali { namespace Problem {

class Base
{
 public:

 virtual double evaluateFitness(Korali::ModelData& data) = 0;
 virtual double evaluateLogPrior(double* sample) = 0;

 size_t N; // Variable Count size_t N; // Variable Count
 size_t _statisticalVariableCount;
 size_t _computationalVariableCount;

 std::vector<Korali::Variable::Base*> _variables;

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

} } // namespace Korali::Problem


#endif // _KORALI_PROBLEM_BASE_H_
