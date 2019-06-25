#ifndef _KORALI_PROBLEM_BASE_H_
#define _KORALI_PROBLEM_BASE_H_

#include "json/json.hpp"
#include "variable.h"
#include "conduits/base.h"

namespace Korali { namespace Problem {

class Base
{
 public:

 virtual void packVariables(double* sample, Korali::ModelData& data) = 0;
 virtual double evaluateFitness(Korali::ModelData& data) = 0;
 virtual double evaluateLogPrior(double* sample) = 0;

 // Serialization Methods
 virtual void getConfiguration(nlohmann::json& js) = 0;
 virtual void setConfiguration(nlohmann::json& js) = 0;
};

} } // namespace Korali::Problem


#endif // _KORALI_PROBLEM_BASE_H_
