#ifndef _KORALI_BASEPROBLEM_H_
#define _KORALI_BASEPROBLEM_H_

#include "json/json.hpp"

namespace Korali::Problem
{

class Base
{
 public:

 virtual double evaluateFitness(double* sample) = 0;
 bool isSampleOutsideBounds(double* sample);

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

} // namespace Korali


#endif // _KORALI_BASEPROBLEM_H_
