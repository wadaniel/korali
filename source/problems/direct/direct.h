#ifndef _KORALI_DIRECT_H_
#define _KORALI_DIRECT_H_

#include "problems/base/base.h"

namespace Korali::Problem
{

class Direct : public Korali::Problem::Base
{
 public:
 double evaluateFitness(double* sample);

 // Constructor / Destructor
 Direct(nlohmann::json& js);
 ~Direct();

 // Serialization Methods
 nlohmann::json getConfiguration();
 void setConfiguration(nlohmann::json& js);
};

} // namespace Korali


#endif // _KORALI_DIRECT_H_
