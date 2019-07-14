#ifndef _KORALI_PROBLEM_BASE_H_
#define _KORALI_PROBLEM_BASE_H_

#include "variable.h"
#include "model.h"
#include "conduits/base.h"

namespace Korali { namespace Problem {

class Base
{
  public:

  virtual void packVariables(double* sample, Korali::Model& data) = 0;
  virtual double evaluateFitness(Korali::Model& data) = 0;
  virtual double evaluateLogPrior(double* sample) = 0;

  virtual void initialize() = 0;
  virtual void finalize() = 0;

  // Serialization Methods
  virtual void getConfiguration() = 0;
  virtual void setConfiguration() = 0;
};

} } // namespace Korali::Problem


#endif // _KORALI_PROBLEM_BASE_H_
