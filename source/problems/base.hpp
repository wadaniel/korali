#ifndef _KORALI_PROBLEM_BASE_HPP_
#define _KORALI_PROBLEM_BASE_HPP_

#include "model.hpp"

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


#endif // _KORALI_PROBLEM_BASE_HPP_
