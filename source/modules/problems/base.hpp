#ifndef _KORALI_PROBLEM_BASE_HPP_
#define _KORALI_PROBLEM_BASE_HPP_

#include "modules/module.hpp"
#include "auxiliars/json.hpp"
#include "variable/variable.hpp"
#include "model.hpp"

namespace Korali { namespace Problem {

class Base : public Korali::Module
{
 public:

  Korali::Model  _modelData;

  virtual void runModel(std::vector<double>, size_t sampleId) = 0;

  virtual size_t getVariableCount() = 0;
  virtual Korali::Variable* getVariable(size_t variableId) = 0;
  virtual bool isSampleFeasible(double* sample) = 0;

  virtual double evaluateSampleFitness() = 0;
  virtual double evaluateSampleLogPrior(double* sample) = 0;
};

} } // namespace Korali


#endif // _KORALI_PROBLEM_BASE_HPP_
