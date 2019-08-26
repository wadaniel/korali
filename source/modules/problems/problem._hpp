#ifndef _KORALI_PROBLEM_BASE_HPP_
#define _KORALI_PROBLEM_BASE_HPP_

#include "modules/module.hpp"
#include "auxiliars/json.hpp"
#include "model.hpp"

namespace Korali { namespace Problem {

class Base : public Korali::Module
{
 public:

  Korali::Model  _modelData;

  virtual void runModel(std::vector<double> target, size_t targetId) = 0;
  virtual bool isTargetFeasible(double* target) = 0;
  virtual double evaluateTarget() = 0;
};

} } // namespace Korali


#endif // _KORALI_PROBLEM_BASE_HPP_
