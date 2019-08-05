#ifndef _KORALI_PROBLEM_BASE_HPP_
#define _KORALI_PROBLEM_BASE_HPP_

namespace Korali { namespace Problem {

class Base
{
  public:

  virtual double evaluateFitness(std::vector<double> sample, size_t sampleId) = 0;
  virtual double evaluateLogPrior(double* sample) = 0;

  virtual void initialize() = 0;
  virtual void finalize() = 0;

  // Serialization Methods
  virtual void getConfiguration() = 0;
  virtual void setConfiguration() = 0;
};

} } // namespace Korali::Problem


#endif // _KORALI_PROBLEM_BASE_HPP_
