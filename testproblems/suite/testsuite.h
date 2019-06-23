#ifndef _KORALI_TESTSUITE_H_
#define _KORALI_TESTSUITE_H_

#include <map>
#include <vector>
#include <string>
#include <functional>

#include "korali.h"

namespace Suite {

typedef double (*TestFun) (int, double*);

class TestSuite
{

public:

  TestSuite();
  ~TestSuite();

  void run();
  void addTestFunction(std::string name, TestFun fptr, size_t dim, double lowerBound, double upperBound, double fitness, size_t numFunEval);
  void addTargetFitness(std::string name, double fitness );
  void addMaxFunctionEvaluations(std::string, size_t numFunEval);

private:

  size_t _repetitions;
  double _precision;

  std::vector<std::pair<std::string, TestFun>> _functions;
  std::map<std::string, size_t> _dimMap;
  std::map<std::string, std::pair<double, double>> _domainMap;
  std::map<std::string, double> _fitnessMap;
  std::map<std::string, size_t> _maxModelEvals;

};

} // namespace Suite

#endif // _KORALI_TESTSUITE_H_
