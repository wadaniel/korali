#ifndef _KORALI_TESTSUITE_H_
#define _KORALI_TESTSUITE_H_

#include <map>
#include <vector>
#include <string>
#include <functional>

#include "korali.h"

namespace Suite
{

typedef double (*TestFun) (int, double*);

class TestSuite
{

public:

  TestSuite(Korali::Engine &engine);
  ~TestSuite();

  void run();
  void addTestFunction(std::string name, TestFun fptr, double fitness = 0.0, size_t numFunEval = 10000);
  void addTargetFitness(std::string name, double);
  void addMaxFunctionEvaluations(std::string, size_t numFunEval);

private:

  Korali::Engine _engine;
  size_t _repetitions;
  double _precision;

  std::vector<std::pair<std::string, TestFun>> _functions;
  std::map<std::string, double> _fitnessMap;
  std::map<std::string, size_t> _maxModelEvals;

};

} // namespace Suite

#endif // _KORALI_TESTSUITE_H_
