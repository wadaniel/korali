#include "testsuite.h"

#include <stdio.h>

namespace Suite
{

TestSuite::TestSuite(Korali::Engine &engine) : _repetitions(100), _precision(1e-4), _engine(engine) {};
TestSuite::~TestSuite() {};

void TestSuite::run()
{
  for(auto func : _functions)
  {
    auto js = _engine.getConfiguration();
    std::string name = consume( js, { "Solver" }, Korali::KORALI_STRING) ;

    printf("Run Test: %s\nSolver: %s\n", func.first.c_str(), name.c_str());

    auto model = [func](Korali::ModelData& d)
    {
      double res = func.second( d.getVariableCount(), &d.getVariables()[0] );
      d.addResult(res);
    };
    _engine.setModel(model);

    _engine["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = true;
    _engine["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Value"] = _maxModelEvals[func.first];

    _engine["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Active"] = true;
    _engine["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = _fitnessMap[func.first];

    _engine.run();

  }
}


void TestSuite::addTestFunction(std::string name, TestFun fptr, double fitness, size_t numFunEval )
{
  _functions.push_back( std::pair<std::string, TestFun> (name, fptr) );

  _fitnessMap.insert( std::pair<std::string, double>(name, fitness) );

  _maxModelEvals.insert( std::pair<std::string, size_t>(name, numFunEval) );
}

void TestSuite::addTargetFitness(std::string name, double fitness)
{
  _fitnessMap.insert( std::pair<std::string, double>(name, fitness) );
}


void TestSuite::addMaxFunctionEvaluations(std::string name, size_t numFunEval)
{
  _maxModelEvals.insert( std::pair<std::string, size_t>(name, numFunEval) );
}

} // namespace Suite
