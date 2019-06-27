#include "testsuite.h"

#include <stdio.h>

namespace Suite
{

TestSuite::TestSuite() : _repetitions(100), _precision(1e-4)
{
  _factories.push_back(new CMAESFactory());
  _factories.push_back(new DEFactory("Best", "Self Adaptive", "Greedy"));
  _factories.push_back(new DEFactory("Best", "Default", "Greedy"));
};

TestSuite::~TestSuite() {};

void TestSuite::run()
{


  for(auto func : _functions)
  {

    std::string name = func.first;


    auto k = _factories[0]->createEngine(_dimMap[name], _domainMap[name].first, _domainMap[name].second, _maxModelEvals[name], _fitnessMap[name] - _precision);

    auto model = [func](Korali::ModelData& d)
    {
      double res = func.second( d.getVariableCount(), &d.getVariables()[0] );
      d.addResult(res);
    };
    k.setModel(model);

    printf("\n\nRun Test: %s\nSolver: %s\n\n", name.c_str(), "CMA-ES");

    k.run();

  }
}


void TestSuite::addTestFunction(std::string name, TestFun fptr, size_t dim, double lowerBound, double upperBound, double fitness, size_t numFunEval )
{
  _functions.push_back( std::pair<std::string, TestFun> (name, fptr) );

  _dimMap.insert( std::pair<std::string, size_t>(name, dim) );

  std::pair<double, double> domain(lowerBound, upperBound);
  _domainMap.insert( std::pair<std::string, std::pair<double, double> > (name, domain ) );

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
