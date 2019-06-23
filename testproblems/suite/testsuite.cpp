#include "testsuite.h"

#include <stdio.h>

namespace Suite
{

TestSuite::TestSuite() : _repetitions(100), _precision(1e-4) {};
TestSuite::~TestSuite() {};

void TestSuite::run()
{
  for(auto func : _functions)
  {

    std::string name = func.first;

    auto k = Korali::Engine();
    k["Verbosity"] = "Minimal";
    k["Output Frequency"] = 100;
    k["CMA-ES"]["Result Output Frequency"] = 5000;

    k["Problem"] = "Direct Evaluation";
    k["Solver"]  = "CMA-ES";
    
    k["CMA-ES"]["Objective"]    = "Minimize";
    k["CMA-ES"]["Sample Count"] = 4+3*log(_dimMap[name]);
 
    for (int i = 0; i < _dimMap[name]; i++)
    {
        k["Variables"][i]["Name"] = "X" + std::to_string(i);
        k["Variables"][i]["CMA-ES"]["Lower Bound"] = _domainMap[name].first;
        k["Variables"][i]["CMA-ES"]["Upper Bound"] = _domainMap[name].second;
    }

    k["CMA-ES"]["Termination Criteria"]["Max Generations"]["Active"]          = false;
    k["CMA-ES"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"]   = false;
    k["CMA-ES"]["Termination Criteria"]["Min Standard Deviation"]["Active"]   = false;
    k["CMA-ES"]["Termination Criteria"]["Max Standard Deviation"]["Active"]   = false;
    k["CMA-ES"]["Termination Criteria"]["Max Condition Covariance"]["Active"] = false;
    k["CMA-ES"]["Termination Criteria"]["Min Step Size"]["Active"]            = false;
    
    k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Active"] = true;
    k["CMA-ES"]["Termination Criteria"]["Max Model Evaluations"]["Value"] = _maxModelEvals[name];
    
    k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Active"] = true;
    k["CMA-ES"]["Termination Criteria"]["Min Fitness"]["Value"] = _fitnessMap[name]-_precision;
  
    auto model = [func](Korali::ModelData& d) { double res = func.second( d.getVariableCount(), &d.getVariables()[0] ); d.addResult(res); };
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
