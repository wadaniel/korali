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
    printf("Run Test: %s\n", func.first.c_str());

  }
}


void TestSuite::addTestFunction(std::string name, TestFun fptr, double fitness, size_t numFunEval )
{
  _functions.push_back( std::pair<std::string, TestFun> (name, fptr) );
  
  _fitnessMap.insert( std::pair<std::string, double>(name, fitness) );
  
  _maxFunEvals.insert( std::pair<std::string, size_t>(name, numFunEval) );
}

void TestSuite::addTargetFitness(std::string name, double fitness)
{
  _fitnessMap.insert( std::pair<std::string, double>(name, fitness) );
}


void TestSuite::addMaxFunctionEvaluations(std::string name, size_t numFunEval)
{
  _maxFunEvals.insert( std::pair<std::string, size_t>(name, numFunEval) );
}

} // namespace Suite
