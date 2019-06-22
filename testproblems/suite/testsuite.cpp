#include "testsuite.h"

#include <stdio.h>

namespace TestSuite
{

TestSuite::TestSuite() : _repetitions(100) {};

void TestSuite::run()
{
  printf("TODO\n");
}


void TestSuite::addTestFunction(std::string name, double (&fRef) (int, double*))
{
  _functions.insert( std::pair<std::string, (double) & (int, double*)> (name, fRef) );
}

void TestSuite::addTargetFitness(std::string name, double fitness)
{
  _fitnessMap.insert( std::pair<std::string, double>(name, fitness) );
}


void TestSuite::addMaxFunctionEvaluations(std::string name, size_t numFunEval)
{
  _maxFunEvals.insert( std::pair<std::string, size_t>(name, numFunEval) );
}

}
