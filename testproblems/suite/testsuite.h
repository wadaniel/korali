#ifndef _KORALI_TESTSUITE_H_
#define _KORALI_TESTSUITE_H_

#include <map>
#include <vector>
#include <string>
#include <functional>

namespace TestSuite {

using TestFun = typename std::function<double(int, double*)>;

class TestSuite
{

TestSuite();
~TestSuite();

void run();
void addTestFunction(std::string name, double (&fRef) (int, double*));
void addTargetFitness(std::string name, double);
void addMaxFunctionEvaluations(std::string, size_t numFunEval);

private:

size_t _repetitions;
std::vector<std::pair<std::string, TestFun>> _functions;
std::map<std::string, double> _fitnessMap;
std::map<std::string, size_t> _maxFunEvals;

};

} // namespace Testsuite

#endif // _KORALI_TESTSUITE_H_
