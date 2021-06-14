#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/problem/optimization/optimization.hpp"
#include "sample/sample.hpp"

namespace
{
 using namespace korali;
 using namespace korali::problem;


 TEST(Problem, Optimization)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  v._distributionIndex = 0;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Initial Mean"] = 0.0;
  e["Variables"][0]["Initial Standard Deviation"] = 0.25;
  e["Variables"][0]["Lower Bound"] = -1.0;
  e["Variables"][0]["Upper Bound"] = 1.0;
  // Configuring Problem
  e["Problem"]["Type"] = "Optimization";
  Optimization* pO;
  knlohmann::json problemJs;
  problemJs["Type"] = "Optimization";
  problemJs["Objective Function"] = 0;
  e["Solver"]["Type"] = "Optimizer/CMAES";

  ASSERT_NO_THROW(pO = dynamic_cast<Optimization *>(Module::getModule(problemJs, &e)));
  e._problem = pO;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pO->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pO->applyVariableDefaults();
  ASSERT_NO_THROW(pO->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pO->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ 1.0 });
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  pO->_constraints = std::vector<size_t>({0});

  // Evaluating correct execution of evaluation
  Sample s;
  ASSERT_NO_THROW(pO->evaluateConstraints(s));
  ASSERT_NO_THROW(pO->evaluate(s));
  ASSERT_NO_THROW(pO->evaluateWithGradients(s));

  // Evaluating incorrect execution of evaluation
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::numeric_limits<double>::infinity();
   s["Gradient"] = std::vector<double>(1.0);
  };

  ASSERT_ANY_THROW(pO->evaluateConstraints(s));
  ASSERT_ANY_THROW(pO->evaluate(s));
  ASSERT_ANY_THROW(pO->evaluateWithGradients(s));

  // Evaluating incorrect execution of gradient
  modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ std::numeric_limits<double>::infinity() });
  };

  ASSERT_NO_THROW(pO->evaluateConstraints(s));
  ASSERT_NO_THROW(pO->evaluate(s));
  ASSERT_ANY_THROW(pO->evaluateWithGradients(s));

 // Evaluating correct execution of multiple evaluations
 modelFc = [](Sample& s)
 {
  s["F(x)"] = std::vector<double>({1.0, 1.0});
 };

 ASSERT_NO_THROW(pO->evaluateMultiple(s));

 // Evaluating incorrect execution of multiple evaluations
 modelFc = [](Sample& s)
 {
  s["F(x)"] = std::vector<double>({std::numeric_limits<double>::infinity(), 1.0});
 };

 ASSERT_ANY_THROW(pO->evaluateMultiple(s));

 // Testing optional parameters
 problemJs = baseProbJs;
 experimentJs = baseExpJs;
 problemJs["Has Discrete Variables"] = "Not a Number";
 ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

 problemJs = baseProbJs;
 experimentJs = baseExpJs;
 problemJs["Has Discrete Variables"] = true;
 ASSERT_NO_THROW(pO->setConfiguration(problemJs));

 problemJs = baseProbJs;
 experimentJs = baseExpJs;
 problemJs["Num Objectives"] = "Not a Number";
 ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

 problemJs = baseProbJs;
 experimentJs = baseExpJs;
 problemJs["Num Objectives"] = 1;
 ASSERT_NO_THROW(pO->setConfiguration(problemJs));

 problemJs = baseProbJs;
 experimentJs = baseExpJs;
 problemJs["Objective Function"] = "Not a Number";
 ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

 problemJs = baseProbJs;
 experimentJs = baseExpJs;
 problemJs["Objective Function"] = 1;
 ASSERT_NO_THROW(pO->setConfiguration(problemJs));

 problemJs = baseProbJs;
 experimentJs = baseExpJs;
 problemJs["Constraints"] = "Not a Number";
 ASSERT_ANY_THROW(pO->setConfiguration(problemJs));

 problemJs = baseProbJs;
 experimentJs = baseExpJs;
 problemJs["Constraints"] = std::vector<uint64_t>({1});
 ASSERT_NO_THROW(pO->setConfiguration(problemJs));

 }
} // namespace
