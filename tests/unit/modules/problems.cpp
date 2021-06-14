#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/problem/optimization/optimization.hpp"
#include "modules/problem/sampling/sampling.hpp"
#include "modules/problem/supervisedLearning/supervisedLearning.hpp"
#include "modules/problem/reinforcementLearning/reinforcementLearning.hpp"
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
  Optimization* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Optimization";
  problemJs["Objective Function"] = 0;
  e["Solver"]["Type"] = "Optimizer/CMAES";

  ASSERT_NO_THROW(pObj = dynamic_cast<Optimization *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ 1.0 });
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  pObj->_constraints = std::vector<size_t>({0});

  // Evaluating correct execution of evaluation
  Sample s;
  ASSERT_NO_THROW(pObj->evaluateConstraints(s));
  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateWithGradients(s));

  // Evaluating incorrect execution of evaluation
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::numeric_limits<double>::infinity();
   s["Gradient"] = std::vector<double>(1.0);
  };

  ASSERT_ANY_THROW(pObj->evaluateConstraints(s));
  ASSERT_ANY_THROW(pObj->evaluate(s));
  ASSERT_ANY_THROW(pObj->evaluateWithGradients(s));

  // Evaluating incorrect execution of gradient
  modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ std::numeric_limits<double>::infinity() });
  };

  ASSERT_NO_THROW(pObj->evaluateConstraints(s));
  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_ANY_THROW(pObj->evaluateWithGradients(s));

  // Evaluating incorrect size of gradients
  modelFc = [](Sample& s)
  {
   s["F(x)"] = 1.0;
   s["Gradient"] = std::vector<double>({ });
  };

  ASSERT_ANY_THROW(pObj->evaluateWithGradients(s));

  // Evaluating correct execution of multiple evaluations
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::vector<double>({1.0, 1.0});
  };

  ASSERT_NO_THROW(pObj->evaluateMultiple(s));

  // Trying to run unknown operation
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Evaluating incorrect execution of multiple evaluations
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::vector<double>({std::numeric_limits<double>::infinity(), 1.0});
  };

  ASSERT_ANY_THROW(pObj->evaluateMultiple(s));

  // Testing optional parameters
  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Has Discrete Variables"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Has Discrete Variables"] = true;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs.erase("Num Objectives");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Num Objectives"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Num Objectives"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs.erase("Objective Function");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));


  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Objective Function"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Objective Function"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs.erase("Constraints");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Constraints"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  problemJs["Constraints"] = std::vector<uint64_t>({1});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Granularity");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Granularity"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Granularity"] = 1.0;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 };

 TEST(Problem, SupervisedLearning)
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
  SupervisedLearning* pObj;
  knlohmann::json problemJs;

  problemJs["Type"] = "Supervised Learning";
  problemJs["Max Timesteps"] = 1;
  problemJs["Training Batch Size"] = 1;
  problemJs["Inference Batch Size"] = 1;

  problemJs["Input"]["Data"] = std::vector<std::vector<std::vector<float>>>({{{0.0}}});
  problemJs["Input"]["Size"] = 1;
  problemJs["Solution"]["Data"] = std::vector<std::vector<float>>({{0.0}});
  problemJs["Solution"]["Size"] = 1;

  e["Solver"]["Type"] = "Learner/DeepSupervisor";

  ASSERT_NO_THROW(pObj = dynamic_cast<SupervisedLearning *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing data verifications
  ASSERT_NO_THROW(pObj->verifyData());
  pObj->_inputData = std::vector<std::vector<std::vector<float>>>({{{0.0, 0.0}}});
  ASSERT_ANY_THROW(pObj->verifyData());
  pObj->_inputData = std::vector<std::vector<std::vector<float>>>({{{0.0}, {0.0}}});
  ASSERT_ANY_THROW(pObj->verifyData());
  pObj->_inputData = std::vector<std::vector<std::vector<float>>>({{{0.0}}, {{0.0}}});
  ASSERT_ANY_THROW(pObj->verifyData());
  pObj->_inputData = std::vector<std::vector<std::vector<float>>>({{{0.0}}});
  ASSERT_NO_THROW(pObj->verifyData());
  pObj->_solutionData = std::vector<std::vector<float>>({{0.0, 0.0}});
  ASSERT_ANY_THROW(pObj->verifyData());
  pObj->_solutionData = std::vector<std::vector<float>>({{0.0}, {0.0}});
  ASSERT_ANY_THROW(pObj->verifyData());

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Training Batch Size");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Training Batch Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Training Batch Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Inference Batch Size");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Inference Batch Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Inference Batch Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Max Timesteps");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Max Timesteps"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Max Timesteps"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"].erase("Data");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"]["Data"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"]["Data"] = std::vector<std::vector<std::vector<float>>>({{{0.0}}});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"].erase("Size");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"]["Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Input"]["Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"].erase("Data");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"]["Data"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"]["Data"] = std::vector<std::vector<float>>({{0.0}});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"].erase("Size");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"]["Size"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Solution"]["Size"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }
} // namespace
