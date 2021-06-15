#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/problem/optimization/optimization.hpp"
#include "modules/problem/sampling/sampling.hpp"
#include "modules/problem/bayesian/bayesian.hpp"
#include "modules/problem/bayesian/custom/custom.hpp"
#include "modules/problem/bayesian/reference/reference.hpp"
#include "modules/problem/hierarchical/hierarchical.hpp"
#include "modules/problem/hierarchical/psi/psi.hpp"
#include "modules/problem/hierarchical/theta/theta.hpp"
#include "modules/problem/hierarchical/thetaNew/thetaNew.hpp"
#include "modules/problem/propagation/propagation.hpp"
#include "modules/problem/integration/integration.hpp"
#include "modules/problem/supervisedLearning/supervisedLearning.hpp"
#include "modules/problem/reinforcementLearning/reinforcementLearning.hpp"
#include "sample/sample.hpp"

namespace
{
 using namespace korali;
 using namespace korali::problem;
 using namespace korali::problem::bayesian;
 using namespace korali::problem::hierarchical;

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
  e["Variables"][0].erase("Name");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Name"] = 1.0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseOptJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Name"] = "Var X";
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

 TEST(Problem, Sampling)
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
  e["Problem"]["Type"] = "Sampling";
  Sampling* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Sampling";
  problemJs["Probability Function"] = 0;
  e["Solver"]["Type"] = "Sampler/MCMC";

  ASSERT_NO_THROW(pObj = dynamic_cast<Sampling *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  // Evaluating correct execution of evaluation
  Sample s;
  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateGradient(s));
  ASSERT_NO_THROW(pObj->evaluateHessian(s));

  // Trying to run unknown operation
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // running correct operations
  ASSERT_NO_THROW(pObj->runOperation("Evaluate", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Gradient", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Hessian", s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logP(x)"] = std::numeric_limits<double>::infinity();
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_ANY_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateGradient(s));
  ASSERT_NO_THROW(pObj->evaluateHessian(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{0.5}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_ANY_THROW(pObj->evaluateGradient(s));
  ASSERT_NO_THROW(pObj->evaluateHessian(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>(2);
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateHessian(s));

  modelFc = [](Sample& s)
  {
   s["logP(x)"] = 0.5;
   s["grad(logP(x))"] = std::vector<double>({0.5});
   s["H(logP(x))"] = std::vector<std::vector<double>>({{}});
  };
  _functionVector[0] = &modelFc;
  ASSERT_NO_THROW(pObj->evaluate(s));
  ASSERT_NO_THROW(pObj->evaluateGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateHessian(s));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Probability Function");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Probability Function"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Probability Function"] = modelFc;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, Propagation)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);

  // Configuring Problem
  Propagation* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Propagation";
  problemJs["Execution Model"] = 0;
  e["Solver"]["Type"] = "Executor";

  ASSERT_NO_THROW(pObj = dynamic_cast<Propagation *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Triggering error on variable with distribution and distributions defined
  pObj->_numberOfSamples = 0;
  pObj->initialize();
  ASSERT_NO_THROW(pObj->initialize());

  pObj->_numberOfSamples = 10;
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_numberOfSamples = 0;
  v._priorDistribution = "Uniform";
  ASSERT_ANY_THROW(pObj->initialize());

  // Testing configuration

  baseExpJs["Variables"][0]["Name"] = "Var 1";
  baseExpJs["Variables"][0]["Precomputed Values"] = std::vector<double>({0.0});
  baseExpJs["Variables"][0]["Prior Distribution"] = "Uniform";
  baseExpJs["Variables"][0]["Distribution Index"] = 0;
  baseExpJs["Variables"][0]["Sampled Values"] = std::vector<double>({0.0});

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Execution Model");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Execution Model"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Execution Model"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Number Of Samples");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Number Of Samples"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Number Of Samples"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Precomputed Values");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Precomputed Values"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Precomputed Values"] = std::vector<double>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Prior Distribution");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = 0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = "Uniform";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Distribution Index");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Sampled Values");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sampled Values"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sampled Values"] = std::vector<double>();
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, Integration)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);

  // Integrand function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s) { s["Evaluation"] = 1.0; };

  // Configuring Problem
  Integration* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Integration";
  problemJs["Integrand"] = modelFc;
  problemJs["Integration Method"] = "Rectangle";
  e["Solver"]["Type"] = "Integrator";

  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Number Of Gridpoints"] = 10;
  e["Variables"][0]["Sampling Distribution"] = "Uniform";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;
  e["Variables"][0]["Sample Points"] = std::vector<double>({ 0.0, 0.1, 0.2, 0.3 });
  e["Variables"][0]["Quadrature Weights"] = std::vector<double>({ 0.0, 0.1, 0.2, 0.3 });

  ASSERT_NO_THROW(pObj = dynamic_cast<Integration *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  ASSERT_NO_THROW(pObj->initialize());

  // Testing fail cases
  pObj->_integrationMethod = "Monte Carlo";
  e._variables[0]->_samplingDistribution = "Unknown";
  ASSERT_ANY_THROW(pObj->initialize());

  pObj->_integrationMethod = "Custom";
  e._variables[0]->_samplingDistribution = "";
  e._variables[0]->_quadratureWeights = std::vector<double>({});
  e._variables[0]->_samplePoints = std::vector<double>({0.0});
  ASSERT_ANY_THROW(pObj->initialize());

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Integrand");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integrand"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integrand"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Integration Method");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integration Method"] = 1.0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integration Method"] = "Unknown";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Integration Method"] = "Rectangle";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Lower Bound");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Lower Bound"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Lower Bound"] = 1.0;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Upper Bound");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Upper Bound"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Upper Bound"] = 1.0;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Number Of Gridpoints");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Number Of Gridpoints"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Number Of Gridpoints"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Sampling Distribution");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sampling Distribution"] = 1.0;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sampling Distribution"] = "Uniform";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Distribution Index");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Sample Points");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sample Points"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Sample Points"] = std::vector<double>({});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Quadrature Weights");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Quadrature Weights"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Quadrature Weights"] = std::vector<double>({});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, BayesianCustom)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);

  // Configuring Problem
  Custom* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Bayesian/Custom";
  problemJs["Likelihood Model"] = 0;
  e["Solver"]["Type"] = "Sampler/TMCMC";

  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Number Of Gridpoints"] = 10;
  e["Variables"][0]["Prior Distribution"] = "Uniform";

  ASSERT_NO_THROW(pObj = dynamic_cast<Custom *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  ASSERT_NO_THROW(pObj->initialize());

  // Testing initialization fail
  v._priorDistribution = "Unknown";
  ASSERT_ANY_THROW(pObj->initialize());
  v._priorDistribution = "Uniform";

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["logLikelihood"] = 0.1;
   s["logLikelihood Gradient"] = std::vector<double>({0.1});
   s["Fisher Information"] = std::vector<std::vector<double>>({{0.1}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateFisherInformation(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s._js.getJson().erase("logLikelihood");
   s._js.getJson().erase("logLikelihood Gradient");
   s._js.getJson().erase("Fisher Information");
  };

  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateFisherInformation(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logLikelihood"] = 0.1;
   s["logLikelihood Gradient"] = std::vector<double>();
   s["Fisher Information"] = std::vector<std::vector<double>>({{0.1}});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  pObj->evaluateLoglikelihood(s);
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateFisherInformation(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logLikelihood"] = 0.1;
   s["logLikelihood Gradient"] = std::vector<double>({0.0});
   s["Fisher Information"] = std::vector<std::vector<double>>();
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  pObj->evaluateLoglikelihood(s);
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateFisherInformation(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["logLikelihood"] = 0.1;
   s["logLikelihood Gradient"] = std::vector<double>({0.0});
   s["Fisher Information"] = std::vector<std::vector<double>>({});
  };
  _functionVector.clear();
  _functionVector.push_back(&modelFc);

  pObj->evaluateLoglikelihood(s);
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateFisherInformation(s));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Likelihood Model");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Prior Distribution");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = 1;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Prior Distribution"] = "Uniform";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0].erase("Distribution Index");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Variables"][0]["Distribution Index"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, BayesianReference)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);

  // Configuring Problem
  Reference* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Bayesian/Reference";
  problemJs["Computational Model"] = 0;
  problemJs["Likelihood Model"] = "Normal";
  problemJs["Reference Data"] = std::vector<double>({ 0.0 });
  e["Solver"]["Type"] = "Sampler/TMCMC";

  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Number Of Gridpoints"] = 10;
  e["Variables"][0]["Prior Distribution"] = "Uniform";

  ASSERT_NO_THROW(pObj = dynamic_cast<Reference *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  s["Sample Id"] = 0;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  ASSERT_NO_THROW(pObj->initialize());

  // Evaluation function
  std::function<void(korali::Sample&)> modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  pObj->_likelihoodModel = "Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "StudentT";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Poisson";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Geometric";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s._js.getJson().erase("Standard Deviation");
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  _functionVector.clear();
  _functionVector.push_back(&modelFc);
  pObj->_likelihoodModel = "Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s._js.getJson().erase("Reference Evaluations");
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Poisson";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Geometric";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Unknown";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>();
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Poisson";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Geometric";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  // Evaluation function
  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>();
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s._js.getJson().erase("Degrees Of Freedom");
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>();
   s["Dispersion"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s._js.getJson().erase("Dispersion");
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  modelFc = [](Sample& s)
  {
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>();
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
  };

  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  // Gradient evaluation
  modelFc = [](Sample& s)
  {
   s["F(x)"] = std::numeric_limits<double>::infinity();
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});

   s["Gradient Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Dispersion"] = std::vector<std::vector<double>>({{0.1}});

   s["Hessian Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Dispersion"] = std::vector<std::vector<double>>({{0.1}});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));

  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));

  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));


  modelFc = [](Sample& s)
  {
   s["F(x)"] = 0.1;
   s["Reference Evaluations"] = std::vector<double>({0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});

   s["Gradient Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Dispersion"] = std::vector<std::vector<double>>({{0.1}});

   s["Hessian Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Dispersion"] = std::vector<std::vector<double>>({{0.1}});
  };

  pObj->_likelihoodModel = "Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));

  // Running operations
  ASSERT_NO_THROW(pObj->runOperation("Evaluate", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate logPrior", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate logLikelihood", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate logPosterior", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Gradient", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Hessian", s));
  ASSERT_NO_THROW(pObj->runOperation("Evaluate Fisher Information", s));

  pObj->_likelihoodModel = "Positive Normal";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "StudentT";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "Positive StudentT";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "Poisson";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "Geometric";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_ANY_THROW(pObj->evaluateLogLikelihoodHessian(s));
  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  ASSERT_NO_THROW(pObj->evaluateLoglikelihoodGradient(s));
  ASSERT_NO_THROW(pObj->evaluateLogLikelihoodHessian(s));

  modelFc = [](Sample& s)
  {
   s["F(x)"] = 0.1;
   s["Reference Evaluations"] = std::vector<double>({-0.1});
   s["Standard Deviation"] = std::vector<double>({0.1});
   s["Degrees Of Freedom"] = std::vector<double>({0.1});
   s["Dispersion"] = std::vector<double>({0.1});

   s["Gradient Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Gradient Dispersion"] = std::vector<std::vector<double>>({{0.1}});

   s["Hessian Mean"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Standard Deviation"] = std::vector<std::vector<double>>({{0.1}});
   s["Hessian Dispersion"] = std::vector<std::vector<double>>({{0.1}});
  };

  pObj->_likelihoodModel = "Negative Binomial";
  ASSERT_NO_THROW(pObj->evaluateLoglikelihood(s));
  pObj->_referenceData[0] = -1.0;
  ASSERT_ANY_THROW(pObj->evaluateLoglikelihood(s));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Computational Model");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Computational Model"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Computational Model"] = 1;
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Reference Data");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Reference Data"] = "Not a Number";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Reference Data"] = std::vector<double>({0.0});
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs.erase("Likelihood Model");
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = "Unknown";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = 1;
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  problemJs["Likelihood Model"] = "Normal";
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));
 }

 TEST(Problem, HierarchicalPsi)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  knlohmann::json uniformDistroJs;
  uniformDistroJs["Type"] = "Univariate/Uniform";
  uniformDistroJs["Minimum"] = 0.0;
  uniformDistroJs["Maximum"] = 1.0;
  auto uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(uniformDistroJs, &e));
  uniformGenerator->applyVariableDefaults();
  uniformGenerator->applyModuleDefaults(uniformDistroJs);
  uniformGenerator->setConfiguration(uniformDistroJs);
  e._distributions.push_back(uniformGenerator);
  e._distributions[0]->_name = "Uniform";

  // Creating initial variable
  Variable v;
  v._precomputedValues = std::vector<double>({0.0});
  e._variables.push_back(&v);
  e["Solver"]["Type"] = "Sampler/TMCMC";
  e["Solver"]["Sample Database"] = std::vector<std::vector<double>>({{0.0}});
  e["Solver"]["Sample LogPrior Database"] = std::vector<double>({0.0});
  e["Solver"]["Sample LogLikelihood Database"] = std::vector<double>({0.0});
  e["Variables"][0]["Name"] = "Var X";
  e["Variables"][0]["Prior Distribution"] = "Uniform";

  // Configuring Problem
  Psi* pObj;
  knlohmann::json problemJs;
  problemJs["Type"] = "Hierarchical/Psi";
  problemJs["Sub Experiments"] = std::vector<knlohmann::json>({e._js.getJson(), e._js.getJson()});
  problemJs["Conditional Priors"] = std::vector<std::string>({"Uniform"});

  ASSERT_NO_THROW(pObj = dynamic_cast<Psi *>(Module::getModule(problemJs, &e)));
  e._problem = pObj;

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(pObj->applyModuleDefaults(problemJs));

  // Covering variable functions (no effect)
  pObj->applyVariableDefaults();
  ASSERT_NO_THROW(pObj->applyVariableDefaults());

  // Trying to run unknown operation
  Sample s;
  ASSERT_ANY_THROW(pObj->runOperation("Unknown", s));

  // Backup the correct base configuration
  auto baseProbJs = problemJs;
  auto baseExpJs = experimentJs;

  // Testing correct configuration
  ASSERT_NO_THROW(pObj->setConfiguration(problemJs));

  // Testing unrecognized solver
  problemJs = baseProbJs;
  experimentJs = baseExpJs;
  e["Solver"]["Type"] = "";
  ASSERT_ANY_THROW(pObj->setConfiguration(problemJs));

  // Testing initialization
  pObj->initialize();
  ASSERT_NO_THROW(pObj->initialize());

 }
} // namespace
