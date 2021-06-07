#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/optimizer/Adam/Adam.hpp"
#include "modules/solver/optimizer/AdaBelief/AdaBelief.hpp"
#include "modules/solver/optimizer/CMAES/CMAES.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::solver::optimizer;

 TEST(optimizers, AdaBelief)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;

  // Creating optimizer configuration Json
  knlohmann::json optimizerJs;
  optimizerJs["Type"] = "Optimizer/AdaBelief";

  // Creating module
  AdaBelief* opt;
  ASSERT_NO_THROW(opt = dynamic_cast<AdaBelief *>(Module::getModule(optimizerJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(opt->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = optimizerJs;
  auto baseExpJs = experimentJs;

  // Setting up optimizer correctly
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing infinite initial value fail
  v._initialValue = std::numeric_limits<double>::infinity();
  ASSERT_ANY_THROW(opt->setInitialConfiguration());

  // Testing initial configuration success
  v._initialValue = 1.0;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing optional parameters
  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Variable"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Variable"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Norm"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Norm"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["First Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["First Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected First Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected First Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Second Central Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Second Central Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected Second Central Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected Second Central Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta1"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Beta1");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta1"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta2"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Beta2");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta2"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Eta"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Eta");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Eta"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Epsilon"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Epsilon");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Epsilon"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Gradient Norm"] = std::vector<double>({ 1.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Gradient Norm");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Gradient Norm"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Gradient Norm"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Gradient Norm");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Gradient Norm"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing termination criteria
  e._currentGeneration = 2;
  opt->_gradientNorm = 0.0;
  ASSERT_TRUE(opt->checkTermination());
  opt->_gradientNorm = 1.5;
  ASSERT_FALSE(opt->checkTermination());
  opt->_gradientNorm = 3.0;
  ASSERT_TRUE(opt->checkTermination());
 }

 TEST(optimizers, Adam)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;

  // Creating optimizer configuration Json
  knlohmann::json optimizerJs;
  optimizerJs["Type"] = "Optimizer/Adam";

  // Creating module
  Adam* opt;
  ASSERT_NO_THROW(opt = dynamic_cast<Adam *>(Module::getModule(optimizerJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(opt->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = optimizerJs;
  auto baseExpJs = experimentJs;

  // Setting up optimizer correctly
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing infinite initial value fail
  v._initialValue = std::numeric_limits<double>::infinity();
  ASSERT_ANY_THROW(opt->setInitialConfiguration());

  // Testing initial configuration success
  v._initialValue = 1.0;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing optional parameters
  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Variable"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Variable"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Ever Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Squared Gradient"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Squared Gradient"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Norm"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Norm"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["First Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["First Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected First Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected First Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Second Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Second Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected Second Moment"] = std::vector<double>({ 2.0 });
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Bias Corrected Second Moment"] = 2.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta1"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Beta1");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta1"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta2"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Beta2");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Beta2"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Eta"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Eta");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Eta"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Epsilon"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Epsilon");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Epsilon"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Gradient Norm"] = std::vector<double>({ 1.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Gradient Norm");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Gradient Norm"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Gradient Norm"] = std::vector<double>({ 2.0 });
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Gradient Norm");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Gradient Norm"] = 2.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing termination criteria
  e._currentGeneration = 2;
  opt->_gradientNorm = 0.0;
  ASSERT_TRUE(opt->checkTermination());
  opt->_gradientNorm = 1.5;
  ASSERT_FALSE(opt->checkTermination());
  opt->_gradientNorm = 3.0;
  ASSERT_TRUE(opt->checkTermination());
 }

 TEST(optimizers, CMAES)
 {
  // Creating base experiment
  Experiment e;
  auto& experimentJs = e._js.getJson();

  // Creating initial variable
  Variable v;
  e._variables.push_back(&v);
  e["Variables"][0]["Name"] = "Var 1";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 1.0;

  // Creating optimizer configuration Json
  knlohmann::json optimizerJs;
  optimizerJs["Type"] = "Optimizer/CMAES";

  // Creating module
  CMAES* opt;
  ASSERT_NO_THROW(opt = dynamic_cast<CMAES *>(Module::getModule(optimizerJs, &e)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(opt->applyModuleDefaults(optimizerJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(opt->applyVariableDefaults());

  // Backup the correct base configuration
  auto baseOptJs = optimizerJs;
  auto baseExpJs = experimentJs;

  // Setting up optimizer correctly
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  // Testing initial configuration success
  v._initialValue = 1.0;
  ASSERT_NO_THROW(opt->setInitialConfiguration());

  // Testing optional parameters
  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Viability Regime"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Viability Regime"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Value Vector"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Value Vector"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Value Vector"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Value Vector"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradients"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradients"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Population Size"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Population Size"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mu Value"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mu Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Weights"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Weights"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Effective Mu"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Effective Mu"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sigma Cumulation Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sigma Cumulation Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Damp Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Damp Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Cumulative Covariance"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Cumulative Covariance"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Chi Square Number"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Chi Square Number"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Eigenvalue Evaluation Frequency"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Eigenvalue Evaluation Frequency"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sigma"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sigma"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Trace"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Trace"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Population"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Population"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Finished Sample Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Finished Sample Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variables"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Best Variables"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Value"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Sample Index"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Sample Index"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Ever Value"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Best Ever Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sorting Index"] = std::vector<size_t>(1.0);
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sorting Index"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Coveriance Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Covariance Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Coveriance Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Eigenvector Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Eigenvector Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Covariance Eigenvector Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar Covariance Eigenvector Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Axis Lengths"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Axis Lengths"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["BDZ Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["BDZ Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar BDZ Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Auxiliar BDZ Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mean"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Mean"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Mean"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Previous Mean"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mean Update"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mean Update"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Evolution Path"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Evolution Path"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Conjugate Evolution Path"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Conjugate Evolution Path"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Conjugate Evolution Path L2 Norm"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Conjugate Evolution Path L2 Norm"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Infeasible Sample Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Infeasible Sample Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Maximum Diagonal Covariance Matrix Element"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Maximum Diagonal Covariance Matrix Element"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Minimum Diagonal Covariance Matrix Element"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Minimum Diagonal Covariance Matrix Element"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Maximum Covariance Eigenvalue"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Maximum Covariance Eigenvalue"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Minimum Covariance Eigenvalue"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Minimum Covariance Eigenvalue"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Eigensystem Updated"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Eigensystem Updated"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Indicator"] = std::vector<std::vector<int>>({{1}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Indicator"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Are Constraints Defined"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Are Constraints Defined"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaption Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaption Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Valid Sample"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Valid Sample"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Global Success Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Global Success Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Function Value"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Function Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Resampled Parameter Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Resampled Parameter Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaptation Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaptation Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Boundaries"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Boundaries"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Improvement"] = std::vector<int>({1});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Improvement"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Constraint Violation Count"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Constraint Violation Count"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Constraint Violation Counts"] = std::vector<size_t>({1});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Sample Constraint Violation Counts"] = 1;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Constraint Evaluations"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Constraint Evaluations"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Normal Constraint Approximation"] = std::vector<std::vector<double>>({{1.0}});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Normal Constraint Approximation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Constraint Evaluations"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Best Constraint Evaluations"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Has Discrete Variables"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Has Discrete Variables"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Discrete Mutations"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Discrete Mutations"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Number Of Discrete Mutations"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Number Of Discrete Mutations"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Number Masking Matrix Entries"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Number Masking Matrix Entries"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Masking Matrix"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Masking Matrix"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Masking Matrix Sigma"] = std::vector<double>({1.0});
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Masking Matrix Sigma"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Chi Square Number Discrete Mutations"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Chi Square Number Discrete Mutations"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Min Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Min Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Max Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Current Max Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Population Size"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Population Size");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Population Size"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Value"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Mu Value");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Type"] = "Linear";
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Type"] = 1.0;
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Mu Type"] = "Undefined";
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Mu Type");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Sigma Cumulation Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Sigma Cumulation Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Initial Sigma Cumulation Factor");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Damp Factor"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Damp Factor"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Initial Damp Factor");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Use Gradient Information"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Use Gradient Information");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Use Gradient Information"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Step Size"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Gradient Step Size");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Gradient Step Size"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Sigma Bounded"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Is Sigma Bounded");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Sigma Bounded"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Cumulative Covariance"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Initial Cumulative Covariance");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Initial Cumulative Covariance"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Diagonal"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Is Diagonal");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Is Diagonal"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Population Size"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Viability Population Size");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Population Size"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Mu Value"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Viability Mu Value");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Viability Mu Value"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Covariance Matrix Corrections"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Max Covariance Matrix Corrections");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Max Covariance Matrix Corrections"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Target Success Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Target Success Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Target Success Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaption Strength"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Covariance Matrix Adaption Strength");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Covariance Matrix Adaption Strength"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Normal Vector Learning Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Normal Vector Learning Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Normal Vector Learning Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Global Success Learning Rate"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs.erase("Global Success Learning Rate");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Global Success Learning Rate"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Infeasible Resamplings"] = 1;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Infeasible Resamplings");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Infeasible Resamplings"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Condition Covariance Matrix"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Condition Covariance Matrix");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Condition Covariance Matrix"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Min Standard Deviation");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Min Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"].erase("Max Standard Deviation");
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));

  optimizerJs = baseOptJs;
  experimentJs = baseExpJs;
  optimizerJs["Termination Criteria"]["Max Standard Deviation"] = std::vector<double>({1.0});
  ASSERT_ANY_THROW(opt->setConfiguration(optimizerJs));
 }

} // namespace
