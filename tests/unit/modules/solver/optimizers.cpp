#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/optimizer/Adam/Adam.hpp"
#include "modules/solver/optimizer/AdaBelief/AdaBelief.hpp"

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

} // namespace
