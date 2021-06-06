#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/optimizer/AdaBelief/AdaBelief.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::solver::optimizer;

 TEST(Conduit, SequentialConduit)
 {
  // Creating base experiment
  Experiment e;
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

  // Setting up optimizer
  ASSERT_NO_THROW(opt->setConfiguration(optimizerJs));
 }

 TEST(Conduit, VariableFails)
 {
  // Creating base experiment
  Experiment e;
  Variable v;
  v._initialValue = std::numeric_limits<double>::infinity();
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

  // A complaint for initial value to be Infinity should happen now
  ASSERT_ANY_THROW(opt->setInitialConfiguration());
 }

} // namespace
