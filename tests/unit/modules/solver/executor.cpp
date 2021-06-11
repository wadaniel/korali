#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/executor/executor.hpp"
#include "modules/problem/propagation/propagation.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::problem;

 //////////////// Executor CLASS ////////////////////////

  TEST(samplers, executorClass)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating optimizer configuration Json
   knlohmann::json executorJs;
   executorJs["Type"] = "Executor";

   // Creating module
   Executor* exec;
   ASSERT_NO_THROW(exec = dynamic_cast<Executor *>(Module::getModule(executorJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(exec->applyModuleDefaults(executorJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(exec->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = executorJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(exec->setConfiguration(executorJs));

   // Testing optional parameters
   executorJs = baseOptJs;
   experimentJs = baseExpJs;
   executorJs["Sample Count"] = "Not a Number";
   ASSERT_ANY_THROW(exec->setConfiguration(executorJs));

   executorJs = baseOptJs;
   experimentJs = baseExpJs;
   executorJs["Sample Count"] = 1;
   ASSERT_NO_THROW(exec->setConfiguration(executorJs));

   executorJs = baseOptJs;
   experimentJs = baseExpJs;
   executorJs.erase("Executions Per Generation");
   ASSERT_ANY_THROW(exec->setConfiguration(executorJs));

   executorJs = baseOptJs;
   experimentJs = baseExpJs;
   executorJs["Executions Per Generation"] = "Not a Number";
   ASSERT_ANY_THROW(exec->setConfiguration(executorJs));

   executorJs = baseOptJs;
   experimentJs = baseExpJs;
   executorJs["Executions Per Generation"] = 1;
   ASSERT_NO_THROW(exec->setConfiguration(executorJs));
  }


} // namespace
