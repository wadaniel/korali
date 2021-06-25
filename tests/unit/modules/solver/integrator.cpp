#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/integrator/integrator.hpp"
#include "modules/problem/integration/integration.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::problem;

 //////////////// Integrator CLASS ////////////////////////

  TEST(samplers, executorClass)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();

   // Creating optimizer configuration Json
   knlohmann::json integratorJs;
   integratorJs["Type"] = "Integrator";

   // Creating module
   Integrator* itr;
   ASSERT_NO_THROW(itr = dynamic_cast<Integrator *>(Module::getModule(integratorJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(itr->applyModuleDefaults(integratorJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(itr->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = integratorJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(itr->setConfiguration(integratorJs));

   // Testing optional parameters
   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Sample Count"] = "Not a Number";
   ASSERT_ANY_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Sample Count"] = 1;
   ASSERT_NO_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Integral"] = "Not a Number";
   ASSERT_ANY_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Integral"] = 1.0;
   ASSERT_NO_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Indices Helper"] = "Not a Number";
   ASSERT_ANY_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Indices Helper"] = std::vector<double>({1.0});
   ASSERT_NO_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs.erase("Executions Per Generation");
   ASSERT_ANY_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Executions Per Generation"] = "Not a Number";
   ASSERT_ANY_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Executions Per Generation"] = 1;
   ASSERT_NO_THROW(itr->setConfiguration(integratorJs));
  }


} // namespace
