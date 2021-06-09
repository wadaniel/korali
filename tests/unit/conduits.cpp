#include "gtest/gtest.h"
#include "korali.hpp"

namespace
{
 using namespace korali;

 TEST(Conduit, SequentialConduit)
 {
  knlohmann::json conduitJs;
  conduitJs["Type"] = "Sequential";

  // Creating module
  Conduit* conduit;
  ASSERT_NO_THROW(conduit = dynamic_cast<Conduit *>(Module::getModule(conduitJs, NULL)));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(conduit->applyModuleDefaults(conduitJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(conduit->applyVariableDefaults());

  // Setting up conduit
  ASSERT_NO_THROW(conduit->setConfiguration(conduitJs));

  // Initializing server
  ASSERT_NO_THROW(conduit->initServer());

  // Broadcasting message
  knlohmann::json message;
  message["Conduit Action"] == "Terminate";
  ASSERT_NO_THROW(conduit->broadcastMessageToWorkers(message));
 }

 TEST(Conduit, ConcurrentConduit)
 {
  knlohmann::json conduitJs;
  conduitJs["Type"] = "Concurrent";

  // Creating module
  Conduit* conduit;
  ASSERT_NO_THROW(conduit = dynamic_cast<Conduit *>(Module::getModule(conduitJs, NULL)));

  // Testing configuration without mandatory field(s)
  ASSERT_ANY_THROW(conduit->setConfiguration(conduitJs));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(conduit->applyModuleDefaults(conduitJs));

  // Testing wrong value type (string) for the concurrent jobs parameter
  conduitJs["Concurrent Jobs"] = "16";
  ASSERT_ANY_THROW(conduit->setConfiguration(conduitJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(conduit->applyVariableDefaults());

  // Testing correct configuration value type
  conduitJs["Concurrent Jobs"] = 16;
  ASSERT_NO_THROW(conduit->setConfiguration(conduitJs));
 }

 TEST(Conduit, DistributedConduit)
 {
  knlohmann::json conduitJs;
  conduitJs["Type"] = "Distributed";

  // Creating module
  Conduit* conduit;
  ASSERT_NO_THROW(conduit = dynamic_cast<Conduit *>(Module::getModule(conduitJs, NULL)));

  // Testing configuration without mandatory field(s)
  ASSERT_ANY_THROW(conduit->setConfiguration(conduitJs));

  // Defaults should be applied without a problem
  ASSERT_NO_THROW(conduit->applyModuleDefaults(conduitJs));

  // Testing wrong value type (string) for the concurrent jobs parameter
  conduitJs["Ranks Per Worker"] = "16";
  ASSERT_ANY_THROW(conduit->setConfiguration(conduitJs));

  // Covering variable functions (no effect)
  ASSERT_NO_THROW(conduit->applyVariableDefaults());

  // Testing correct configuration value type
  conduitJs["Ranks Per Worker"] = 16;
  ASSERT_NO_THROW(conduit->setConfiguration(conduitJs));
 }

} // namespace
