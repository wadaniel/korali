#include "gtest/gtest.h"
#include "korali.hpp"

namespace
{

 using namespace korali;

 TEST(Conduit, ConcurrentConfiguration)
 {
  knlohmann::json conduitJs;
  conduitJs["Type"] = "Concurrent"

  // Creating module
  Conduit* conduit;
  ASSERT_OK(dynamic_cast<Conduit *>(getModule(conduitJs, _k)));

  // Defaults should be applied without a problem
  ASSERT_OK(conduit->applyModuleDefaults(conduitJs);

  // Testing wrong value type (string) for the concurrent jobs parameter
  conduitJs["Concurrent Jobs"] = "16";
  ASSERT_ANY_THROW(conduit->setConfiguration(conduitJs);

  // Covering variable functions (no effect)
  ASSERT_OK(conduit->applyVariableDefaults());
 }

} // namespace
