#include "gtest/gtest.h"
#include "korali.hpp"

namespace
{

 using namespace korali;

 TEST(Engine, badParametersException)
 {
  Engine e;

  e._js["Condruit"] = "Typo";
  ASSERT_ANY_THROW(e.initialize());

  e._js["Conduit"] = "No Typo";
  ASSERT_ANY_THROW(e.initialize());
 }

} // namespace
