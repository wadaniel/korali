#include "gtest/gtest.h"
#include "korali.hpp"

namespace
{
 using namespace korali;

 TEST(Conduit, BaseDistribution)
 {
  knlohmann::json distributionJs;
  distributionJs["Type"] = "Distribution";

  // Expecting exception b4ecause the distribution name is not given
  Distribution* dist;
  ASSERT_THROW(dist = dynamic_cast<Distribution *>(Module::getModule(distributionJs, NULL)));
 }


} // namespace
