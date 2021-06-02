#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fCMAES.hpp"

namespace
{

 using namespace korali;

 TEST(fastOptimizers, fCMAES)
 {
  const size_t N = 4;
  fCMAES c(N, 32, 16);
 }

} // namespace
