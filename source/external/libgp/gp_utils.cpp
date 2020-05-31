// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp_utils.hpp"

namespace libgp
{
double Utils::randn()
{
  double u1 = 1.0 - drand48(), u2 = 1.0 - drand48();
  return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

double Utils::sign(double x)
{
  if (x > 0) return 1.0;
  if (x < 0) return -1.0;
  return 0.0;
}
} // namespace libgp
