// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#ifndef __GP_UTILS_H__
#define __GP_UTILS_H__

#include <cstdlib>
#include <cmath>
#include <cassert>

namespace libgp {

  /** Various auxiliary functions. */
  class Utils {
  public:
    /** Generate independent standard normally distributed (zero mean,
     *  unit variance) random numbers using the Box–Muller transform. */
    static double randn();

    /** Sign function.
     */
    static double sign(double x);

  };
}

#endif /* __GP_UTILS_H__ */
