#include "gtest/gtest.h"
#include "korali.hpp"

namespace
{
 using namespace korali;

 TEST(Conduit, BetaDistribution)
 {
//  knlohmann::json distributionJs;
//  distributionJs["Type"] = "Distribution/Univariate/Beta";
//
//  // Creating distribution
//  Distribution* dist;
//  ASSERT_NO_THROW(dist = dynamic_cast<korali::distribution::univariate::Beta *>(Module::getModule(distributionJs, NULL)));

//  double @className::getDensity(const double x) const
//  {
//    return gsl_ran_beta_pdf(x, _alpha, _beta);
//  }
//
//  double @className::getLogDensity(const double x) const
//  {
//    if (x < 0.) return -INFINITY;
//    if (x > 1.) return -INFINITY;
//    return _aux + (_alpha - 1.) * std::log(x) + (_beta - 1.) * std::log(1. - x);
//  }
//
//  double @className::getRandomNumber()
//  {
//    return gsl_ran_beta(_range, _alpha, _beta);
//  }
//
//  double @className::getLogDensityGradient(const double x) const
//  {
//    if (x < 0.) return 0.;
//    if (x > 1.) return 0.;
//    return (_alpha - 1.) / x - (_beta - 1.) / (1. - x);
//  }
//
//  double @className::getLogDensityHessian(const double x) const
//  {
//    if (x < 0.) return 0.;
//    if (x > 1.) return 0.;
//    return (1. - _alpha) / (x * x) - (_beta - 1.) / ((1. - x) * (1. - x));
//  }
//
//  void @className::updateDistribution()
//  {
//    if (_alpha <= 0.0) KORALI_LOG_ERROR("Incorrect Shape parameter (alpha) of Beta distribution: %f.\n", _alpha);
//    if (_beta <= 0.0) KORALI_LOG_ERROR("Incorrect Shape (beta) parameter of Beta distribution: %f.\n", _beta);
//
//    _aux = gsl_sf_lngamma(_alpha + _beta) - gsl_sf_lngamma(_alpha) - gsl_sf_lngamma(_beta);
//  }
 }


} // namespace
