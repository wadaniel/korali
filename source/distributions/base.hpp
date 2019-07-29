#ifndef _KORALI_DISTRIBUTION_BASE_HPP_
#define _KORALI_DISTRIBUTION_BASE_HPP_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>

namespace Korali { namespace Distribution {

class Base {

 private:

 size_t _seed;
 gsl_rng* _range;
 double _aux;

 public:

 virtual void initialize()
 {
  _seed = _k->_seed++;
  _range = gsl_rng_alloc (gsl_rng_default);
  gsl_rng_set(_range, _seed);
 }

 virtual void finalize()
 {
  gsl_rng_free(_range);
 }

 virtual double getDensity(double x) = 0;
 virtual double getLogDensity(double x) = 0;
 virtual double getRandomNumber() = 0;
 virtual void finalize() = 0;
 virtual void setConfiguration() = 0;
 virtual void getConfiguration() = 0;
};

} } // namespace Korali::Distribution

#endif // _KORALI_DISTRIBUTION_BASE_HPP_
