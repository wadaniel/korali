#ifndef _KORALI_MULTINOMIAL_HPP_
#define _KORALI_MULTINOMIAL_HPP_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <auxiliars/json.hpp>

namespace Korali { namespace Distribution {

class Multinomial {

 private:

 gsl_rng* _range;

 public:

 Multinomial(size_t seed);
 ~Multinomial();

 void getSelections(std::vector<double>& p, std::vector<unsigned int>& n, int N);
};

} } // namespace Korali::Distribution

#endif // _KORALI_MULTINOMIAL_HPP_
