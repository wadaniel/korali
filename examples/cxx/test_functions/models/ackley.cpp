#include "ackley.h"
#include <stdlib.h>
#include <math.h>

// Minus Ackley
void m_ackley(std::vector<double>& pars, std::vector<double>& result)
{
 size_t nDims = pars.size();
 double a = 20, b = .2, c = 2.*M_PI, s1 = 0., s2 = 0.;

 for (size_t i = 0; i < nDims; ++i) {
  s1 += pars[i]*pars[i];
  s2 += cos(c*pars[i]);
 }

 double res = -a*exp(-b*sqrt(s1/nDims)) - exp(s2/nDims) + a + exp(1.);

 result.push_back(-res);
}
