#include <cmath>

#include "ellipsoid.h"

void ellipsoid(std::vector<double>& pars, std::vector<double>& result)
{
 double res = 0.0;
 size_t npar = pars.size();
 for(size_t i = 0; i < npar; ++i)
    res += std::pow( 10, 6.0*i/((double)npar)) * std::round(pars[i]) * std::round(pars[i]);
 
 result.push_back(-res);
}

