#include "rosenbrock.h"
#include <stdlib.h>
#include <math.h>

void rosenbrock(std::vector<double>& pars, std::vector<double>& result)
{
 size_t nPars = pars.size();

 double s = 0.;
 for (size_t i = 0; i < nPars-1; i++)
  s += 100.0*pow(pars[i+1]-pars[i]*pars[i], 2) + pow(1.0 - pars[i], 2);

 result.push_back(-s);
}
