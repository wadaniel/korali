#include "model.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"

void model(Korali::Model& k)
{
 std::vector<double> pars = k.getVariables();
 size_t npar = pars.size();
 
 double res = 0.0;
 for(size_t i = 0; i < npar; ++i)
 if( i == 0 || i == 1 || i == 3 || i == 6)
     res += std::pow( 10, 6.0*i/((double)npar)) * std::round(pars[i]) * std::round(pars[i]);
 else
     res += std::pow( 10, 6.0*i/((double)npar)) * pars[i] * pars[i];

 k.addResult(-res);
}
