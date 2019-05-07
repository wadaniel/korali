#include "eggholder.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void eggholder(std::vector<double>& pars, std::vector<double>& result)
{
 size_t nPars = pars.size();
 if( nPars != 2 ) { printf("eggholder: error, num pars must be 2 (is %zu)\n", nPars); exit(-1); }

 double s = -(pars[1]+47.0)*std::sin(std::sqrt(std::abs(0.5*pars[0]+pars[1]+47.0))) - pars[0]*std::sin(std::sqrt(std::abs(pars[0]-pars[1]-47.0)));

 result.push_back(-s);
}
