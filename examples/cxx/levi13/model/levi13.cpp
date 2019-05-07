#include "levi13.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void eggholder(std::vector<double>& pars, std::vector<double>& result)
{
 size_t nPars = pars.size();
 if( nPars != 2 ) { printf("eggholder: error, num pars must be 2 (is %zu)\n", nPars); exit(-1); }

 double s = std::pow(std::sin(3*M_PI*pars[0]),2)+std::pow((pars[0]-1),2)*(1.0+std::pow(std::sin(3*M_PI*pars[1]),2))+std::pow(pars[1]-1,2)*(1.0+std::pow(std::sin(2.0*M_PI*pars[1]),2));

 result.push_back(s);
}
