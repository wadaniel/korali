#include "levi13.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Minus Levi13
void m_levi13(std::vector<double>& pars, std::vector<double>& result)
{
 size_t nPars = pars.size();
 if( nPars != 2 ) { printf("levi13: error, num pars must be 2 (is %zu)\n", nPars); exit(-1); }

 double s = pow(sin(3*M_PI*pars[0]),2)+pow((pars[0]-1),2)*(1.0+pow(sin(3*M_PI*pars[1]),2))+pow(pars[1]-1,2)*(1.0+pow(sin(2.0*M_PI*pars[1]),2));

 result.push_back(-s);
}
