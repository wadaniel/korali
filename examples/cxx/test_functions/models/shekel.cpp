#include "shekel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Minus Shekel
void m_shekel(std::vector<double>& pars, std::vector<double>& result)
{
 size_t nPars = pars.size();
 if( nPars != 4 ) { printf("shekel: error, num pars must be 4 (is %zu)\n", nPars); exit(-1); }


 std::vector<double> beta { 0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5 };


 std::vector<double> C { 4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0,
                        4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6,
                        4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0,
                        4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6 };

 double t;
 double s = 0.0;
 for(int i = 0; i < 10; ++i){
    for(int j = 0; j < 4; ++j) t = pow( pars[j] - C[i*4+j], 2) + beta[i];
    s -= 1.0/t;
 }

 result.push_back(-s);
}
