#include "tr2.h"
#include <stdlib.h>
#include <math.h>

// Minus Rosenbrock
void m_tr2(std::vector<double>& pars, std::vector<double>& result)
{

 double res = pars[0]*pars[0] + pars[1]*pars[1];
 result.push_back(-res);
}
