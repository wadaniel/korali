#include "model.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"

void model(std::vector<double>& pars, std::vector<double>& result)
{
 double res = pow(pars[0] - 10.0, 2.0) + 5.0 * pow(pars[1] - 12.0, 2.0)
       + pow(pars[2], 4.0)  + 3.0 * pow(pars[3] - 11.0, 2.0)
       + 10.0 * pow(pars[4], 6.0) + 7.0 * pow(pars[5], 2.0) + pow(pars[6], 4.0)
       - 4.0 * pars[5] * pars[6] - 10.0 * pars[5] - 8.0 * pars[6];

 result.push_back(-res);
}
