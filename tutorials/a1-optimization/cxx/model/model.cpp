#include "model.h"

void model(std::vector<double>& pars, std::vector<double>& result)
{
 double x = pars[0];
 double r =  -0.5*x*x;
 result.push_back(r);
}

