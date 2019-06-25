#include "directModel.h"

void directModel(std::vector<double>& pars, std::vector<double>& result)
{
 double x = pars[0];
 double r = -x*x;
 result.push_back(r);
}

