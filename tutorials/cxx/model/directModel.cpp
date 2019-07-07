#include "directModel.h"

void directModel(std::vector<double>& pars, std::vector<double>& result)
{
 double x = pars[0];
 double r = -x + 0.5;
 result.push_back(r);
}

