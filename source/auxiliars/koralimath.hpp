#ifndef _KORALI_AUXILIARS_KORALIMATH_HPP_
#define _KORALI_AUXILIARS_KORALIMATH_HPP_

#include "korali.hpp"
#include <stdlib.h>
#include <vector>

namespace Korali
{

const double NaN = std::numeric_limits<double>::quiet_NaN();
const double Inf = std::numeric_limits<double>::infinity();
const double Lowest = std::numeric_limits<double>::lowest();
const double Max = std::numeric_limits<double>::max();

double logSumExp( std::vector<double> logValues );

}

#endif // _KORALI_AUXILIARS_KORALIMATH_HPP_
