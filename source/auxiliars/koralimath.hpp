#ifndef _KORALI_AUXILIARS_KORALIMATH_HPP_
#define _KORALI_AUXILIARS_KORALIMATH_HPP_

// Auxiliar library for Korali's math and time operations.

#include "korali.hpp"
#include <stdlib.h>
#include <vector>

namespace Korali
{

const double NaN    = std::numeric_limits<double>::quiet_NaN();
const double Inf    = std::numeric_limits<double>::infinity();
const double Lowest = std::numeric_limits<double>::lowest();
const double Max    = std::numeric_limits<double>::max();
const double Min    = std::numeric_limits<double>::min();
const double Eps    = std::numeric_limits<double>::epsilon();

double logSumExp( std::vector<double> logValues );
std::string getTimestamp();
size_t getTimehash();

}

#endif // _KORALI_AUXILIARS_KORALIMATH_HPP_
