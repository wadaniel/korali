#include "constraints.h"
#include "stdio.h"
#include "math.h"

double g1(std::vector<double> x)
{
 return -127.0 + 2 * x[0] * x[0] + 3.0 * pow(x[1], 4) + x[2] + 4.0 * x[3] * x[3] + 5.0 * x[4]; 
}

double g2(std::vector<double> x)
{
 return -282.0 + 7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2] * x[2] + x[3] - x[4]; 
}

double g3(std::vector<double> x)
{
 return -196.0 + 23.0 * x[0] + x[1] * x[1] + 6.0 * x[5] * x[5] - 8.0 * x[6]; 
}

double g4(std::vector<double> x)
{
 return  4.0 * x[0] * x[0] + x[1] * x[1] - 3.0 * x[0] * x[1] + 2.0 * x[2] * x[2] + 5.0 * x[5] - 11.0 * x[6];
}

