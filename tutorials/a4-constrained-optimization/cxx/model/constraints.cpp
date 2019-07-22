#include "constraints.h"
#include "stdio.h"
#include "math.h"

void g1(Korali::Model& k)
{
 std::vector<double> x = k.getVariables();
 double c = -127.0 + 2 * x[0] * x[0] + 3.0 * pow(x[1], 4) + x[2] + 4.0 * x[3] * x[3] + 5.0 * x[4];
 k.addResult(c);
}

void g2(Korali::Model& k)
{
 std::vector<double> x = k.getVariables();
 double c = -282.0 + 7.0 * x[0] + 3.0 * x[1] + 10.0 * x[2] * x[2] + x[3] - x[4];
 k.addResult(c);
}

void g3(Korali::Model& k)
{
 std::vector<double> x = k.getVariables();
 double c = -196.0 + 23.0 * x[0] + x[1] * x[1] + 6.0 * x[5] * x[5] - 8.0 * x[6];
 k.addResult(c);
}

void g4(Korali::Model& k)
{
 std::vector<double> x = k.getVariables();
 double c = 4.0 * x[0] * x[0] + x[1] * x[1] - 3.0 * x[0] * x[1] + 2.0 * x[2] * x[2] + 5.0 * x[5] - 11.0 * x[6];
 k.addResult(c);
}

