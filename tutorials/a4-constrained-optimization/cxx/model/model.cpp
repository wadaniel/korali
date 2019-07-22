#include "model.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"

void model(Korali::Model& k)
{
 std::vector<double> x = k.getVariables();
 double res = pow(x[0] - 10.0, 2.0) + 5.0 * pow(x[1] - 12.0, 2.0)
       + pow(x[2], 4.0)  + 3.0 * pow(x[3] - 11.0, 2.0)
       + 10.0 * pow(x[4], 6.0) + 7.0 * pow(x[5], 2.0) + pow(x[6], 4.0)
       - 4.0 * x[5] * x[6] - 10.0 * x[5] - 8.0 * x[6];

 k.addResult(-res);
}
