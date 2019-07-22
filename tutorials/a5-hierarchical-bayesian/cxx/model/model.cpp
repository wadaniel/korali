#include "model.h"

void myLikelihood(Korali::Model& k)
{
 double x = k.getVariable(0);
 double r =  -0.5*x*x;
 k.addResult(r);
}

