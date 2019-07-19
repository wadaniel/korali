#include "korali.h"

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Problem::Optimization::getConfiguration()
{
 _k->_js["Problem"] = "Optimization";
}

void Korali::Problem::Optimization::setConfiguration()
{


}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::Optimization::initialize()
{
 if (_k->_constraints.size() > 0) koraliError("Optimization problems do not allow constraint definitions.\n");
 if (_k->_modelDefined == false) koraliError("Optimization problems require defining a computational model.\n");
 if (_k->_likelihoodDefined == true) koraliError("Optimization problems do not accept a likelihood function, only a computational model.\n");
}

void Korali::Problem::Optimization::finalize()
{

}

void Korali::Problem::Optimization::packVariables(double* sample, Korali::Model& data)
{
 for (size_t i = 0; i < _k->N; i++) data._computationalVariables.push_back(sample[i]);
}

double Korali::Problem::Optimization::evaluateFitness(Korali::Model& data)
{

 if (data._results.size() != 1)
  koraliError("Optimization problems require exactly a 1-element result array. Provided: %lu.\n", data._results.size());

 return data._results[0];
}

double Korali::Problem::Optimization::evaluateLogPrior(double* sample)
{
  return 0.0;
}
