#include "korali.hpp"

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Problem::Sampling::getConfiguration()
{
 _k->_js["Problem"]["Type"] = "Sampling";
}

void Korali::Problem::Sampling::setConfiguration()
{


}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::Sampling::initialize()
{
 if (_k->_constraints.size() > 0) koraliError("Sampling problems do not allow constraint definitions.\n");
 if (_k->_modelDefined == false) koraliError("Sampling problems require defining a computational model.\n");
}

void Korali::Problem::Sampling::finalize()
{

}

void Korali::Problem::Sampling::packVariables(double* sample, Korali::Model& data)
{
 for (size_t i = 0; i < _k->N; i++) data._computationalVariables.push_back(sample[i]);
}

double Korali::Problem::Sampling::evaluateFitness(Korali::Model& data)
{

 if (data._results.size() != 1)
  koraliError("Sampling problems require exactly a 1-element result array. Provided: %lu.\n", data._results.size());

 return data._results[0];
}

double Korali::Problem::Sampling::evaluateLogPrior(double* sample)
{
  return 0.0;
}
