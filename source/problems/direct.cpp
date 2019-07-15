#include "korali.h"

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Problem::Direct::getConfiguration()
{
 _k->_js["Problem"] = "Direct Evaluation";
}

void Korali::Problem::Direct::setConfiguration()
{


}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::Direct::initialize()
{
 if (_k->_modelDefined == false) koraliError("Direct Problem requires defining a computational model.\n");
 if (_k->_likelihoodDefined == true) koraliError("Direct Problem does not accept a likelihood function, only a computational model.\n");
}

void Korali::Problem::Direct::finalize()
{

}

void Korali::Problem::Direct::packVariables(double* sample, Korali::Model& data)
{
 for (size_t i = 0; i < _k->N; i++) data._computationalVariables.push_back(sample[i]);
}

double Korali::Problem::Direct::evaluateFitness(Korali::Model& data)
{

 if (data._results.size() != 1)
  koraliError("Direct Evaluation type requires exactly a 1-element result array. Provided: %lu.\n", data._results.size());

 return data._results[0];
}

double Korali::Problem::Direct::evaluateLogPrior(double* sample)
{
  return 0.0;
}
