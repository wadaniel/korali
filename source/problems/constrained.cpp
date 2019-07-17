#include "korali.h"

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Problem::Constrained::getConfiguration()
{
 _k->_js["Problem"] = "Constrained Evaluation";
}

void Korali::Problem::Constrained::setConfiguration()
{


}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::Constrained::initialize()
{
 if (_k->_constraints.size() < 1) koraliError("Constrained Problem requires defining at least one constraint.\n");
 if (_k->_modelDefined == false) koraliError("Constrained Problem requires defining a computational model.\n");
 if (_k->_likelihoodDefined == true) koraliError("Constrained Problem does not accept a likelihood function, only a computational model.\n");
}

void Korali::Problem::Constrained::finalize()
{

}

void Korali::Problem::Constrained::packVariables(double* sample, Korali::Model& data)
{
 for (size_t i = 0; i < _k->N; i++) data._computationalVariables.push_back(sample[i]);
}

double Korali::Problem::Constrained::evaluateFitness(Korali::Model& data)
{

 if (data._results.size() != 1)
  koraliError("Constrained Evaluation type requires exactly a 1-element result array. Provided: %lu.\n", data._results.size());

 return data._results[0];
}

double Korali::Problem::Constrained::evaluateLogPrior(double* sample)
{
  return 0.0;
}
