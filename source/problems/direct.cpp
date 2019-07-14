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
 if (_k->_modelDefined == false)
 {
	fprintf(stderr, "[Korali] Error: Direct Problem requires defining a computational model.\n");
	exit(-1);
 }

 if (_k->_likelihoodDefined == true)
 {
	fprintf(stderr, "[Korali] Error: Direct Problem does not accept a likelihood function, only a computational model.\n");
	exit(-1);
 }
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
 {
  fprintf(stderr, "[Korali] Error: Direct Evaluation type requires exactly a 1-element result array.\n");
  fprintf(stderr, "[Korali]        Provided: %lu.\n", data._results.size());
  exit(-1);
 }

 return data._results[0];
}

double Korali::Problem::Direct::evaluateLogPrior(double* sample)
{
  return 0.0;
}
