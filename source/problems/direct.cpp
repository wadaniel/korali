#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Problem::Direct::Direct(nlohmann::json& js)
{
 setConfiguration(js);
}

Korali::Problem::Direct::~Direct()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Problem::Direct::getConfiguration()
{
 auto js = nlohmann::json();

 js["Evaluation Type"] = "Direct";

 return js;
}

void Korali::Problem::Direct::setConfiguration(nlohmann::json& js)
{
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::Direct::initialize()
{
 _isBayesian = false;

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

 if (_k->_statisticalVariableCount != 0)
 {
  fprintf(stderr, "[Korali] Error: Direct Evaluation type requires 0 statistical parameters.\n");
  exit(-1);
 }

}

double Korali::Problem::Direct::evaluateFitness(Korali::ModelData& data)
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
