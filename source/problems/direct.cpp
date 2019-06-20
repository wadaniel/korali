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

void Korali::Problem::Direct::getConfiguration(nlohmann::json& js)
{
 js["Problem"] = "Direct Evaluation";

 for (size_t i = 0; i < _k->N; i++)
 {
  js["Variables"][i]["Name"] = _k->_variables[i]->_name;
 }
}

void Korali::Problem::Direct::setConfiguration(nlohmann::json& js)
{
 if (isArray(js, { "Variables" } ))
 for (size_t i = 0; i < js["Variables"].size(); i++)
 {
  auto varName = consume(js["Variables"][i], { "Name" }, KORALI_STRING);
  _k->_variables.push_back(new Korali::Variable(varName));
 }
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
}

void Korali::Problem::Direct::packVariables(double* sample, Korali::ModelData& data)
{
 for (size_t i = 0; i < _k->N; i++) data._computationalVariables.push_back(sample[i]);
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
