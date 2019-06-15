#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Problem::DirectBayesian::DirectBayesian(nlohmann::json& js) : Korali::Problem::Base::Base(js)
{
 setConfiguration(js);
}

Korali::Problem::DirectBayesian::~DirectBayesian()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Problem::DirectBayesian::getConfiguration()
{
 auto js = this->Korali::Problem::Base::getConfiguration();

 js["Evaluation Type"] = "Direct Bayesian";

 return js;
}

void Korali::Problem::DirectBayesian::setConfiguration(nlohmann::json& js)
{
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::DirectBayesian::initialize()
{

 _isBayesian = true;

 if (_k->_likelihoodDefined == false)
 {
  fprintf(stderr, "[Korali] Error: Direct Bayesian requires defining a likelihood function.\n");
  exit(-1);
 }

 if (_k->_modelDefined == true)
 {
  fprintf(stderr, "[Korali] Error: Direct Bayesian does not accept a computational model, only a likelihood function.\n");
  exit(-1);
 }

 if (_statisticalVariableCount != 0)
 {
  fprintf(stderr, "[Korali] Error: Direct Bayesian Evaluation type requires 0 statistical parameters.\n");
  exit(-1);
 }

 for (size_t i = 0; i < N; i++)
 if (_variables[i]->_hasDistribution == false)
 {
  fprintf(stderr, "[Korali] Error: Variable %s has no defined distribution as required by a Bayesian model.\n", _variables[i]->_name.c_str());
  exit(-1);
 }
}

double Korali::Problem::DirectBayesian::evaluateFitness(Korali::ModelData& data)
{

 if (data._results.size() != 1)
 {
  fprintf(stderr, "[Korali] Error: Direct Bayesian Evaluation type requires exactly a 1-element result array.\n");
  fprintf(stderr, "[Korali]        Provided: %lu.\n", data._results.size());
  exit(-1);
 }

 return data._results[0];
}

double Korali::Problem::DirectBayesian::evaluateLogPrior(double* sample)
{
 double logPrior = 0.0;
 for (size_t i = 0; i < N; i++) logPrior += _variables[i]->getLogDensity(sample[i]);
 return logPrior;
}
