#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Problem::Bayesian::Bayesian(nlohmann::json& js) : Korali::Problem::Base::Base(js)
{
 setConfiguration(js);
}

Korali::Problem::Bayesian::~Bayesian()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Problem::Bayesian::getConfiguration()
{
 auto js = this->Korali::Problem::Base::getConfiguration();

 js["Evaluation Type"] = "Bayesian";

 for (size_t i = 0; i < _referenceDataSize; i++) js["Reference Data"][i] = _referenceData[i];

 return js;
}

void Korali::Problem::Bayesian::setConfiguration(nlohmann::json& js)
{
 auto ref = consume(js, { "Reference Data" }, KORALI_ARRAY);
 _referenceDataSize = ref.size();
 _referenceData = (double*) calloc (_referenceDataSize, sizeof(double));
 for (size_t i = 0; i < _referenceDataSize; i++) _referenceData[i] = ref[i];

 if (_referenceDataSize == 0)
 {
  fprintf(stderr, "[Korali] Error: No Reference Data set provided for the Bayesian Model.\n");
  exit(-1);
 }
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::Bayesian::initialize()
{

 _isBayesian = true;

 if (_k->_modelDefined == false)
 {
  fprintf(stderr, "[Korali] Error: Bayesian Problem requires defining a computational model.\n");
  exit(-1);
 }

 if (_k->_likelihoodDefined == true)
 {
  fprintf(stderr, "[Korali] Error: Bayesian Problem does not accept a likelihood function, only a computational model.\n");
  exit(-1);
 }

 if (_statisticalVariableCount != 1)
 {
  fprintf(stderr, "[Korali] Error: The Bayesian model requires 1 statistical parameter.\n");
  exit(-1);
 }

 for (size_t i = 0; i < N; i++)
  if (_variables[i]->_hasDistribution == false)
  {
   fprintf(stderr, "[Korali] Error: Variable %s has no defined distribution as required by a Bayesian model.\n", _variables[i]->_name.c_str());
   exit(-1);
  }
}

double Korali::Problem::Bayesian::evaluateFitness(Korali::ModelData& data)
{

 if (data._results.size() != _referenceDataSize)
 {
  fprintf(stderr, "[Korali] Error: This Bayesian Model requires a %lu-sized result array.\n", _referenceDataSize);
  fprintf(stderr, "[Korali]        Provided: %lu.\n", data._results.size());
  exit(-1);
 }

 double sigma = data._statisticalVariables[0];

 double fitness = Korali::Variable::Gaussian::logLikelihood(sigma, _referenceDataSize, _referenceData, data._results.data());

 return fitness;
}

double Korali::Problem::Bayesian::evaluateLogPrior(double* sample)
{
 double logPrior = 0.0;
 for (size_t i = 0; i < N; i++) logPrior += _variables[i]->getLogDensity(sample[i]);
 return logPrior;
}
