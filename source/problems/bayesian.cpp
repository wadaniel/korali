#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Problem::Bayesian::Bayesian(nlohmann::json& js)
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
 auto js = nlohmann::json();

 js["Evaluation Type"] = "Bayesian";

// if(_type == KORALI_COMPUTATIONAL) js["Type"] = "Computational";
// if(_type == KORALI_STATISTICAL)   js["Type"] = "Statistical";

 for (size_t i = 0; i < _referenceDataSize; i++) js["Reference Data"][i] = _referenceData[i];

 return js;
}

void Korali::Problem::Bayesian::setConfiguration(nlohmann::json& js)
{
 auto ref = consume(js, { "Reference Data" }, KORALI_ARRAY);
 _referenceDataSize = ref.size();
 _referenceData = (double*) calloc (_referenceDataSize, sizeof(double));
 for (size_t i = 0; i < _referenceDataSize; i++) _referenceData[i] = ref[i];

// auto typeString = consume(js, { "Type" }, KORALI_STRING, "Computational");
// if (typeString == "Computational")  _type = KORALI_COMPUTATIONAL;
// if (typeString == "Statistical")    _type = KORALI_STATISTICAL;

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
// _statisticalVariableCount = 0;
// _computationalVariableCount = 0;
// for (size_t i = 0; i < tmp.size(); i++) if (tmp[i]->_type == KORALI_COMPUTATIONAL) { _variables.push_back(tmp[i]); _computationalVariableCount++; }
// for (size_t i = 0; i < tmp.size(); i++) if (tmp[i]->_type == KORALI_STATISTICAL)   { _variables.push_back(tmp[i]); _statisticalVariableCount++; };
// N = _variables.size();

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

 if (_k->_statisticalVariableCount != 1)
 {
  fprintf(stderr, "[Korali] Error: The Bayesian model requires 1 statistical parameter.\n");
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
 for (size_t i = 0; i < _k->N; i++) logPrior += _k->_variables[i]->getLogDensity(sample[i]);
 return logPrior;
}
