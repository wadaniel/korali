#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Problem::DirectBayesian::DirectBayesian(nlohmann::json& js)
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
 auto js = nlohmann::json();

 js["Evaluation Type"] = "Direct Bayesian";

 return js;
}

void Korali::Problem::DirectBayesian::setConfiguration(nlohmann::json& js)
{
 auto ref = consume(js, { "Reference Data" }, KORALI_ARRAY);
 _referenceDataSize = ref.size();
 _referenceData = (double*) calloc (_referenceDataSize, sizeof(double));
 for (size_t i = 0; i < _referenceDataSize; i++) _referenceData[i] = ref[i];

 for (size_t i = 0; i < _k->N; i++ )
 {
  auto typeString = consume(js["Variables"][i], { "Type" }, KORALI_STRING, "Computational");
  if (typeString == "Computational") _computationalVariableIndices.push_back(i);
  if (typeString == "Statistical")   _statisticalVariableIndices.push_back(i);
 }
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::DirectBayesian::initialize()
{
 _isBayesian = true;

 if (_referenceDataSize != 0)
 {
  fprintf(stderr, "[Korali] Error: Reference Data is not required by the Direct Bayesian model.\n");
  exit(-1);
 }

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

 if (_statisticalVariableIndices.size() != 0)
 {
  fprintf(stderr, "[Korali] Error: Direct Bayesian Evaluation type requires no statistical parameters.\n");
  exit(-1);
 }
}

void Korali::Problem::DirectBayesian::packVariables(double* sample, Korali::ModelData& data)
{
 for (size_t i = 0; i < _computationalVariableIndices.size(); i++) data._computationalVariables.push_back(sample[i]);
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
 for (size_t i = 0; i < _k->N; i++) logPrior += _k->_variables[i]->getLogDensity(sample[i]);
 return logPrior;
}
