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

 js["Problem"] = "Bayesian";

 for (size_t i = 0; i < _referenceDataSize; i++) js["Reference Data"][i] = _referenceData[i];

 return js;
}

void Korali::Problem::Bayesian::setConfiguration(nlohmann::json& js)
{
 bool foundLikelihoodType = false;
 std::string likelihoodString = consume(js, { "Bayesian", "Likelihood", "Type" }, KORALI_STRING, "Undefined");
 if (likelihoodString == "Direct")    { _likelihood = DirectLikelihood;    foundLikelihoodType = true; }
 if (likelihoodString == "Reference") { _likelihood = ReferenceLikelihood; foundLikelihoodType = true; }
 if (foundLikelihoodType == false) { fprintf(stderr, "[Korali] Error: Incorrect or no Likelihood Type selected: %s.\n", likelihoodString.c_str()); exit(-1); }

 auto ref = consume(js, { "Bayesian", "Likelihood", "Reference Data" }, KORALI_ARRAY);
 _referenceDataSize = ref.size();
 _referenceData = (double*) calloc (_referenceDataSize, sizeof(double));
 for (size_t i = 0; i < _referenceDataSize; i++) _referenceData[i] = ref[i];

 for (size_t i = 0; i < _k->N; i++ )
 {
  auto typeString = consume(js["Variables"][i], { "Bayesian", "Type" }, KORALI_STRING, "Computational");
  if (typeString == "Computational") _computationalVariableIndices.push_back(i);
  if (typeString == "Statistical")   _statisticalVariableIndices.push_back(i);
 }
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::Bayesian::initialize()
{
 _isBayesian = true;

 if (_likelihood == ReferenceLikelihood)
 {
  if (_referenceDataSize == 0)
  {
   fprintf(stderr, "[Korali] Error: No Reference Data set provided for the Bayesian Model.\n");
   exit(-1);
  }

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

  if (_statisticalVariableIndices.size() != 1)
  {
   fprintf(stderr, "[Korali] Error: The Bayesian model requires 1 statistical parameter.\n");
   exit(-1);
  }
 }

 if (_likelihood == DirectLikelihood)
 {
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
}

void Korali::Problem::Bayesian::packVariables(double* sample, Korali::ModelData& data)
{
 for (size_t i = 0; i < _computationalVariableIndices.size(); i++) data._computationalVariables.push_back(sample[_computationalVariableIndices[i]]);
 for (size_t i = 0; i < _statisticalVariableIndices.size();   i++) data._statisticalVariables.push_back(sample[_statisticalVariableIndices[i]]);
}

double Korali::Problem::Bayesian::evaluateFitness(Korali::ModelData& data)
{

 double fitness = 0.0;

 if (_likelihood == ReferenceLikelihood)
 {
  if (data._results.size() != _referenceDataSize)
  {
   fprintf(stderr, "[Korali] Error: Reference Likelihood requires a %lu-sized result array.\n", _referenceDataSize);
   fprintf(stderr, "[Korali]        Provided: %lu.\n", data._results.size());
   exit(-1);
  }

  double sigma = data._statisticalVariables[0];
  double sigma2 = sigma*sigma;
  double ssn = 0.0;

  for(size_t i = 0; i < _referenceDataSize; i++)
  {
   double diff = _referenceData[i] - data._results[i];
   ssn += diff*diff;
  }

  fitness = -0.5*( _referenceDataSize*log(2*M_PI) + ssn/sigma2) - _referenceDataSize*log(sigma);
 }

 if (_likelihood == DirectLikelihood)
 {
  if (data._results.size() != 1)
  {
   fprintf(stderr, "[Korali] Error: Direct Likelihood requires exactly a 1-element result array.\n");
   fprintf(stderr, "[Korali]        Provided: %lu.\n", data._results.size());
   exit(-1);
  }

  fitness =  data._results[0];
 }

 return fitness;
}

double Korali::Problem::Bayesian::evaluateLogPrior(double* sample)
{
 double logPrior = 0.0;
 for (size_t i = 0; i < _k->N; i++) logPrior += _k->_variables[i]->getLogDensity(sample[i]);
 return logPrior;
}
