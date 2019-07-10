#include "korali.h"

void Korali::Problem::Bayesian::getConfiguration()
{
 _k->_js["Problem"] = "Bayesian";

 if (_likelihood == DirectLikelihood)    _k->_js["Bayesian"]["Likelihood"]["Type"] = "Direct";
 if (_likelihood == ReferenceLikelihood) _k->_js["Bayesian"]["Likelihood"]["Type"] = "Reference";

 for (size_t i = 0; i < _referenceDataSize; i++) _k->_js["Bayesian", "Likelihood", "Reference Data"][i] = _referenceData[i];

 for (size_t i = 0; i < _computationalVariableIndices.size(); i++)
 {
  size_t idx = _computationalVariableIndices[i];
  _k->_js["Variables"][idx]["Bayesian"]["Type"] = "Computational";
 }

 for (size_t i = 0; i < _statisticalVariableIndices.size(); i++)
  {
   size_t idx = _statisticalVariableIndices[i];
   _k->_js["Variables"][idx]["Bayesian"]["Type"] = "Statistical";
  }

 for (size_t i = 0; i < _k->N; i++)  _k->_variables[i]->getDistribution(_k->_js["Variables"][i]["Bayesian"]["Prior Distribution"]);
}

void Korali::Problem::Bayesian::setConfiguration()
{
  bool foundLikelihoodType = false;
  std::string likelihoodString = consume(_k->_js, { "Bayesian", "Likelihood", "Type" }, KORALI_STRING, "Undefined");
  if (likelihoodString == "Direct")    { _likelihood = DirectLikelihood;    foundLikelihoodType = true; }
  if (likelihoodString == "Reference") { _likelihood = ReferenceLikelihood; foundLikelihoodType = true; }
  if (foundLikelihoodType == false) { fprintf(stderr, "[Korali] Error: Incorrect or no Likelihood Type selected: %s.\n", likelihoodString.c_str()); exit(-1); }

  bool foundLikelihoodModel = false;
  std::string likelihoodModelString = consume(_k->_js, { "Bayesian", "Likelihood", "Model" }, KORALI_STRING, "Undefined");
  if (likelihoodModelString == "Additive Gaussian")          { _likelihoodModel = AdditiveGaussian;       foundLikelihoodModel = true; }
  if (likelihoodModelString == "Multiplicative Gaussian")    { _likelihoodModel = MultiplicativeGaussian; foundLikelihoodModel = true; }
  if (foundLikelihoodModel  == false) { fprintf(stderr, "[Korali] Error: Incorrect or no Likelihood Model selected: %s.\n", likelihoodString.c_str()); exit(-1); }

  auto ref = consume(_k->_js, { "Bayesian", "Likelihood", "Reference Data" }, KORALI_ARRAY);
  _referenceDataSize = ref.size();
  _referenceData.resize(_referenceDataSize);
  for (size_t i = 0; i < _referenceDataSize; i++) _referenceData[i] = ref[i];

  if (isArray(_k->_js, { "Variables" } ))
  for (size_t i = 0; i < _k->N; i++)
  {
  auto typeString = consume(_k->_js["Variables"][i], { "Bayesian", "Type" }, KORALI_STRING, "Computational");
  if (typeString == "Computational") _computationalVariableIndices.push_back(i);
  if (typeString == "Statistical")   _statisticalVariableIndices.push_back(i);

  bool foundPriorDistribution = isDefined(_k->_js["Variables"][i], {"Bayesian", "Prior Distribution" });
  if (foundPriorDistribution == false) { fprintf(stderr, "[Korali] Error: No Prior Distribution information provided for variable: %s.\n", _k->_variables[i]->_name.c_str()); exit(-1); }

  _k->_js["Variables"][i]["Bayesian"]["Prior Distribution"]["Seed"] = _k->_seed++;
  _k->_variables[i]->setDistribution(_k->_js["Variables"][i]["Bayesian"]["Prior Distribution"]);
  }
}

void Korali::Problem::Bayesian::initialize()
{
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

	if (_statisticalVariableIndices.size() != 0)
	{
	 fprintf(stderr, "[Korali] Error: Direct Bayesian Evaluation type requires no statistical parameters.\n");
	 exit(-1);
	}
 }
}

void Korali::Problem::Bayesian::finalize()
{

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

   if (_likelihoodModel == AdditiveGaussian)       fitness = likelihoodGaussianAdditive(data);
   if (_likelihoodModel == MultiplicativeGaussian) fitness = likelihoodGaussianMultiplicative(data);
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

double Korali::Problem::Bayesian::likelihoodGaussianAdditive(Korali::ModelData& data)
{
  double sigma   = data._statisticalVariables[0];
  double sigma2  = sigma*sigma;
  double ssn     = 0.0;
  double fitness = 0.0;

  for(size_t i = 0; i < _referenceDataSize; i++)
  {
   double diff = _referenceData[i] - data._results[i];
   ssn += diff*diff;
  }

  fitness = -0.5*( _referenceDataSize*log(2*M_PI) + ssn/sigma2) - _referenceDataSize*log(sigma);
  return fitness;
}

double Korali::Problem::Bayesian::likelihoodGaussianMultiplicative(Korali::ModelData& data)
{
  double sigma    = data._statisticalVariables[0];
  double ssn      = 0.0;
  double fitness  = 0.0;
  double logSigma = 0.0;
  for(size_t i = 0; i < _referenceDataSize; i++)
  {
   double diff   = _referenceData[i] - data._results[i];
   double denom  = sigma*data._results[i];
   ssn += diff*diff / (denom*denom);
   logSigma += log(denom);
  }

  fitness = -0.5*( _referenceDataSize*log(2*M_PI) + ssn) - _referenceDataSize*logSigma;
  return fitness;
}
