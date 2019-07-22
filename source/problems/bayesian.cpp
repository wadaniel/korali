#include "korali.hpp"

void Korali::Problem::Bayesian::getConfiguration()
{
 _k->_js["Problem"]["Type"] = "Bayesian Inference";

 if (_likelihoodModel == CustomLikelihood)  _k->_js["Problem"]["Likelihood"]["Model"] = "Custom";
 if (_likelihoodModel == AdditiveGaussianLikelihood)  _k->_js["Problem"]["Likelihood"]["Model"] = "Additive Gaussian";
 if (_likelihoodModel == MultiplicativeGaussianLikelihood)  _k->_js["Problem"]["Likelihood"]["Model"] = "Multiplicative Gaussian";
 if (_likelihoodModel == MultiplicativeGaussianDataLikelihood)  _k->_js["Problem"]["Likelihood"]["Model"] = "Multiplicative Gaussian Data";

 for (size_t i = 0; i < _referenceDataSize; i++) _k->_js["Problem", "Likelihood", "Reference Data"][i] = _referenceData[i];

 for (size_t i = 0; i < _computationalVariableIndices.size(); i++)
 {
  size_t idx = _computationalVariableIndices[i];
  _k->_js["Variables"][idx]["Type"] = "Computational";
 }

 for (size_t i = 0; i < _statisticalVariableIndices.size(); i++)
  {
   size_t idx = _statisticalVariableIndices[i];
   _k->_js["Variables"][idx]["Type"] = "Statistical";
  }
 }

void Korali::Problem::Bayesian::setConfiguration()
{
  bool foundLikelihoodModelType = false;
  std::string likelihoodModelString = consume(_k->_js, { "Problem", "Likelihood", "Model" }, KORALI_STRING, "Undefined");
  if (likelihoodModelString == "Custom")    { _likelihoodModel = CustomLikelihood;    foundLikelihoodModelType = true; }
  if (likelihoodModelString == "Additive Gaussian") { _likelihoodModel = AdditiveGaussianLikelihood; foundLikelihoodModelType = true; }
  if (likelihoodModelString == "Multiplicative Gaussian") { _likelihoodModel = MultiplicativeGaussianLikelihood; foundLikelihoodModelType = true; }
  if (likelihoodModelString == "Multiplicative Gaussian Data") { _likelihoodModel = MultiplicativeGaussianDataLikelihood; foundLikelihoodModelType = true; }
  if (foundLikelihoodModelType == false) { koraliError("Incorrect or no Likelihood Type selected: %s.\n", likelihoodModelString.c_str()); }

  auto ref = consume(_k->_js, { "Problem", "Likelihood", "Reference Data" }, KORALI_ARRAY);
  _referenceDataSize = ref.size();
  _referenceData.resize(_referenceDataSize);
  for (size_t i = 0; i < _referenceDataSize; i++) _referenceData[i] = ref[i];

  for (size_t i = 0; i < _k->N; i++)
  {
    auto typeString = consume(_k->_js["Variables"][i], { "Type" }, KORALI_STRING, "Computational");
    if (typeString == "Computational") _computationalVariableIndices.push_back(i);
    if (typeString == "Statistical")   _statisticalVariableIndices.push_back(i);
  }
}

void Korali::Problem::Bayesian::initialize()
{
 for(size_t i = 0; i < _k->N; i++) if(_k->_variables[i]->_distributionType == KoraliDefaultDistribution)
	koraliError("Bayesian inference requires prior distribution for all variables. (Missing for %s).\n", _k->_variables[i]->_name.c_str());

 if (_k->_constraints.size() > 0) koraliError("Bayesian inference problems do not allow constraint definitions.\n");

 if (_likelihoodModel != CustomLikelihood)
 {
   if (_referenceDataSize == 0) koraliError("The selected likelihood model requires defining reference data.\n");
   if (_k->_modelDefined == false) koraliError("The selected likelihood model requires defining a computational model.\n");
   if (_statisticalVariableIndices.size() != 1) koraliError("The selected likelihood model requires 1 statistical parameter.\n");
 }

 if (_likelihoodModel == CustomLikelihood)
 {
   if (_referenceDataSize != 0) koraliError("Custom likelihood models do not accept reference data.\n");
   if (_k->_likelihoodDefined == false) koraliError("No likelihood model was specified.\n");
   if (_statisticalVariableIndices.size() != 0) koraliError("Custom likelihood models do not accept statistical variables.\n");
 }
}

void Korali::Problem::Bayesian::finalize()
{

}

void Korali::Problem::Bayesian::packVariables(double* sample, Korali::Model& data)
{
 for (size_t i = 0; i < _computationalVariableIndices.size(); i++) data._computationalVariables.push_back(sample[_computationalVariableIndices[i]]);
 for (size_t i = 0; i < _statisticalVariableIndices.size();   i++) data._statisticalVariables.push_back(sample[_statisticalVariableIndices[i]]);
}

double Korali::Problem::Bayesian::evaluateFitness(Korali::Model& data)
{
 double fitness = 0.0;

 if (_likelihoodModel != CustomLikelihood)
 {
   if (data._results.size() != _referenceDataSize)
     koraliError("The selected likelihood model requires a %lu-sized result array. Provided: %lu.\n", _referenceDataSize, data._results.size());

   if (_likelihoodModel == AdditiveGaussianLikelihood)           fitness = likelihoodGaussianAdditive(data);
   if (_likelihoodModel == MultiplicativeGaussianLikelihood)     fitness = likelihoodGaussianMultiplicative(data);
   if (_likelihoodModel == MultiplicativeGaussianDataLikelihood) fitness = likelihoodGaussianMultiplicativeData(data);
 }

 if (_likelihoodModel == CustomLikelihood)
 {
  if (data._results.size() != 1)  koraliError("Custom likelihoods require exactly a 1-element result array. Provided: %lu.\n", data._results.size() );
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

double Korali::Problem::Bayesian::likelihoodGaussianAdditive(Korali::Model& data)
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

double Korali::Problem::Bayesian::likelihoodGaussianMultiplicative(Korali::Model& data)
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

double Korali::Problem::Bayesian::likelihoodGaussianMultiplicativeData(Korali::Model& data)
{
  double sigma    = data._statisticalVariables[0];
  double ssn      = 0.0;
  double fitness  = 0.0;
  double logSigma = 0.0;
  for(size_t i = 0; i < _referenceDataSize; i++)
  {
   double diff   = _referenceData[i] - data._results[i];
   double denom  = sigma*_referenceData[i];
   ssn += diff*diff / (denom*denom);
   logSigma += log(denom);
  }

  fitness = -0.5*( _referenceDataSize*log(2*M_PI) + ssn) - _referenceDataSize*logSigma;
  return fitness;
}
