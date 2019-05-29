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

 js["Type"] = "Bayesian";

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

double Korali::Problem::Bayesian::evaluateFitness(double* sample, bool isLeader
#ifdef _KORALI_USE_MPI
, MPI_Comm comm
#endif
)
{

 if (_statisticalParameterCount != 1)
 {
  fprintf(stderr, "[Korali] Error: The Bayesian model requires 1 statistical parameter.\n");
  exit(-1);
 }

 if (isSampleOutsideBounds(sample)) return -DBL_MAX;

 double sigma = sample[_computationalParameterCount];
 double fitnessData[_referenceDataSize];

 Korali::ModelData data;

 #ifdef _KORALI_USE_MPI
 data._comm = comm;
 #endif

 for (size_t i = 0; i < N; i++) data._parameters.push_back(sample[i]);

 _k->_model(data);

 if (isLeader)
 {
  if (data._results.size() != _referenceDataSize)
  {
   fprintf(stderr, "[Korali] Error: This Bayesian Model requires a %lu-sized result array.\n", _referenceDataSize);
   fprintf(stderr, "[Korali]        Provided: %lu.\n", data._results.size());
   exit(-1);
  }

  //TODO: can we avoid this copy? (DW)
  for (size_t i = 0; i < _referenceDataSize; i++) fitnessData[i] = data._results[i];

  return Korali::Variable::Gaussian::logLikelihood(sigma, _referenceDataSize, _referenceData, fitnessData);
 }

 return 0;
}

double Korali::Problem::Bayesian::evaluateLogPrior(double* sample)
{
 double logPrior = 0.0;
 for (size_t i = 0; i < N; i++) logPrior += _parameters[i]->getLogDensity(sample[i]);
 return logPrior;
}
