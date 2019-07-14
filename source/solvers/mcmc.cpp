#include "korali.h"
#include <numeric>
#include <limits>
#include <chrono>

#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_multimin.h>

Korali::Solver::MCMC::MCMC()
{
 // Initializing Gaussian Generator
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"] = "Gaussian";
 jsGaussian["Mean"] = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = new Variable();
 _gaussianGenerator->setDistribution(jsGaussian);

 auto jsUniform = nlohmann::json();
 jsUniform["Type"] = "Uniform";
 jsUniform["Minimum"] = 0.0;
 jsUniform["Maximum"] = 1.0;
 jsUniform["Seed"] = _k->_seed++;
 _uniformGenerator = new Variable();
 _uniformGenerator->setDistribution(jsUniform);
}

Korali::Solver::MCMC::~MCMC()
{
 delete _gaussianGenerator;
 delete _uniformGenerator;
}

void Korali::Solver::MCMC::runGeneration()
{
 _rejectionCount = 0;
 while( _rejectionCount < _rejectionLevels )
 {
  generateCandidate(_rejectionCount);
  evaluateSample();
  _k->_conduit->checkProgress();
  acceptReject(_rejectionCount);
  _rejectionCount++;
 }
 _chainLength++;
 if (_chainLength > _burnIn ) updateDatabase(_chainLeaderParameters, _chainLeaderLogLikelihood);
 updateState();
}

void Korali::Solver::MCMC::initialize()
{
 // Allocating MCMC memory
 _covarianceMatrix.resize(_k->N*_k->N);
 _chainLeaderParameters.resize(_k->N);
 _chainCandidatesParameters.resize(_k->N*_rejectionLevels);
 _chainCandidatesLogPriors.resize(_rejectionLevels);
 _chainCandidatesLogLikelihoods.resize(_rejectionLevels);
 _rejectionAlphas.resize(_rejectionLevels);
 _sampleParametersDatabase.resize(_k->N*_maxChainLength);
 _sampleFitnessDatabase.resize(_maxChainLength);
 _chainMean.resize(_k->N);
 _chainCovariancePlaceholder.resize(_k->N*_k->N);
 _chainCovariance.resize(_k->N*_k->N);

 for(size_t i = 0; i < _k->N; i++) _chainLeaderParameters[i] = _variableSettings[i].initialMean;
 for(size_t i = 0; i < _k->N; i++) _covarianceMatrix[i*_k->N+i] = _variableSettings[i].standardDeviation;

 // Init Generation
 _acceptanceCount = 0;
 _proposedSampleCount = 0;
 _chainLength = 0;
 _rejectionCount = 0;
 _databaseEntryCount = 0;
 _chainLeaderLogLikelihood = -std::numeric_limits<double>::max();
 _acceptanceRate  = 1.0;
 _chainCovarianceScaling = 2.4*2.4/_k->N; //[Gelman et. al. 1995]
}

void Korali::Solver::MCMC::processSample(size_t sampleIdx, double fitness)
{
 _chainCandidatesLogLikelihoods[sampleIdx] = fitness + _chainCandidatesLogPriors[sampleIdx];
}

void Korali::Solver::MCMC::acceptReject(size_t trial)
{

 double denom;
 double _rejectionAlphas = recursiveAlpha(denom, _chainLeaderLogLikelihood, &_chainCandidatesLogLikelihoods[0], trial);
 if ( _rejectionAlphas == 1.0 || _rejectionAlphas > _uniformGenerator->getRandomNumber() ) {
   _acceptanceCount++;
   _chainLeaderLogLikelihood = _chainCandidatesLogLikelihoods[trial];
   for (size_t d = 0; d < _k->N; d++) _chainLeaderParameters[d] = _chainCandidatesParameters[trial*_k->N+d];
 }
}

double Korali::Solver::MCMC::recursiveAlpha(double& denom, const double llk0, const double* logliks, size_t N) const
{
 // recursive formula from Trias[2009]

 if(N==0)
 {
  denom = exp(llk0);
  return std::min(1.0, exp(logliks[0] - llk0));
 }
 else
 {
  // revert sample array
  double* revLlks = new double[N];
  for(size_t i = 0; i < N; ++i) revLlks[i] = logliks[N-1-i];
  
  // update numerator (w. recursive calls)
  double numerator = std::exp(logliks[N]);
  for(size_t i = 0; i < N; ++i)
  {
   double denom2;
   double recalpha2 = recursiveAlpha(denom2, logliks[N], revLlks, i);
   numerator *=  ( 1.0 - recalpha2 );
  }
  delete [] revLlks;

  if (numerator == 0.0) return 0.0;

  // update denomiator
  double denom1;
  double recalpha1 = recursiveAlpha(denom1, llk0, logliks, N-1);
  denom = denom1 * (1.0 - recalpha1);

  return std::min(1.0, numerator/denom);
 }
}


void Korali::Solver::MCMC::updateDatabase(std::vector<double>& point, double loglik)
{
 for (size_t d = 0; d < _k->N; d++) _sampleParametersDatabase[_databaseEntryCount*_k->N + d] = point[d];
 _sampleFitnessDatabase[_databaseEntryCount] = loglik;
 _databaseEntryCount++;
}


void Korali::Solver::MCMC::generateCandidate(size_t sampleIdx)
{  
 size_t initialgens = _proposedSampleCount;
 sampleCandidate(sampleIdx); _proposedSampleCount++;
 setCandidatePriorAndCheck(sampleIdx);
 
 /*do { /TODO: fix check (DW)
   sampleCandidate(level); _proposedSampleCount++;
   setCandidatePriorAndCheck(level);
   if ( (_proposedSampleCount - initialgens) > _maxresamplings)
   {       
     if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: exiting resampling loop, max resamplings (%zu) reached.\n", _maxresamplings);
     exit(-1);
   }
  } while (setCandidatePriorAndCheck(level)); */
}

void Korali::Solver::MCMC::evaluateSample()
{
 std::vector<double> _logTransformedSample(_k->N);

 for(size_t d = 0; d<_k->N; ++d)
   if (_k->_variables[d]->_isLogSpace == true)
       _logTransformedSample[d] = std::exp(_chainCandidatesParameters[_rejectionCount*_k->N+d]);
   else
       _logTransformedSample[d] = _chainCandidatesParameters[_rejectionCount*_k->N+d];

  _k->_conduit->evaluateSample(_logTransformedSample.data(), _rejectionCount);
}

void Korali::Solver::MCMC::sampleCandidate(size_t sampleIdx)
{  
 for (size_t d = 0; d < _k->N; ++d) _chainCandidatesParameters[sampleIdx*_k->N+d] = 0.0;

 if ( (_useAdaptiveSampling == false) || (_databaseEntryCount <= _nonAdaptionPeriod + _burnIn))
     for (size_t d = 0; d < _k->N; ++d) for (size_t e = 0; e < _k->N; ++e) _chainCandidatesParameters[sampleIdx*_k->N+d] += _covarianceMatrix[d*_k->N+e] * _gaussianGenerator->getRandomNumber();
 else
     for (size_t d = 0; d < _k->N; ++d) for (size_t e = 0; e < _k->N; ++e) _chainCandidatesParameters[sampleIdx*_k->N+d] += _chainCovariance[d*_k->N+e] * _gaussianGenerator->getRandomNumber();

 for (size_t d = 0; d < _k->N; ++d) _chainLeaderParameters[d] += _chainCandidatesParameters[sampleIdx*_k->N+d];
}

bool Korali::Solver::MCMC::setCandidatePriorAndCheck(size_t sampleIdx)
{
 _chainCandidatesLogPriors[sampleIdx] = _k->_problem->evaluateLogPrior(&_chainCandidatesParameters[_k->N*sampleIdx]);
 if (_chainCandidatesLogPriors[sampleIdx] > -INFINITY) return true;
 return false;
}

void Korali::Solver::MCMC::updateState()
{

 _acceptanceRate = ( (double)_acceptanceCount/ (double)_chainLength );

 if(_databaseEntryCount < 1) return;
 
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
   _chainCovariancePlaceholder[d*_k->N+e] = (_databaseEntryCount-1)*_chainMean[d]*_chainMean[e] + _chainLeaderParameters[d]*_chainLeaderParameters[e];
   _chainCovariancePlaceholder[e*_k->N+d] = (_databaseEntryCount-1)*_chainMean[d]*_chainMean[e] + _chainLeaderParameters[d]*_chainLeaderParameters[e];
 }
 for (size_t d = 0; d < _k->N; d++) _chainCovariancePlaceholder[d*_k->N+d] += _chainCovarianceIncrement*_chainCovarianceScaling;

 // Chain Mean
 for (size_t d = 0; d < _k->N; d++) _chainMean[d] = ((_chainMean[d] * (_databaseEntryCount-1) + _chainLeaderParameters[d])) / ((double) _databaseEntryCount);
 
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
    _chainCovariancePlaceholder[d*_k->N+e] -= _databaseEntryCount*_chainMean[d]*_chainMean[e];
    _chainCovariancePlaceholder[e*_k->N+d] -= _databaseEntryCount*_chainMean[d]*_chainMean[e];
 }

 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
   _chainCovariance[d*_k->N+e] = (_databaseEntryCount-2.0)/(_databaseEntryCount-1.0) * _chainCovariance[d*_k->N+e] + (_chainCovarianceScaling/( (double) _databaseEntryCount))*_chainCovariancePlaceholder[d*_k->N+e];
   _chainCovariance[e*_k->N+d] = (_databaseEntryCount-2.0)/(_databaseEntryCount-1.0) * _chainCovariance[d*_k->N+e] + (_chainCovarianceScaling/( (double) _databaseEntryCount))*_chainCovariancePlaceholder[d*_k->N+e];
 }
}

bool Korali::Solver::MCMC::checkTermination()
{
 bool isFinished = false;
 if ( _databaseEntryCount == _maxChainLength)
 {
   koraliLog(KORALI_MINIMAL, "Chainlength (%zu) reached.\n",  _chainLength);
   isFinished = true;
 }

 return isFinished;
}
 
void Korali::Solver::MCMC::printGeneration()
{
 koraliLog(KORALI_MINIMAL, "--------------------------------------------------------------------\n");
 koraliLog(KORALI_MINIMAL, "Database Entries %ld\n", _databaseEntryCount);

 koraliLog(KORALI_NORMAL, "Accepted Samples: %zu\n", _acceptanceCount);
 koraliLog(KORALI_NORMAL, "Acceptance Rate Proposals: %.2f%%\n", 100*_acceptanceRate);

 koraliLog(KORALI_DETAILED, "Variable = (Current Sample, Current Candidate):\n");
 for (size_t d = 0; d < _k->N; d++)  koraliLog(KORALI_DETAILED, "         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _chainLeaderParameters[d], _chainCandidatesParameters[d]);
 koraliLog(KORALI_DETAILED, "Current Chain Mean:\n");
 for (size_t d = 0; d < _k->N; d++) koraliLog(KORALI_DETAILED, " %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _chainMean[d]);
 koraliLog(KORALI_DETAILED, "Current Chain Covariance:\n");
 for (size_t d = 0; d < _k->N; d++)
 {
	 for (size_t e = 0; e <= d; e++) koraliLog(KORALI_DETAILED, "   %+6.3e  ", _chainCovariance[d*_k->N+e]);
	 koraliLog(KORALI_DETAILED, "\n");
 }
}

void Korali::Solver::MCMC::finalize()
{
	koraliLog(KORALI_MINIMAL, "MCMC Finished\n");
	koraliLog(KORALI_MINIMAL, "Number of Generated Samples: %zu\n", _proposedSampleCount);
	koraliLog(KORALI_MINIMAL, "Acceptance Rate: %.2f%%\n", 100*_acceptanceRate);
	if (_databaseEntryCount == _chainLength) koraliLog(KORALI_MINIMAL, "Max Samples Reached.\n");
	koraliLog(KORALI_MINIMAL, "--------------------------------------------------------------------\n");
}

