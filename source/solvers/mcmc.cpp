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

void Korali::Solver::MCMC::runGeneration()
{
 rejectionCount = 0;
 while( rejectionCount < rejectionLevels )
 {
  generateCandidate(rejectionCount);
  evaluateSample();
  _k->_conduit->checkProgress();
  acceptReject(rejectionCount);
  rejectionCount++;
 }
 chainLength++;
 if (chainLength > burnIn ) updateDatabase(chainLeaderParameters, chainLeaderLogLikelihood);
 updateState();
}

void Korali::Solver::MCMC::initialize()
{
 // Allocating MCMC memory
 covarianceMatrix.resize(_k->N*_k->N);
 chainLeaderParameters.resize(_k->N);
 chainCandidatesParameters.resize(_k->N*rejectionLevels);
 logTransformedSamples.resize(_k->N*rejectionLevels);
 chainCandidatesLogPriors.resize(rejectionLevels);
 chainCandidatesLogLikelihoods.resize(rejectionLevels);
 rejectionAlphas.resize(rejectionLevels);
 sampleParametersDatabase.resize(_k->N*maxChainLength);
 sampleFitnessDatabase.resize(maxChainLength);
 chainMean.resize(_k->N);
 chainCovariancePlaceholder.resize(_k->N*_k->N);
 chainCovariance.resize(_k->N*_k->N);

 for(size_t i = 0; i < _k->N; i++) chainLeaderParameters[i] = _variableSettings[i].initialMean;
 for(size_t i = 0; i < _k->N; i++) covarianceMatrix[i*_k->N+i] = _variableSettings[i].standardDeviation;

 // Init Generation
 acceptanceCount = 0;
 proposedSampleCount = 0;
 chainLength = 0;
 rejectionCount = 0;
 databaseEntryCount = 0;
 chainLeaderLogLikelihood = -std::numeric_limits<double>::max();
 acceptanceRate  = 1.0;
}

void Korali::Solver::MCMC::processSample(size_t sampleIdx, double fitness)
{
 chainCandidatesLogLikelihoods[sampleIdx] = fitness + chainCandidatesLogPriors[sampleIdx];
}

void Korali::Solver::MCMC::acceptReject(size_t trial)
{

 double denom;
 double rejectionAlphas = recursiveAlpha(denom, chainLeaderLogLikelihood, &chainCandidatesLogLikelihoods[0], trial);
 if ( rejectionAlphas == 1.0 || rejectionAlphas > _uniformGenerator->getRandomNumber() ) {
   acceptanceCount++;
   chainLeaderLogLikelihood = chainCandidatesLogLikelihoods[trial];
   for (size_t d = 0; d < _k->N; d++) chainLeaderParameters[d] = chainCandidatesParameters[trial*_k->N+d];
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
 for (size_t d = 0; d < _k->N; d++) sampleParametersDatabase[databaseEntryCount*_k->N + d] = point[d];
 sampleFitnessDatabase[databaseEntryCount] = loglik;
 databaseEntryCount++;
}


void Korali::Solver::MCMC::generateCandidate(size_t sampleIdx)
{  
 size_t initialgens = proposedSampleCount;
 sampleCandidate(sampleIdx); proposedSampleCount++;
 setCandidatePriorAndCheck(sampleIdx);
 
 /*do { /TODO: fix check (DW)
   sampleCandidate(level); proposedSampleCount++;
   setCandidatePriorAndCheck(level);
   if ( (proposedSampleCount - initialgens) > _maxresamplings)
   {       
     if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: exiting resampling loop, max resamplings (%zu) reached.\n", _maxresamplings);
     exit(-1);
   }
  } while (setCandidatePriorAndCheck(level)); */
}

void Korali::Solver::MCMC::evaluateSample()
{
  for(size_t d = 0; d < _k->N; ++d)
    //if(_k->_variables[d]->_isLogSpace == true)
    //    logTransformedSamples[rejectionCount*_k->N+d] = std::exp(chainCandidatesParameters[rejectionCount*_k->N+d]);
    //else 
        logTransformedSamples[rejectionCount*_k->N+d] = chainCandidatesParameters[rejectionCount*_k->N+d];

  _k->_conduit->evaluateSample(&logTransformedSamples[0], rejectionCount);
}

void Korali::Solver::MCMC::sampleCandidate(size_t sampleIdx)
{  
 for (size_t d = 0; d < _k->N; ++d) chainCandidatesParameters[sampleIdx*_k->N+d] = 0.0;

 if ( (useAdaptiveSampling == false) || (databaseEntryCount <= nonAdaptionPeriod + burnIn))
     for (size_t d = 0; d < _k->N; ++d) for (size_t e = 0; e < _k->N; ++e) chainCandidatesParameters[sampleIdx*_k->N+d] += covarianceMatrix[d*_k->N+e] * _gaussianGenerator->getRandomNumber();
 else
     for (size_t d = 0; d < _k->N; ++d) for (size_t e = 0; e < _k->N; ++e) chainCandidatesParameters[sampleIdx*_k->N+d] += chainCovariance[d*_k->N+e] * _gaussianGenerator->getRandomNumber();

 for (size_t d = 0; d < _k->N; ++d) chainCandidatesParameters[sampleIdx*_k->N+d] += chainLeaderParameters[d];
}

bool Korali::Solver::MCMC::setCandidatePriorAndCheck(size_t sampleIdx)
{
 chainCandidatesLogPriors[sampleIdx] = _k->_problem->evaluateLogPrior(&chainCandidatesParameters[_k->N*sampleIdx]);
 if (chainCandidatesLogPriors[sampleIdx] > -INFINITY) return true;
 return false;
}

void Korali::Solver::MCMC::updateState()
{

 acceptanceRate = ( (double)acceptanceCount/ (double)chainLength );

 if(databaseEntryCount < 1) return;
 
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
   chainCovariancePlaceholder[d*_k->N+e] = databaseEntryCount*chainMean[d]*chainMean[e] + chainLeaderParameters[d]*chainLeaderParameters[e];
   chainCovariancePlaceholder[e*_k->N+d] = databaseEntryCount*chainMean[d]*chainMean[e] + chainLeaderParameters[d]*chainLeaderParameters[e];
 }
 for (size_t d = 0; d < _k->N; d++) chainCovariancePlaceholder[d*_k->N+d] = databaseEntryCount*chainMean[d]*chainMean[d] + chainLeaderParameters[d]*chainLeaderParameters[d] + chainCovarianceIncrement;

 // Chain Mean
 for (size_t d = 0; d < _k->N; d++) chainMean[d] = ((chainMean[d] * (databaseEntryCount-1) + chainLeaderParameters[d])) / ((double) databaseEntryCount);
 
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
    chainCovariancePlaceholder[d*_k->N+e] -= (databaseEntryCount+1)*chainMean[d]*chainMean[e];
    chainCovariancePlaceholder[e*_k->N+d] -= (databaseEntryCount+1)*chainMean[d]*chainMean[e];

 }
 for (size_t d = 0; d < _k->N; d++) chainCovariancePlaceholder[d*_k->N+d] -= (databaseEntryCount+1)*chainMean[d]*chainMean[d];

 // Chain Covariance (TODO: careful check N (databasEntires) (DW)
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
   chainCovariance[d*_k->N+e] = (databaseEntryCount-1.0)/( (double) databaseEntryCount) * chainCovariance[d*_k->N+e] + (chainCovarianceScaling/( (double) databaseEntryCount))*chainCovariancePlaceholder[d*_k->N+e];
   chainCovariance[e*_k->N+d] = (databaseEntryCount-1.0)/( (double) databaseEntryCount) * chainCovariance[d*_k->N+e] + (chainCovarianceScaling/( (double) databaseEntryCount))*chainCovariancePlaceholder[d*_k->N+e];
 }
 for (size_t d = 0; d < _k->N; d++) chainCovariance[d*_k->N+d] = (databaseEntryCount-1.0)/( (double) databaseEntryCount) * chainCovariance[d*_k->N+d] + (chainCovarianceScaling/( (double) databaseEntryCount))*chainCovariancePlaceholder[d*_k->N+d];
}

bool Korali::Solver::MCMC::checkTermination()
{
 bool isFinished = false;
 if ( databaseEntryCount == maxChainLength)
 {
	 if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Chainlength (%zu) reached.\n",  chainLength);
  isFinished = true;
 }

 return isFinished;
}
 
void Korali::Solver::MCMC::printGeneration()
{
 if (chainLength % terminalOutputFrequency != 0) return;
 
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
  printf("--------------------------------------------------------------------\n");
  printf("[Korali] Database Entries %ld\n", databaseEntryCount);
 }

 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Accepted Samples: %zu\n", acceptanceCount);
 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Acceptance Rate Proposals: %.2f%%\n", 100*acceptanceRate);

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (Current Sample, Current Candidate):\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), chainLeaderParameters[d], chainCandidatesParameters[d]);
  printf("[Korali] Current Chain Mean:\n");
  for (size_t d = 0; d < _k->N; d++) printf(" %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), chainMean[d]);
  printf("[Korali] Current Chain Covariance:\n");
  for (size_t d = 0; d < _k->N; d++)
  {
   for (size_t e = 0; e <= d; e++) printf("   %+6.3e  ", chainCovariance[d*_k->N+e]);
   printf("\n");
  }
 }
}

void Korali::Solver::MCMC::finalize()
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] MCMC Finished\n");
    printf("[Korali] Number of Generated Samples: %zu\n", proposedSampleCount);
    printf("[Korali] Acceptance Rate: %.2f%%\n", 100*acceptanceRate);
    if (databaseEntryCount == chainLength) printf("[Korali] Max Samples Reached.\n");
    printf("--------------------------------------------------------------------\n");
 }
}

