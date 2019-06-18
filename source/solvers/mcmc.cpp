#include "korali.h"
#include <numeric>
#include <limits>
#include <chrono>

#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_multimin.h>

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Solver::MCMC::MCMC(nlohmann::json& js, std::string name)
{
 _name = name;
 setConfiguration(js);

 // Allocating MCMC memory
 _covarianceMatrix = (double*) calloc (_k->N*_k->N, sizeof(double));
 z                 = (double*) calloc (_k->N, sizeof(double));
 clPoint           = (double*) calloc (_k->N, sizeof(double));
 ccPoint           = (double*) calloc (_k->N*_rejectionLevels, sizeof(double));
 ccLogPrior        = (double*) calloc (_rejectionLevels, sizeof(double));
 ccLogLikelihood   = (double*) calloc (_rejectionLevels, sizeof(double));
 alpha             = (double*) calloc (_rejectionLevels, sizeof(double));
 databasePoints    = (double*) calloc (_k->N*_s, sizeof(double));
 databaseFitness   = (double*) calloc (_s, sizeof(double));
 chainMean         = (double*) calloc (_k->N, sizeof(double));
 tmpC              = (double*) calloc (_k->N*_k->N , sizeof(double));
 chainCov          = (double*) calloc (_k->N*_k->N , sizeof(double));

 for(size_t i = 0; i < _k->N; i++) clPoint[i] = _initialMean[i];
 for(size_t i = 0; i < _k->N; i++) _covarianceMatrix[i*_k->N+i] = _initialStdDevs[i];

 // Initializing Gaussian Generator
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"] = "Gaussian";
 jsGaussian["Mean"] = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = new Variable();
 _gaussianGenerator->setDistribution(jsGaussian);

 countevals               = 0;
 naccept                  = 0;
 countgens                = 0;
 chainLength              = 0;
 databaseEntries          = 0;
 clLogLikelihood          = -std::numeric_limits<double>::max();
 acceptanceRateProposals  = 1.0;

 // If state is defined:
 if (isDefined(js, {"MCMC", "State"}))
 {
  setState(js);
  js["MCMC"].erase("State");
 }

}

Korali::Solver::MCMC::~MCMC()
{
 // TODO: Ensure proper memory deallocation
}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Solver::MCMC::getConfiguration()
{
 auto js = nlohmann::json();

 js["Solver"] = _name;

 js["MCMC"]["Population Size"]                = _s;
 js["MCMC"]["Burn In"]                        = _burnin;
 js["MCMC"]["Rejection Levels"]               = _rejectionLevels;
 js["MCMC"]["Adaptive MCMC"]                  = _adaptive;
 js["MCMC"]["Non Adaption Period"]            = _nonAdaptionPeriod;
 js["MCMC"]["Chain Covariance Learning Rate"] = _cr;
 js["MCMC"]["Chain Covariance Increment"]     = _eps;
 js["MCMC"]["Max Resamplings"]                = _maxresamplings;

 js["MCMC"]["Termination Criteria"]["Max Function Evaluations"]["Value"]  = _termCondMaxFunEvals;
 js["MCMC"]["Termination Criteria"]["Max Function Evaluations"]["Active"] = _isTermCondMaxFunEvals;
 js["MCMC"]["Termination Criteria"]["Max Generated Samples"]["Value"]     = _termCondMaxGenerations;
 js["MCMC"]["Termination Criteria"]["Max Generated Samples"]["Active"]    = _isTermCondMaxGenerations;

 // State Variables
 for (size_t d = 0; d < _k->N*_k->N; d++) js["MCMC"]["State"]["CovarianceMatrix"][d] = _covarianceMatrix[d];
 
 js["MCMC"]["State"]["Function Evaluations"]      = countevals;
 js["MCMC"]["State"]["Number Accepted Samples"]   = naccept;
 js["MCMC"]["State"]["Chain Length"]              = chainLength;
 js["MCMC"]["State"]["Database Entries"]          = databaseEntries;
 js["MCMC"]["State"]["Acceptance Rate Proposals"] = acceptanceRateProposals;
 for (size_t d = 0; d < _k->N; d++) js["MCMC"]["State"]["Leader"][d]                         = clPoint[d];
 //for (size_t d = 0; d < _k->N; d++) js["MCMC"]["State"]["Candidate"][d]                      = ccPoint[(_rejectionLevels-1)*_k->N+d];
 for (size_t i = 0; i < _k->N*databaseEntries; i++) js["MCMC"]["State"]["DatabasePoints"][i] = databasePoints[i];
 for (size_t i = 0; i < databaseEntries; i++) js["MCMC"]["State"]["DatabaseFitness"][i]      = databaseFitness[i];
 for (size_t d = 0; d < _k->N; d++) js["MCMC"]["State"]["Chain Mean"][d]                     = chainMean[d];
 for (size_t d = 0; d < _k->N * _k->N; d++) js["MCMC"]["State"]["Chain Covariance"][d]       = chainCov[d];

 js["MCMC"]["State"]["LeaderFitness"]    = clLogLikelihood;
 //js["MCMC"]["State"]["CandidateFitness"] = ccLogLikelihood;

 return js;
}

void Korali::Solver::MCMC::setConfiguration(nlohmann::json& js)
{
 _s                        = consume(js, { "MCMC", "Population Size" }, KORALI_NUMBER);
 _burnin                   = consume(js, { "MCMC", "Burn In" }, KORALI_NUMBER, std::to_string(0));
 _rejectionLevels          = consume(js, { "MCMC", "Rejection Levels" }, KORALI_NUMBER, std::to_string(1));

 if (_rejectionLevels < 1) { fprintf( stderr, "[Korali] MCMC Error: Rejection Level must be at least One (is %lu)\n", _rejectionLevels); exit(-1); }
 
 size_t nonAdaptionPeriodDefault = _s*0.05; 
 _adaptive                 = consume(js, { "MCMC", "Adaptive MCMC" }, KORALI_BOOLEAN, "false");
 _nonAdaptionPeriod        = consume(js, { "MCMC", "Non Adaption Period" }, KORALI_NUMBER, std::to_string(nonAdaptionPeriodDefault));
 _cr                       = consume(js, { "MCMC", "Chain Covariance Learning Rate" }, KORALI_NUMBER, std::to_string(2.4*2.4/_k->N)); //Gelman et al. 1995
 _eps                      = consume(js, { "MCMC", "Chain Covariance Increment" }, KORALI_NUMBER, std::to_string(0.01)); // sth small (Haario et. al. 2006)
 
 if (_cr < 0) { fprintf( stderr, "[Korali] MCMC Error: Chain Covariance Learning Rate must be larger Zero (is %f)\n", _cr); exit(-1); }
 if (_eps < 0) { fprintf( stderr, "[Korali] MCMC Error: Chain Covariance Increment must be larger Zero (is %f)\n", _eps); exit(-1); }
 
 _maxresamplings           = consume(js, { "MCMC", "Max Resamplings" }, KORALI_NUMBER, std::to_string(1e6));
 _termCondMaxFunEvals      = consume(js, { "MCMC", "Termination Criteria", "Max Function Evaluations", "Value" }, KORALI_NUMBER, std::to_string(1e4));
 _isTermCondMaxFunEvals    = consume(js, { "MCMC", "Termination Criteria", "Max Function Evaluations", "Active" }, KORALI_BOOLEAN, "false");
 _termCondMaxGenerations   = consume(js, { "MCMC", "Termination Criteria", "Max Sample Generations", "Value" }, KORALI_NUMBER, std::to_string(1e12));
 _isTermCondMaxGenerations = consume(js, { "MCMC", "Termination Criteria", "Max Sample Generations", "Active" }, KORALI_BOOLEAN, "false");
  
  _initialMean    = (double*) calloc(sizeof(double), _k->N);
  _initialStdDevs = (double*) calloc(sizeof(double), _k->N);

  for(size_t d = 0; d < _k->N; d++) _initialMean[d] = consume(js["Variables"][d], { "MCMC", "Initial Mean" }, KORALI_NUMBER, std::to_string(0.0));
  for(size_t d = 0; d < _k->N; d++) _initialStdDevs[d] = consume(js["Variables"][d], { "TMCMC", "Initial Standard Deviation" }, KORALI_NUMBER, std::to_string(1.0));
  
  for(size_t d = 0; d < _k->N; d++) if (_initialStdDevs[d] < 0) { fprintf( stderr, "[Korali] MCMC Error: Initial Standard Deviation in dim %zu must be larger Zero (is %f)\n", d, _initialStdDevs[d]); exit(-1); }

}

void Korali::Solver::MCMC::setState(nlohmann::json& js)
{
 countevals              = js["MCMC"]["State"]["FunctionEvaluations"];
 naccept                 = js["MCMC"]["State"]["Number Accepted Samples"];
 chainLength             = js["MCMC"]["State"]["Chain Length"];
 databaseEntries         = js["MCMC"]["State"]["Database Entries"];
 acceptanceRateProposals = js["MCMC"]["State"]["Acceptance Rate Proposals"];

 for (size_t d = 0; d < _k->N; d++) clPoint[d]                        = js["MCMC"]["State"]["Leader"][d];
 //for (size_t d = 0; d < _k->N; d++) ccPoint[d]                        = js["MCMC"]["State"]["Candidate"][d];
 for (size_t i = 0; i < _k->N*databaseEntries; i++) databasePoints[i] = js["MCMC"]["State"]["DatabasePoints"][i];
 for (size_t i = 0; i < databaseEntries; i++) databaseFitness[i]      = js["MCMC"]["State"]["DatabaseFitness"][i];
 for (size_t d = 0; d < _k->N; d++) chainMean[d]                      = js["MCMC"]["State"]["Chain Mean"][d];
 for (size_t d = 0; d < _k->N * _k->N; d++) chainCov[d]               = js["MCMC"]["State"]["Chain Covariance"][d];
 for (size_t d = 0; d < _k->N*_k->N; d++) _covarianceMatrix[d]        = js["MCMC"]["State"]["CovarianceMatrix"][d];
 
 clLogLikelihood = js["MCMC"]["State"]["LeaderFitness"];
 //ccLogLikelihood = js["MCMC"]["State"]["CandidateFitness"];

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Solver::MCMC::run()
{
  _k->saveState(countevals);
 if (_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Starting MCMC.\n");
 
 startTime = std::chrono::system_clock::now();

 while(!checkTermination())
 {
  t0 = std::chrono::system_clock::now();

  for(size_t r = 0; r < _rejectionLevels; ++r)
  {
    generateCandidate(r);
    _k->_conduit->evaluateSample(ccPoint, r); countevals++;
    _k->_conduit->checkProgress();
    acceptReject(r);
  }
  if (chainLength > _burnin ) updateDatabase(clPoint, clLogLikelihood);
  updateState();

  t1 = std::chrono::system_clock::now();

  _k->saveState(countevals);
  
  printGeneration();
  if (databaseEntries == _s) break;
 }

 printFinal();

 endTime = std::chrono::system_clock::now();
}


void Korali::Solver::MCMC::processSample(size_t lvl, double fitness)
{
 ccLogLikelihood[lvl] = fitness + ccLogPrior[lvl];
}


void Korali::Solver::MCMC::acceptReject(size_t level)
{
<<<<<<< HEAD

 double D, N; // denominator, nominator
 double prevLogLikelihood;

 for(size_t l = 0; l <= level; ++l)
 {
  double D = 1.0;
  alpha[l] = recursiveAlpha(D, clLogLikelihood, ccLogLikelihood, l);
  if ( alpha[l] >= 1.0 || alpha[l] > gsl_ran_flat(_gslGen, 0.0, 1.0) ) {
    naccept++;
    clLogLikelihood = ccLogLikelihood[l];
    for (size_t d = 0; d < _k->N; d++) clPoint[d] = ccPoint[l*_k->N+d];
    break;
  }
 }
 chainLength++;
}


double Korali::Solver::MCMC::recursiveAlpha(double& D, const double llk0, const double* logliks, size_t N) const
{
    if(N==0)
    {
        D = logliks[0];
        return std::min(1.0, exp(logliks[0]/llk0));
    }
    else
    {
        double D1 = D;
        D = D1 * (1-recursiveAlpha(D1, logliks[0], logliks+1, N-1));
        double* revLlks = new double[N-1];
        for(size_t i = 0; i < N-1; ++i) revLlks[i] = logliks[N-1-i];
        double D2 = D;
        double num = llk0 * ( 1 - std::min(1.0, exp(llk0/logliks[N-1])) ) * recursiveAlpha( D2, logliks[N], revLlks, N-1);
        delete [] revLlks;
        return num/D;
    }
}


void Korali::Solver::MCMC::updateDatabase(double* point, double loglik)
{
 for (size_t d = 0; d < _k->N; d++) databasePoints[databaseEntries*_k->N + d] = point[d];
 databaseFitness[databaseEntries] = loglik;
 databaseEntries++;
}


void Korali::Solver::MCMC::generateCandidate(size_t level)
{  
 size_t initialgens = countgens;
 sampleCandidate(level); countgens++;
 setCandidatePriorAndCheck(level);
 
 /*do { /TODO: fix check (DW)
   sampleCandidate(level); countgens++;
   setCandidatePriorAndCheck(level);
   if ( (countgens - initialgens) > _maxresamplings) 
   {       
     if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: exiting resampling loop, max resamplings (%zu) reached.\n", _maxresamplings);
     exit(-1);
   }
  } while (setCandidatePriorAndCheck(level)); */
}


void Korali::Solver::MCMC::sampleCandidate(size_t level)
{  
 for (size_t d = 0; d < _k->N; ++d) { z[d] = _gaussianGenerator->getRandomNumber(); ccPoint[level*_k->N+d] = 0.0; }

 if ( _adaptive && databaseEntries < _nonAdaptionPeriod) 
     for (size_t d = 0; d < _k->N; ++d) for (size_t e = 0; e < _k->N; ++e) ccPoint[level*_k->N+d] += _covarianceMatrix[d*_k->N+e] * z[e];
 else
     for (size_t d = 0; d < _k->N; ++d) for (size_t e = 0; e < _k->N; ++e) ccPoint[level*_k->N+d] += chainCov[d*_k->N+e] * z[e];

 for (size_t d = 0; d < _k->N; ++d) ccPoint[level*_k->N+d] += clPoint[d];
}


bool Korali::Solver::MCMC::setCandidatePriorAndCheck(size_t level)
{
 ccLogPrior[level] = _k->_problem->evaluateLogPrior(ccPoint+_k->N*level);
 if (ccLogPrior[level] > -INFINITY) return true;
 return false;
}


void Korali::Solver::MCMC::updateState()
{
 if(chainLength == 0) return;

 acceptanceRateProposals = ( (double)naccept/ (double)chainLength );

 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
   tmpC[d*_k->N+e] = databaseEntries*chainMean[d]*chainMean[e] + clPoint[d]*clPoint[e];
   tmpC[e*_k->N+d] = databaseEntries*chainMean[d]*chainMean[e] + clPoint[d]*clPoint[e];
 }
 for (size_t d = 0; d < _k->N; d++) tmpC[d*_k->N+d] = databaseEntries*chainMean[d]*chainMean[d] + clPoint[d]*clPoint[d] + _eps;

 // Chain Mean
 for (size_t d = 0; d < _k->N; d++) chainMean[d] = (chainMean[d] * (databaseEntries-1) + clPoint[d]) / ((double) databaseEntries);
 
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
    tmpC[d*_k->N+e] -= (databaseEntries+1)*chainMean[d]*chainMean[e];
    tmpC[e*_k->N+d] -= (databaseEntries+1)*chainMean[d]*chainMean[e];

 }
 for (size_t d = 0; d < _k->N; d++) tmpC[d*_k->N+d] -= (databaseEntries+1)*chainMean[d]*chainMean[d];

 // Chain Covariance (TODO: careful check N (databasEntires) (DW)
 for (size_t d = 0; d < _k->N; d++) for (size_t e = 0; e < d; e++)
 {
   chainCov[d*_k->N+e] = (databaseEntries-1.0)/( (double) databaseEntries) * chainCov[d*_k->N+e] + (_cr/( (double) databaseEntries))*tmpC[d*_k->N+e];
   chainCov[e*_k->N+d] = (databaseEntries-1.0)/( (double) databaseEntries) * chainCov[d*_k->N+e] + (_cr/( (double) databaseEntries))*tmpC[d*_k->N+e];
 }
 for (size_t d = 0; d < _k->N; d++) chainCov[d*_k->N+d] = (databaseEntries-1.0)/( (double) databaseEntries) * chainCov[d*_k->N+d] + (_cr/( (double) databaseEntries))*tmpC[d*_k->N+d];

 
}

bool Korali::Solver::MCMC::checkTermination()
{

 bool terminate = false;

 if ( _isTermCondMaxFunEvals && (countevals >= _termCondMaxFunEvals))
 {
  terminate = true;
  sprintf(_terminationReason, "Max Function Evaluations reached (%zu)",  countevals);
 }
 
 if ( _isTermCondMaxGenerations && (countgens >= _termCondMaxGenerations))
 {
  terminate = true;
  sprintf(_terminationReason, "Max Samples generated (%zu)",  countgens);
 }
 
 return terminate;
}

 
void Korali::Solver::MCMC::printGeneration() const
{
 if ((countgens) % _k->_outputFrequency != 0) return;
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
  printf("--------------------------------------------------------------------\n");
  printf("[Korali] Database Entries %ld\n", databaseEntries);
  printf("[Korali] Duration: %fs (Elapsed Time: %.2fs)\n",  
          std::chrono::duration<double>(t1-t0).count() , 
          std::chrono::duration<double>(t1-startTime).count());
 }

 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Accepted Samples: %zu\n", naccept);
 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Acceptance Rate Proposals: %.2f%%\n", 100*acceptanceRateProposals);

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (Current Sample, Current Candidate):\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), clPoint[d], ccPoint[d]);
  printf("[Korali] Current Chain Mean:\n");
  for (size_t d = 0; d < _k->N; d++) printf(" %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), chainMean[d]);
  printf("[Korali] Current Chain Variance:\n");
  for (size_t d = 0; d < _k->N; d++)
  {
   for (size_t e = 0; e <= d; e++) printf("   %+6.3e  ", chainCov[d*_k->N+e]);
   printf("\n");
  }


 }
}


void Korali::Solver::MCMC::printFinal() const
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] %s Finished\n", _name.c_str());
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Generated Samples: %zu\n", countgens);
    printf("[Korali] Acceptance Rate: %.2f%%\n", 100*acceptanceRateProposals);
    if (databaseEntries == _s) printf("[Korali] Max Samples Reached.\n");
    else printf("[Korali] Stopping Criterium: %s\n", _terminationReason);
    printf("[Korali] Total Elapsed Time: %fs\n", std::chrono::duration<double>(t1-startTime).count());
    printf("--------------------------------------------------------------------\n");
 }
}

