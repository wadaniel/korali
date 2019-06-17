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
 ccPoint           = (double*) calloc (_k->N, sizeof(double));
 databasePoints    = (double*) calloc (_k->N*_s, sizeof(double));
 databaseFitness   = (double*) calloc (_s, sizeof(double));
 chainMean         = (double*) calloc (_k->N, sizeof(double));
 chainVar          = (double*) calloc (_k->N, sizeof(double));

 for(size_t d = 0; d < _k->N; ++d) clPoint[d] = _initialMean[d];
 for(size_t d = 0; d < _k->N; ++d) _covarianceMatrix[d*_k->N+d] = _initialStdDevs[d];

 /*
 if(_useLocalCov) {
   double *LCmem       = (double*)  calloc (_s*_k->N*_k->N, sizeof(double));
   local_cov           = (double**) calloc ( _s, sizeof(double*));
   for (size_t pos = 0; pos < _s; ++pos)
   {
    local_cov[pos] = LCmem + pos*_k->N*_k->N;
    for (size_t i = 0; i < _k->N; i++) local_cov[pos][i*_k->N+i] = 1;
   }
 }*/
 
 // Initializing Gaussian Generator
 auto jsGaussian = nlohmann::json();
 jsGaussian["Name"] = "CMA-ES Generator";
 jsGaussian["Type"] = "Gaussian";
 jsGaussian["Mean"] = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = new Variable(jsGaussian);

 countevals               = 0;
 naccept                  = 0;
 countgens                = 0;
 chainLength              = 0;
 databaseEntries          = 0;
 clLogLikelihood          = -std::numeric_limits<double>::max();
 ccLogLikelihood          = -std::numeric_limits<double>::max();
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

 js["MCMC"]["Population Size"] = _s;
 js["MCMC"]["Burn In"]         = _burnin;
 js["MCMC"]["Max Resamplings"] = _maxresamplings;
 //js["Use Local Covariance"]     = _useLocalCov;

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
 for (size_t d = 0; d < _k->N; d++) js["MCMC"]["State"]["Candidate"][d]                      = ccPoint[d];
 for (size_t i = 0; i < _k->N*databaseEntries; i++) js["MCMC"]["State"]["DatabasePoints"][i] = databasePoints[i];
 for (size_t i = 0; i < databaseEntries; i++) js["MCMC"]["State"]["DatabaseFitness"][i]                = databaseFitness[i];
 for (size_t d = 0; d < _k->N; d++) js["MCMC"]["State"]["Chain Mean"][d]                     = chainMean[d];
 for (size_t d = 0; d < _k->N; d++) js["MCMC"]["State"]["Chain Variance"][d]                 = chainVar[d];
 //if (_useLocalCov) for (size_t i = 0; i < _chainLength; i++) for (size_t d = 0; d < _k->N; d++) js["State"]["LocalCovarianceMatrix"][i][d] = local_cov[i][d];

 js["MCMC"]["State"]["LeaderFitness"]    = clLogLikelihood;
 js["MCMC"]["State"]["CandidateFitness"] = ccLogLikelihood;

 return js;
}

void Korali::Solver::MCMC::setConfiguration(nlohmann::json& js)
{
 _s                        = consume(js, { "MCMC", "Population Size" }, KORALI_NUMBER);
 _burnin                   = consume(js, { "MCMC", "Burn In" }, KORALI_NUMBER, std::to_string(0));
 _maxresamplings           = consume(js, { "MCMC", "Max Resamplings" }, KORALI_NUMBER, std::to_string(1e6));
 
 _termCondMaxFunEvals      = consume(js, { "MCMC", "Termination Criteria", "Max Function Evaluations", "Value" }, KORALI_NUMBER, std::to_string(1e4));
 _isTermCondMaxFunEvals    = consume(js, { "MCMC", "Termination Criteria", "Max Function Evaluations", "Active" }, KORALI_BOOLEAN, "false");
 _termCondMaxGenerations   = consume(js, { "MCMC", "Termination Criteria", "Max Sample Generations", "Value" }, KORALI_NUMBER, std::to_string(1e12));
 _isTermCondMaxGenerations = consume(js, { "MCMC", "Termination Criteria", "Max Sample Generations", "Active" }, KORALI_BOOLEAN, "false");
  
  _initialMean    = (double*) calloc(sizeof(double), _k->N);
  _initialStdDevs = (double*) calloc(sizeof(double), _k->N);

  for(size_t i = 0; i < _k->N; i++) _initialMean[i] = consume(js["Variables"][i], { "MCMC", "Initial Mean" }, KORALI_NUMBER);
  for(size_t i = 0; i < _k->N; i++) _initialStdDevs[i] = consume(js["Variables"][i], { "MCMC", "Initial Standard Deviation" }, KORALI_NUMBER);

 //_useLocalCov       = consume(js, { "Use Local Covariance" }, KORALI_BOOLEAN, "false");
}

void Korali::Solver::MCMC::setState(nlohmann::json& js)
{
 countevals              = js["MCMC"]["State"]["FunctionEvaluations"];
 naccept                 = js["MCMC"]["State"]["Number Accepted Samples"];
 chainLength             = js["MCMC"]["State"]["Chain Length"];
 databaseEntries         = js["MCMC"]["State"]["Database Entries"];
 acceptanceRateProposals = js["MCMC"]["State"]["Acceptance Rate Proposals"];

 for (size_t d = 0; d < _k->N; d++) clPoint[d]                        = js["MCMC"]["State"]["Leader"][d];
 for (size_t d = 0; d < _k->N; d++) ccPoint[d]                        = js["MCMC"]["State"]["Candidate"][d];
 for (size_t i = 0; i < _k->N*databaseEntries; i++) databasePoints[i] = js["MCMC"]["State"]["DatabasePoints"][i];
 for (size_t i = 0; i < databaseEntries; i++) databaseFitness[i]      = js["MCMC"]["State"]["DatabaseFitness"][i];
 for (size_t d = 0; d < _k->N; d++) chainMean[d]                      = js["MCMC"]["State"]["Chain Mean"][d];
 for (size_t d = 0; d < _k->N; d++) chainVar[d]                       = js["MCMC"]["State"]["Chain Variance"][d];
 for (size_t d = 0; d < _k->N*_k->N; d++) _covarianceMatrix[d] = js["MCMC"]["State"]["CovarianceMatrix"][d];
 //if (_useLocalCov) for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->N; j++) local_cov[i][j] = js["State"]["LocalCovarianceMatrix"][i][j];
 
 clLogLikelihood = js["MCMC"]["State"]["LeaderFitness"];
 ccLogLikelihood = js["MCMC"]["State"]["CandidateFitness"];

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

  generateCandidate();
  _k->_conduit->evaluateSample(ccPoint, 0); countevals++;
  _k->_conduit->checkProgress();
  acceptReject();
  updateState();

  t1 = std::chrono::system_clock::now();

  _k->saveState(countevals);
  
  printGeneration();
  if (databaseEntries == _s) break;
 }

 printFinal();

 endTime = std::chrono::system_clock::now();
}


void Korali::Solver::MCMC::processSample(size_t c, double fitness)
{
 ccLogLikelihood = fitness + ccLogPrior;
}

void Korali::Solver::MCMC::acceptReject()
{
 double L = exp(ccLogLikelihood-clLogLikelihood);
 if ( L >= 1.0 || L > gsl_ran_flat(_gslGen, 0.0, 1.0) ) {
   naccept++;
   clLogLikelihood = ccLogLikelihood;
   for (size_t d = 0; d < _k->N; d++) clPoint[d] = ccPoint[d];
 }
 chainLength++;
 if (chainLength > _burnin ) updateDatabase(clPoint, clLogLikelihood);
}

void Korali::Solver::MCMC::updateDatabase(double* point, double loglik)
{
 for (size_t d = 0; d < _k->N; d++) databasePoints[databaseEntries*_k->N + d] = point[d];
 databaseFitness[databaseEntries] = loglik;
 databaseEntries++;
}


void Korali::Solver::MCMC::generateCandidate()
{  
 size_t initialgens = countgens;
 //do { /TODO: fix check (DW)
   sampleCandidate(); countgens++;
   if ( (countgens - initialgens) > _maxresamplings) 
   {       
     if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: exiting resampling loop, max resamplings (%zu) reached.\n", _maxresamplings);
     exit(-1);
   }
  //} while (setCandidatePriorAndCheck());
}


void Korali::Solver::MCMC::sampleCandidate()
{  

 for (size_t d = 0; d < _k->N; ++d) { z[d] = _gaussianGenerator->getRandomNumber(); ccPoint[d] = 0.0; }
 for (size_t d = 0; d < _k->N; ++d) for (size_t e = 0; e < _k->N; ++e) ccPoint[d] += _covarianceMatrix[d*_k->N+e] * z[e];

 for (size_t d = 0; d < _k->N; ++d) ccPoint[d] += clPoint[d];
}


bool Korali::Solver::MCMC::setCandidatePriorAndCheck()
{
 ccLogPrior = _k->_problem->evaluateLogPrior(ccPoint);
 if (ccLogPrior > -INFINITY) return true;
 return false;
}


void Korali::Solver::MCMC::updateState()
{
 if(databaseEntries < 2) return;
 acceptanceRateProposals = ( (double)naccept/ (double)chainLength );

 // Chain Mean and Variance
 for (size_t d = 0; d < _k->N; d++)
 {
    double delta = (clPoint[d] - chainMean[d]);
    chainMean[d] += (delta)/((double)databaseEntries);
    chainVar[d] = (chainVar[d] * (databaseEntries-2.0) + delta * delta)/(databaseEntries-1.0); //TODO: verify (DW)
 }

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
  printf("[Korali] Current Sample:\n");
  for (size_t d = 0; d < _k->N; d++) printf(" %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), clPoint[d]);
  printf("[Korali] Current Candidate:\n");
  for (size_t d = 0; d < _k->N; d++) printf(" %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), ccPoint[d]);
  printf("[Korali] Current Chain Mean:\n");
  for (size_t d = 0; d < _k->N; d++) printf(" %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), chainMean[d]);
  printf("[Korali] Current Chain Variance:\n");
  for (size_t d = 0; d < _k->N; d++) printf(" %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), chainVar[d]);
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

