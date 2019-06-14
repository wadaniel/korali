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

Korali::Solver::MCMC::MCMC(nlohmann::json& js, std::string name) : Korali::Solver::Base::Base(js)
{
 _name = name;
 setConfiguration(js);

 // Setting Chain-Specific Seeds
 gslGen = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(gslGen, _k->_seed++);

 // Allocating MCMC memory
 _covarianceMatrix = (double*) calloc (_k->_problem->N*_k->_problem->N, sizeof(double));
 clPoint           = (double*) calloc (_k->_problem->N, sizeof(double));
 ccPoint           = (double*) calloc (_k->_problem->N, sizeof(double));
 databasePoints    = (double*) calloc (_k->_problem->N*_s, sizeof(double));
 databaseFitness   = (double*) calloc (_s, sizeof(double));

 /*
 if(_useLocalCov) {
   double *LCmem       = (double*)  calloc (_s*_k->_problem->N*_k->_problem->N, sizeof(double));
   local_cov           = (double**) calloc ( _s, sizeof(double*));
   for (size_t pos = 0; pos < _s; ++pos)
   {
    local_cov[pos] = LCmem + pos*_k->_problem->N*_k->_problem->N;
    for (size_t i = 0; i < _k->_problem->N; i++) local_cov[pos][i*_k->_problem->N+i] = 1;
   }
 }*/

 countevals               = 0;
 countgens                = 0;
 chainLength              = 0;
 databaseEntries          = 0;
 clLogLikelihood          = -std::numeric_limits<double>::max();
 ccLogLikelihood          = -std::numeric_limits<double>::max();
 acceptanceRateProposals  = 1.0;

 // If state is defined:
 if (isDefined(js, {"State"}))
 {
  setState(js);
  js.erase("State");
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
 auto js = this->Korali::Solver::Base::getConfiguration();

 js["Method"] = _name;

 js["Population Size"] = _s;
 js["Burn In"]         = _burnin;
 js["Max Resamplings"] = _maxresamplings;
 //js["Use Local Covariance"]     = _useLocalCov;

 js["Termination Criteria"]["Max Function Evaluations"]["Value"]  = _termCondMaxFunEvals;
 js["Termination Criteria"]["Max Function Evaluations"]["Active"] = _isTermCondMaxFunEvals;
 js["Termination Criteria"]["Max Generated Samples"]["Value"]     = _termCondMaxGenerations;
 js["Termination Criteria"]["Max Generated Samples"]["Active"]    = _isTermCondMaxGenerations;

 // State Variables
 for (size_t d = 0; d < _k->_problem->N*_k->_problem->N; d++) js["State"]["CovarianceMatrix"][d] = _covarianceMatrix[d];
 
 js["State"]["Function Evaluations"]    = countevals;
 js["State"]["Chain Length"]            = chainLength;
 js["State"]["AcceptanceRateProposals"] = acceptanceRateProposals;
 for (size_t d = 0; d < _k->_problem->N; d++) js["State"]["Leader"][d]                         = clPoint[d];
 for (size_t d = 0; d < _k->_problem->N; d++) js["State"]["Candidate"][d]                      = ccPoint[d];
 for (size_t i = 0; i < _k->_problem->N*databaseEntries; i++) js["State"]["DatabasePoints"][i] = databasePoints[i];
 for (size_t i = 0; i < databaseEntries; i++) js["State"]["DatabaseFitness"][i]                = databaseFitness[i];
 //if (_useLocalCov) for (size_t i = 0; i < _chainLength; i++) for (size_t d = 0; d < _k->_problem->N; d++) js["State"]["LocalCovarianceMatrix"][i][d] = local_cov[i][d];

 js["State"]["LeaderFitness"]    = clLogLikelihood;
 js["State"]["CandidateFitness"] = ccLogLikelihood;

 return js;
}

void Korali::Solver::MCMC::setConfiguration(nlohmann::json& js)
{
 this->Korali::Solver::Base::setConfiguration(js);

 _s                        = consume(js, { "Population Size" }, KORALI_NUMBER);
 _burnin                   = consume(js, { "Burn In" }, KORALI_NUMBER, std::to_string(0));
 _maxresamplings           = consume(js, { "Max Resamplings" }, KORALI_NUMBER, std::to_string(1e6));
 
 _termCondMaxFunEvals      = consume(js, { "Termination Criteria", "Max Function Evaluations", "Value" }, KORALI_NUMBER, std::to_string(1e4));
 _isTermCondMaxFunEvals    = consume(js, { "Termination Criteria", "Max Function Evaluations", "Active" }, KORALI_BOOLEAN, "false");
 _termCondMaxGenerations   = consume(js, { "Termination Criteria", "Max Sample Generations", "Value" }, KORALI_NUMBER, std::to_string(1e12));
 _isTermCondMaxGenerations = consume(js, { "Termination Criteria", "Max Sample Generations", "Active" }, KORALI_BOOLEAN, "false");
 for (size_t d = 0; d < _k->_problem->N*_k->_problem->N; d++) _covarianceMatrix[d] = js["State"]["CovarianceMatrix"][d];
 //_useLocalCov       = consume(js, { "Use Local Covariance" }, KORALI_BOOLEAN, "false");
}

void Korali::Solver::MCMC::setState(nlohmann::json& js)
{
 this->Korali::Solver::Base::setState(js);

 countevals              = js["State"]["FunctionEvaluations"];
 chainLength             = js["State"]["Chain Length"];
 acceptanceRateProposals = js["State"]["AcceptanceRateProposals"];

 for (size_t d = 0; d < _k->_problem->N; d++) clPoint[d]                        = js["State"]["Leader"][d];
 for (size_t d = 0; d < _k->_problem->N; d++) ccPoint[d]                        = js["State"]["Candidate"][d];
 for (size_t i = 0; i < _k->_problem->N*databaseEntries; i++) databasePoints[i] = js["State"]["DatabasePoints"][i];
 for (size_t i = 0; i < databaseEntries; i++) databaseFitness[i]                = js["State"]["DatabaseFitness"][i];
 //if (_useLocalCov) for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->_problem->N; j++) local_cov[i][j] = js["State"]["LocalCovarianceMatrix"][i][j];
 
 clLogLikelihood = js["State"]["LeaderFitness"];
 ccLogLikelihood = js["State"]["CandidateFitness"];

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Solver::MCMC::run()
{
  if (_k->_conduit->_name != "Sequential")                                        
  {                                                                              
    fprintf(stderr, "[Korali] Error: The MCMC Method can only be used with Sequanetial conduit (is %s).\n", _k->_conduit->_name.c_str());
    exit(-1);
  } 
 
 if (_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Starting MCMC.\n");
 
 startTime = std::chrono::system_clock::now();

 while(!checkTermination())
 {
  t0 = std::chrono::system_clock::now();

  // Generating Samples
  generateCandidate();
  _k->_conduit->evaluateSample(ccPoint, 0); countevals++;
  updateState();

  t1 = std::chrono::system_clock::now();

  _k->saveState(countevals);
  
  if (databaseEntries >= _s) break;
  printGeneration();
 }

 printFinal();

 endTime = std::chrono::system_clock::now();
}


void Korali::Solver::MCMC::processSample(size_t c, double fitness)
{
 ccLogLikelihood = fitness + ccLogPrior;
 double L = exp(ccLogLikelihood-clLogLikelihood);

 if ( L >= 1.0 || L > gsl_ran_flat(gslGen, 0.0, 1.0) ) {
   chainLength++;
   clLogLikelihood = ccLogLikelihood;
   for (size_t d = 0; d < _k->_problem->N; d++) clPoint[d] = ccPoint[d];

   if (chainLength > _burnin ) updateDatabase(ccPoint, ccLogLikelihood);
 }
}


void Korali::Solver::MCMC::updateDatabase(double* point, double loglik)
{
 databaseEntries++;
 for (size_t d = 0; d < _k->_problem->N; d++) databasePoints[databaseEntries*_k->_problem->N + d] = point[d];
 databaseFitness[databaseEntries] = loglik;
}


void Korali::Solver::MCMC::generateCandidate()
{  
 size_t initialgens = countgens;
 do {
   sampleCandidate(); countgens++;
   if ( (countgens - initialgens) > _maxresamplings) 
   {       
     if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: exiting resampling loop, max resamplings (%zu) reached.\n", _maxresamplings);
     exit(-1);
   }
  } while (setCandidatePriorAndCheck());
}


void Korali::Solver::MCMC::sampleCandidate()
{
 //double* covariance = _useLocalCov ? local_cov[c] : _covarianceMatrix;
 gsl_vector_view out_view    = gsl_vector_view_array(ccPoint, _k->_problem->N);
 gsl_vector_view mean_view   = gsl_vector_view_array(clPoint, _k->_problem->N);
 gsl_matrix_view sigma_view  = gsl_matrix_view_array(_covarianceMatrix, _k->_problem->N,_k->_problem->N);
 gsl_ran_multivariate_gaussian(gslGen, &mean_view.vector, &sigma_view.matrix, &out_view.vector);
}


bool Korali::Solver::MCMC::setCandidatePriorAndCheck()
{
 ccLogPrior = _k->_problem->evaluateLogPrior(ccPoint);
 if (ccLogPrior > -INFINITY) return true;
 return false;
}


void Korali::Solver::MCMC::updateState()
{
 
 acceptanceRateProposals = ( (double) countgens / (double) chainLength );

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

 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Acceptance Rate proposals: %.2f%%\n", 100*acceptanceRateProposals);

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Current Sample:\n");
  for (size_t d = 0; d < _k->_problem->N; d++) printf(" %s = %+6.3e\n", _k->_problem->_variables[d]->_name.c_str(), ccPoint[d]);
 }
}


void Korali::Solver::MCMC::printFinal() const
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] %s Finished\n", _name.c_str());
    if (databaseEntries == _s) printf("[Korali] Max Samples Reached.\n");
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Generated Samples: %zu\n", countgens);
    printf("[Korali] Acceptance Rate: %zu\n", acceptanceRateProposals);
    printf("[Korali] Stopping Criterium: %s\n", _terminationReason);
    printf("[Korali] Total Elapsed Time: %fs\n", std::chrono::duration<double>(t1-t0).count());
    printf("--------------------------------------------------------------------\n");
 }
}

