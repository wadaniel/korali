#include "korali.h"
#include <numeric>
#include <limits>
#include <chrono>

#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_multimin.h>

typedef struct fparam_s {
    const double *fj;
    size_t        fn;
    double        pj;
    double        cov;
} fparam_t;

void Korali::Solver::TMCMC::initialize()
{
 // Setting Chain-Specific Seeds
 range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(range, _k->_seed++);

 chainGSLRange = (gsl_rng**) calloc (populationSize, sizeof(gsl_rng*));
 for (size_t c = 0; c < populationSize; c++)
 {
 chainGSLRange[c] = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(chainGSLRange[c], _k->_seed++);
 }

 // Allocating TMCMC memory
 _covarianceMatrix   = (double*) calloc (_k->N*_k->N, sizeof(double));
 _meanTheta          = (double*) calloc (_k->N, sizeof(double));
 _variableLogSpace   = (bool*) calloc (_k->N, sizeof(bool));
 logTransformedSamples.reserve(_k->N*populationSize);
 chainCandidatesParameters.reserve(_k->N*populationSize);
 chainCandidatesLogLikelihoods.reserve(populationSize);
 chainLeadersParameters.reserve(_k->N*populationSize);
 chainLeadersLogLikelihoods.reserve(populationSize);
 chainPendingFitness.reserve(populationSize);
 currentChainStep.reserve(populationSize);
 chainLengths.reserve(populationSize);
 _databasePoints     = (double*) calloc (_k->N*populationSize, sizeof(double));
 _databaseFitness    = (double*) calloc (populationSize, sizeof(double));

 if(useLocalCovariance) {
  double *LCmem       = (double*)  calloc (populationSize*_k->N*_k->N, sizeof(double));
  local_cov           = (double**) calloc ( populationSize, sizeof(double*));
  for (size_t pos = 0; pos < populationSize; ++pos)
  {
  local_cov[pos] = LCmem + pos*_k->N*_k->N;
  for (size_t i = 0; i < _k->N; i++) local_cov[pos][i*_k->N+i] = 1;
  }
 }

 // Initializing Runtime Variables
 for (size_t c = 0; c < populationSize; c++) currentChainStep[c] = 0;
 for (size_t c = 0; c < populationSize; c++) chainLengths[c] = 1;
 for (size_t c = 0; c < populationSize; c++) chainPendingFitness[c] = false;

 // Init Generation
 _isFinished = false;
 _databaseEntries          = 0;
 annealingExponent        = 0;
 logEvidence              = 0;
 coefficientOfVariation = initialCoefficientOfVariation;
 finishedChainsCount            = 0;
 _databaseEntries          = 0;
 proposalsAcceptanceRate  = 1.0;
 selectionAcceptanceRate = 1.0;
 acceptedSamplesCount            = populationSize;
 chainCount                  = populationSize;
 for (size_t c = 0; c < chainCount; c++) currentChainStep[c] = 0;
 for (size_t c = 0; c < chainCount; c++) chainPendingFitness[c] = false;

 initializeSamples();
 printGeneration();
}

void Korali::Solver::TMCMC::runGeneration()
{
 if (_k->currentGeneration > 1) resampleGeneration();

 while (finishedChainsCount < chainCount)
 {
  for (size_t c = 0; c < chainCount; c++) if (currentChainStep[c] < chainLengths[c]) if (chainPendingFitness[c] == false)
  {
  chainPendingFitness[c] = true;
  generateCandidate(c);
  evaluateSample(c);
  }
  _k->_conduit->checkProgress();
 }
}

void Korali::Solver::TMCMC::processSample(size_t c, double fitness)
{
 double ccLogPrior = _k->_problem->evaluateLogPrior(&chainCandidatesParameters[c*_k->N]);
 double clLogPrior = _k->_problem->evaluateLogPrior(&chainLeadersParameters[c*_k->N]);

 chainCandidatesLogLikelihoods[c] = fitness;
 double L = exp((chainCandidatesLogLikelihoods[c]-chainLeadersLogLikelihoods[c])*annealingExponent + (ccLogPrior-clLogPrior));

 if ( L >= 1.0 || L > gsl_ran_flat(chainGSLRange[c], 0.0, 1.0) ) {
   for (size_t i = 0; i < _k->N; i++) chainLeadersParameters[c*_k->N + i] = chainCandidatesParameters[c*_k->N + i];
   chainLeadersLogLikelihoods[c] = chainCandidatesLogLikelihoods[c];
   if (currentChainStep[c] == chainLengths[c]-1) acceptedSamplesCount++; // XXX: is that correct? (DW)
 }

 currentChainStep[c]++;
 if (currentChainStep[c] > burnIn ) updateDatabase(&chainLeadersParameters[c*_k->N], chainLeadersLogLikelihoods[c]);
 chainPendingFitness[c] = false;
 if (currentChainStep[c] == chainLengths[c]) finishedChainsCount++;
}

void Korali::Solver::TMCMC::evaluateSample(size_t c)
{
  for(size_t d = 0; d<_k->N; ++d) 
      if (_variableLogSpace[d] == true) 
          logTransformedSamples[c*_k->N+d] = std::exp(chainCandidatesParameters[c*_k->N+d]);
      else 
          logTransformedSamples[c*_k->N+d] = chainCandidatesParameters[c*_k->N+d];

  _k->_conduit->evaluateSample(&logTransformedSamples[0], c);
}

void Korali::Solver::TMCMC::updateDatabase(double* point, double fitness)
{
 for (size_t i = 0; i < _k->N; i++) _databasePoints[_databaseEntries*_k->N + i] = point[i];
 _databaseFitness[_databaseEntries] = fitness;
 _databaseEntries++;
}

void Korali::Solver::TMCMC::generateCandidate(size_t c)
{
 double* covariance = useLocalCovariance ? local_cov[c] : _covarianceMatrix;
 gsl_vector_view out_view    = gsl_vector_view_array(&chainCandidatesParameters[c*_k->N], _k->N);
 gsl_matrix_view sigma_view  = gsl_matrix_view_array(covariance, _k->N,_k->N);
 gsl_vector_view mean_view   = gsl_vector_view_array(&chainLeadersParameters[c*_k->N], _k->N);
 gsl_ran_multivariate_gaussian(chainGSLRange[c], &mean_view.vector, &sigma_view.matrix, &out_view.vector);
}

void Korali::Solver::TMCMC::initializeSamples()
{
  for (size_t c = 0; c < populationSize; c++) {
     for (size_t d = 0; d < _k->N; d++) {
       chainLeadersParameters[c*_k->N + d] = chainCandidatesParameters[c*_k->N + d] = _k->_variables[d]->getRandomNumber();
       chainLeadersLogLikelihoods[c] += log( _k->_variables[d]->getDensity(chainLeadersParameters[c*_k->N + d]) );
     }
     updateDatabase(&chainLeadersParameters[c*_k->N], chainLeadersLogLikelihoods[c]);
     finishedChainsCount++;
  }
}

void Korali::Solver::TMCMC::resampleGeneration()
{
 double* flcp     = (double*) calloc (_databaseEntries, sizeof(double));
 double* weight   = (double*) calloc (_databaseEntries, sizeof(double));
 double* q        = (double*) calloc (_databaseEntries, sizeof(double));
 unsigned int* nn = (unsigned int*) calloc (_databaseEntries, sizeof(unsigned int));
 size_t* sel      = (size_t*) calloc (_databaseEntries, sizeof(size_t));

 double fmin = 0, xmin = 0;
 minSearch(_databaseFitness, _databaseEntries, annealingExponent, coefficientOfVariation, xmin, fmin);

 double _prevAnnealingExponent = annealingExponent;

 if (xmin > _prevAnnealingExponent + maxRhoUpdate)
 {
  if ( _k->_verbosity >= KORALI_DETAILED ) printf("[Korali] Warning: Annealing Step larger than Max Rho Update, updating Annealing Exponent by %f (Max Rho Update). \n", maxRhoUpdate);
  annealingExponent      = _prevAnnealingExponent + maxRhoUpdate;
  coefficientOfVariation = sqrt(tmcmc_objlogp(annealingExponent, _databaseFitness, _databaseEntries, _prevAnnealingExponent, coefficientOfVariation)) + coefficientOfVariation;
 }
 else if (xmin > _prevAnnealingExponent)
 {
  annealingExponent      = xmin;
  coefficientOfVariation = sqrt(fmin) + coefficientOfVariation;
 }
 else
 {
  if ( _k->_verbosity >= KORALI_DETAILED ) printf("[Korali] Warning: Annealing Step smaller than Min Rho Update, updating Annealing Exponent by %f (Min Rho Update). \n", minRhoUpdate);
  annealingExponent      = _prevAnnealingExponent + minRhoUpdate;
  coefficientOfVariation = sqrt(tmcmc_objlogp(annealingExponent, _databaseFitness, _databaseEntries, _prevAnnealingExponent, coefficientOfVariation)) + coefficientOfVariation;
 }

 /* Compute weights and normalize*/

 for (size_t i = 0; i < _databaseEntries; i++) flcp[i] = _databaseFitness[i]*(annealingExponent-_prevAnnealingExponent);
 const double fjmax = gsl_stats_max(flcp, 1, _databaseEntries);
 for (size_t i = 0; i < _databaseEntries; i++) weight[i] = exp( flcp[i] - fjmax );

 double sum_weight = std::accumulate(weight, weight+_databaseEntries, 0.0);
 logEvidence  += log(sum_weight) + fjmax - log(_databaseEntries);

 for (size_t i = 0; i < _databaseEntries; i++) q[i] = weight[i]/sum_weight;

 gsl_ran_multinomial(range, _databaseEntries, populationSize, q, nn);
 size_t zeroCount = 0;
 for (size_t i = 0; i < _databaseEntries; i++) { sel[i] = nn[i]; if ( nn[i] == 0 ) zeroCount++; }

 size_t uniqueSelections   = _databaseEntries - zeroCount;
 proposalsAcceptanceRate  = (1.0*acceptedSamplesCount)/populationSize;
 selectionAcceptanceRate = (1.0*uniqueSelections)/populationSize;

 for (size_t i = 0; i < _k->N; i++)
 {
  _meanTheta[i] = 0;
  for (size_t j = 0; j < _databaseEntries; j++) _meanTheta[i] += _databasePoints[j*_k->N + i]*q[j];
 }

 for (size_t i = 0; i < _k->N; i++) for (size_t j = i; j < _k->N; ++j)
 {
  double s = 0.0;
  for (size_t k = 0; k < _databaseEntries; ++k) s += q[k]*(_databasePoints[k*_k->N+i]-_meanTheta[i])*(_databasePoints[k*_k->N+j]-_meanTheta[j]);
  _covarianceMatrix[i*_k->N + j] = _covarianceMatrix[j*_k->N + i] = s*covarianceScaling;
 }

 gsl_matrix_view sigma = gsl_matrix_view_array(_covarianceMatrix, _k->N,_k->N);
 gsl_linalg_cholesky_decomp( &sigma.matrix );

 size_t ldi = 0;
 for (size_t i = 0; i < _databaseEntries; i++) {
   if (sel[i] != 0) {
     for (size_t j = 0; j < _k->N ; j++) chainLeadersParameters[ldi*_k->N + j] = _databasePoints[i*_k->N + j];
     chainLeadersLogLikelihoods[ldi] = _databaseFitness[i];
     chainLengths[ldi] = sel[i] + burnIn;
     ldi++;
   }
 }
 
 if (useLocalCovariance) computeChainCovariances(local_cov, uniqueSelections);

 _databaseEntries = 0;
 acceptedSamplesCount   = 0;
 finishedChainsCount   = 0;
 chainCount         = uniqueSelections;
 
 for (size_t c = 0; c < chainCount; c++) currentChainStep[c] = 0;
 for (size_t c = 0; c < chainCount; c++) chainPendingFitness[c] = false;

 free(flcp);
 free(weight);
 free(q);
 free(nn);
 free(sel);
}

void Korali::Solver::TMCMC::computeChainCovariances(double** chain_cov, size_t newchains)
{
 //printf("Precomputing chain covariances for the current generation...\n");

 // allocate space
 size_t* nn_ind     = (size_t*) calloc (newchains, sizeof(size_t));
 size_t* nn_count   = (size_t*) calloc (newchains, sizeof(size_t));
 double* diam       = (double*) calloc (_k->N, sizeof(double));
 double* chain_mean = (double*) calloc (_k->N, sizeof(double));
 gsl_matrix* work   = gsl_matrix_alloc(_k->N, _k->N);

 // find diameters
 for (size_t d = 0; d < _k->N; ++d) {
  double d_min = +1e6;
  double d_max = -1e6;
  for (size_t pos = 0; pos < populationSize; ++pos) {
   double s = _databasePoints[pos*_k->N+d];
   if (d_min > s) d_min = s;
   if (d_max < s) d_max = s;
  }
  diam[d] = d_max-d_min;
  //printf("Diameter %ld: %.6lf\n", d, diam[d]);
 }

 size_t idx, pos;
 int status = 0;
 double ds = 0.05;
 for (double scale = 0.1; scale <= 1.0; scale += ds) {
  // find neighbors in a rectangle - O(populationSize^2)
  for (pos = 0; pos < newchains; ++pos) {
   nn_count[pos] = 0;
   double* curr = &chainLeadersParameters[pos*_k->N];
   for (size_t i = 0; i < populationSize; i++) {
    double* s = &_databasePoints[i*_k->N];
    bool isInRectangle = true;
     for (size_t d = 0; d < _k->N; d++)  if (fabs(curr[d]-s[d]) > scale*diam[d]) isInRectangle = false;
     if (isInRectangle) {
      nn_ind[pos*populationSize+nn_count[pos]] = i;
      nn_count[pos]++;
     }
   }
  }

  // compute the covariances
  for (pos = 0; pos < newchains; ++pos) {
   for (size_t d = 0; d < _k->N; ++d) {
    chain_mean[d] = 0;
    for (size_t k = 0; k < nn_count[pos]; ++k) {
     idx = nn_ind[pos*populationSize+k];
     chain_mean[d] += _databasePoints[idx*_k->N+d];
    }
    chain_mean[d] /= nn_count[pos];
   }

   for (size_t i = 0; i < _k->N; i++)
    for (size_t j = 0; j < _k->N; ++j) {
     double s = 0;
     for (size_t k = 0; k < nn_count[pos]; k++) {
      idx = nn_ind[pos*populationSize+k];
      s  += (_databasePoints[idx*_k->N+i]-chain_mean[i]) *
         (_databasePoints[idx*_k->N+j]-chain_mean[j]);
     }
     chain_cov[pos][i*_k->N+j] = chain_cov[pos][j*_k->N+i] = s/nn_count[pos];
    }

   // check if the matrix is positive definite
   for (size_t i = 0; i < _k->N; i++)
    for (size_t j = 0; j < _k->N; ++j) {
     double s = chain_cov[pos][i*_k->N+j];
     gsl_matrix_set(work, i, j, s);
    }
   gsl_set_error_handler_off();
   status = gsl_linalg_cholesky_decomp(work);
   if (status == GSL_SUCCESS) break;
  }
 }

 for (pos = 0; pos < newchains; ++pos) {
   gsl_matrix_view sigma  = gsl_matrix_view_array(chain_cov[pos], _k->N,_k->N);
   gsl_linalg_cholesky_decomp( &sigma.matrix );
 }

 if (status != GSL_SUCCESS) {
  fprintf(stderr, "[Korali] TMCMC Error: GSL failed to create Chain Covariance Matrix.\n");
 }

 // deallocate space
 free(nn_ind);
 free(nn_count);
 free(diam);
 free(chain_mean);
 gsl_matrix_free(work);
}

double Korali::Solver::TMCMC::tmcmc_objlogp(double x, const double *fj, size_t fn, double pj, double zero)
{
 double *weight = (double*) calloc (fn, sizeof(double));
 double *q      = (double*) calloc (fn, sizeof(double));
 const double fjmax = gsl_stats_max(fj, 1, fn);

 for(size_t i = 0; i <fn; i++)weight[i] = exp((fj[i]-fjmax)*(x-pj));
 double sum_weight = std::accumulate(weight, weight+fn, 0.0);
 for(size_t i = 0; i < fn; i++)  q[i] = weight[i]/sum_weight;

 double mean_q = gsl_stats_mean(q, 1, fn);
 double std_q  = gsl_stats_sd_m(q, 1, fn, mean_q);
 double cov2   = (std_q/mean_q-zero); cov2 *= cov2;

 free(weight);
 free(q);

 return cov2;
}

double Korali::Solver::TMCMC::objLog(const gsl_vector *v, void *param)
{
 double x = gsl_vector_get(v, 0);
 fparam_t *fp = (fparam_t *) param;
 return Korali::Solver::TMCMC::tmcmc_objlogp(x, fp->fj, fp->fn, fp->pj, fp->cov);
}

void Korali::Solver::TMCMC::minSearch(double const *fj, size_t fn, double pj, double objCov, double& xmin, double& fmin)
{
 // Minimizer Options
 size_t MaxIter     = 100;    /* Max number of search iterations */
 double Tol         = 1e-12;  /* Tolerance for root finding */
 double Step        = 1e-8;   /* Search stepsize */

 const gsl_multimin_fminimizer_type *T;
 gsl_multimin_fminimizer *s = NULL;
 gsl_vector *ss, *x;
 gsl_multimin_function minex_func;

 size_t iter     = 0;
 int status;
 double size;

 fparam_t fp;
 fp.fj = fj;
 fp.fn = fn;
 fp.pj = pj;
 fp.cov = objCov;

 x = gsl_vector_alloc (1);
 gsl_vector_set (x, 0, pj);

 ss = gsl_vector_alloc (1);
 gsl_vector_set_all (ss, Step);

 minex_func.n      = 1;
 minex_func.f      = objLog;
 minex_func.params = &fp;

 // SELECT ONE MINIMIZER STRATEGY
 T = gsl_multimin_fminimizer_nmsimplex;
 /* T = gsl_multimin_fminimizer_nmsimplex2; */
 /* T = gsl_multimin_fminimizer_nmsimplex2rand; (warning: not reliable)  */
 s = gsl_multimin_fminimizer_alloc (T, 1);
 gsl_multimin_fminimizer_set (s, &minex_func, x, ss);

 fmin = 0;
 xmin = 0.0;

 do {
   iter++;
   status = gsl_multimin_fminimizer_iterate(s);
   size   = gsl_multimin_fminimizer_size (s);
   status = gsl_multimin_test_size (size, Tol);
 } while (status == GSL_CONTINUE && iter < MaxIter);

 if (_k->_verbosity >= KORALI_DETAILED)
 {
   if(status == GSL_SUCCESS && s->fval >  Tol) printf("[Korali] Min Search converged but did not find minimum. \n");
   if(status != GSL_SUCCESS && s->fval <= Tol) printf("[Korali] Min Search did not converge but minimum found\n");
   if(status != GSL_SUCCESS && s->fval >  Tol) printf("[Korali] Min Search did not converge and did not find minimum\n");
   if(iter >= MaxIter) printf("[Korali] Min Search MaxIter (%zu) reached\n", MaxIter);
 }

 if (s->fval <= Tol) {
   fmin = s->fval;
   xmin = gsl_vector_get(s->x, 0);
 }

 if (xmin >= 1.0) {
   fmin = tmcmc_objlogp(1.0, fj, fn, pj, objCov);
   xmin = 1.0;
 }

 gsl_vector_free(x);
 gsl_vector_free(ss);
 gsl_multimin_fminimizer_free (s);
}


bool Korali::Solver::TMCMC::isFeasibleCandidate(size_t c)
{
 double clLogPrior = _k->_problem->evaluateLogPrior(&chainLeadersParameters[c*_k->N]);
 if (clLogPrior > -INFINITY) return true;
 return false;
}

bool Korali::Solver::TMCMC::checkTermination()
{

 _isFinished = (maxGenerationsEnabled && (_k->currentGeneration < maxGenerations));

 _isFinished = (annealingExponent >= 1.0);

 return _isFinished;
}

void Korali::Solver::TMCMC::finalize()
{

}

void Korali::Solver::TMCMC::printGeneration()
{
 if (_k->currentGeneration % terminalOutputFrequency != 0) return;
 
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
  printf("--------------------------------------------------------------------\n");
  printf("[Korali] Generation %ld - Annealing Exponent:  %.3e.\n", _k->currentGeneration, annealingExponent);
  if (maxGenerationsEnabled && (_k->currentGeneration == maxGenerations)) printf("[Korali] Max Generation Reached.\n");
 }

 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Acceptance Rate (proposals / selections): (%.2f%% / %.2f%%)\n", 100*proposalsAcceptanceRate, 100*selectionAcceptanceRate);
  printf("[Korali] Coefficient of Variation: %.2f%%\n", 100.0*coefficientOfVariation);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Sample Mean:\n");
  for (size_t i = 0; i < _k->N; i++) printf(" %s = %+6.3e\n", _k->_variables[i]->_name.c_str(), _meanTheta[i]);
  printf("[Korali] Sample Covariance:\n");
  for (size_t i = 0; i < _k->N; i++)
  {
   printf("   | ");
   for (size_t j = 0; j < _k->N; j++)
    if(j <= i)  printf("%+6.3e  ",_covarianceMatrix[i*_k->N+j]);
    else        printf("     -      ");
   printf(" |\n");
  }
 }
}
