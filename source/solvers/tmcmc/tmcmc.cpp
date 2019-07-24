#include "korali.hpp"
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

Korali::Solver::TMCMC::TMCMC()
{
 range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(range, _k->_seed++);
}

Korali::Solver::TMCMC::~TMCMC()
{
 gsl_rng_free(range);
}

void Korali::Solver::TMCMC::initialize()
{
 // Checking for accepted problem types
 std::string pName = _k->_js["Problem"]["Type"];
 bool acceptableProblem = false;
 if (pName == "Sampling")  acceptableProblem = true;
 if (pName == "Bayesian Inference")  acceptableProblem = true;
 if (pName == "Hierarchical Bayesian")  acceptableProblem = true;
 if (acceptableProblem == false) koraliError("TMCMC cannot solve problems of type: '%s'.\n", pName.c_str());

 if(_maxChainLength == 0) koraliError("Max Chain Length must be greater 0.");

 // Allocating TMCMC memory
 _covarianceMatrix.resize(_k->N*_k->N);
 _meanTheta.resize(_k->N);
 _chainCandidateParameters.resize(_k->N*_populationSize);
 _chainCandidatesLogLikelihoods.resize(_populationSize);
 _chainLeaderParameters.resize(_k->N*_populationSize);
 _chainLeadersLogLikelihoods.resize(_populationSize);
 _chainPendingFitness.resize(_populationSize);
 _currentChainStep.resize(_populationSize);
 _chainLengths.resize(_populationSize);

 if(_useLocalCovariance)
 {
  _localCovarianceMatrices.resize(_populationSize);
  for (size_t pos = 0; pos < _populationSize; ++pos)
  {
   _localCovarianceMatrices[pos].resize(_k->N*_k->N);
   for (size_t i = 0; i < _k->N; i++)
   for (size_t j = 0; j < _k->N; j++) _localCovarianceMatrices[pos][i*_k->N+j] = 0.0;
   for (size_t i = 0; i < _k->N; i++) _localCovarianceMatrices[pos][i*_k->N+i] = 1.0;
  }
 }

 // Initializing Runtime Variables
 for (size_t c = 0; c < _populationSize; c++) _currentChainStep[c] = 0;
 for (size_t c = 0; c < _populationSize; c++) _chainLengths[c] = 1 + _defaultBurnIn;
 for (size_t c = 0; c < _populationSize; c++) _chainPendingFitness[c] = false;

 // Init Generation
 _annealingExponent       = 0.0;
 _logEvidence             = 0.0;
 _coefficientOfVariation  = 0.0;
 _finishedChainsCount     = 0;
 _databaseEntryCount      = 0;
 _proposalsAcceptanceRate = 1.0;
 _selectionAcceptanceRate = 1.0;
 _acceptedSamplesCount    = _populationSize;
 _chainCount              = _populationSize;
 for (size_t c = 0; c < _chainCount; c++) _currentChainStep[c]    = 0;
 for (size_t c = 0; c < _chainCount; c++) _chainPendingFitness[c] = false;

 for (size_t c = 0; c < _populationSize; c++)
 {
   for (size_t d = 0; d < _k->N; d++)
   {
      _chainLeaderParameters[c*_k->N + d] = _chainCandidateParameters[c*_k->N + d] = _k->_variables[d]->getRandomNumber();
      _chainLeadersLogLikelihoods[c] += _k->_variables[d]->getLogDensity(_chainLeaderParameters[c*_k->N + d]);
   }
    updateDatabase(&_chainLeaderParameters[c*_k->N], _chainLeadersLogLikelihoods[c]);
 }
}

void Korali::Solver::TMCMC::runGeneration()
{
 _sampleFitnessDatabase.clear();
 _sampleParametersDatabase.clear();

 _databaseEntryCount   = 0;
 _acceptedSamplesCount = 0;
 _finishedChainsCount  = 0;

 for (size_t c = 0; c < _chainCount; c++) _currentChainStep[c] = 0;
 for (size_t c = 0; c < _chainCount; c++) _chainPendingFitness[c] = false;

 while (_finishedChainsCount < _chainCount)
 {
  for (size_t c = 0; c < _chainCount; c++) if ( (_chainPendingFitness[c] == false) && (_currentChainStep[c] < _chainLengths[c]) )
  {
  _chainPendingFitness[c] = true;
  generateCandidate(c);
  evaluateSample(c);
  }
  _k->_conduit->checkProgress();
 }

 processGeneration();
}

void Korali::Solver::TMCMC::processSample(size_t sampleId, double fitness)
{
 if(std::isfinite(fitness) == false) 
 {
   fitness = -1.0 * std::numeric_limits<double>::max();
   _chainCandidatesLogLikelihoods[sampleId] = fitness;
   koraliWarning(KORALI_DETAILED,"Sample %zu returned non finite fitness (fitness set to %e)!\n", sampleId, fitness);
 }
 else
 {
   _chainCandidatesLogLikelihoods[sampleId] = fitness;
   double ccLogPrior = _k->_problem->evaluateLogPrior(&_chainCandidateParameters[sampleId*_k->N]);
   double clLogPrior = _k->_problem->evaluateLogPrior(&_chainLeaderParameters[sampleId*_k->N]);
   double L = exp((_chainCandidatesLogLikelihoods[sampleId]-_chainLeadersLogLikelihoods[sampleId])*_annealingExponent + (ccLogPrior-clLogPrior));

   if ( L >= 1.0 || L > gsl_ran_flat(range, 0.0, 1.0) ) {
     for (size_t i = 0; i < _k->N; i++) _chainLeaderParameters[sampleId*_k->N + i] = _chainCandidateParameters[sampleId*_k->N + i];
     _chainLeadersLogLikelihoods[sampleId] = _chainCandidatesLogLikelihoods[sampleId];
     if (_currentChainStep[sampleId]+1 > _defaultBurnIn) _acceptedSamplesCount++;
   }
 }

 _currentChainStep[sampleId]++;
 _chainPendingFitness[sampleId] = false;
 if (_currentChainStep[sampleId] >  _defaultBurnIn) updateDatabase(&_chainLeaderParameters[sampleId*_k->N], _chainLeadersLogLikelihoods[sampleId]);
 if (_currentChainStep[sampleId] == _chainLengths[sampleId]) _finishedChainsCount++;
}

void Korali::Solver::TMCMC::evaluateSample(size_t c)
{
  std::vector<double> _logTransformedSample(_k->N);

  for(size_t d = 0; d<_k->N; ++d) 
    if (_k->_variables[d]->_isLogSpace == true)
        _logTransformedSample[d] = std::exp(_chainCandidateParameters[c*_k->N+d]);
    else 
        _logTransformedSample[d] = _chainCandidateParameters[c*_k->N+d];

  _k->_conduit->evaluateSample(_logTransformedSample.data(), c);
}

void Korali::Solver::TMCMC::updateDatabase(double* point, double fitness)
{
 for (size_t i = 0; i < _k->N; i++) _sampleParametersDatabase.push_back(point[i]);
 _sampleFitnessDatabase.push_back(fitness);
 _databaseEntryCount++;
}

void Korali::Solver::TMCMC::generateCandidate(size_t c)
{
 double* covariance = _useLocalCovariance ? &_localCovarianceMatrices[c][0] : &_covarianceMatrix[0];
 gsl_vector_view out_view    = gsl_vector_view_array(&_chainCandidateParameters[c*_k->N], _k->N);
 gsl_matrix_view sigma_view  = gsl_matrix_view_array(covariance, _k->N,_k->N);
 gsl_vector_view mean_view   = gsl_vector_view_array(&_chainLeaderParameters[c*_k->N], _k->N);
 gsl_ran_multivariate_gaussian(range, &mean_view.vector, &sigma_view.matrix, &out_view.vector);
}

void Korali::Solver::TMCMC::processGeneration()
{
	std::vector<double> flcp(_databaseEntryCount);
	std::vector<double> weight(_databaseEntryCount);
	std::vector<double> q(_databaseEntryCount);
	std::vector<unsigned int> nn(_databaseEntryCount);
	std::vector<size_t> sel(_databaseEntryCount);

	double fmin = 0, xmin = 0;
	minSearch(&_sampleFitnessDatabase[0], _databaseEntryCount, _annealingExponent, _targetCoefficientofVariation, xmin, fmin);

	double _prevAnnealingExponent = _annealingExponent;

	if (xmin > _prevAnnealingExponent + _maxRhoUpdate)
	{
	koraliWarning(KORALI_DETAILED, "Annealing Step larger than Max Rho Update, updating Annealing Exponent by %f (Max Rho Update). \n", _maxRhoUpdate);
	_annealingExponent      = _prevAnnealingExponent + _maxRhoUpdate;
	_coefficientOfVariation = sqrt(tmcmc_objlogp(_annealingExponent, &_sampleFitnessDatabase[0], _databaseEntryCount, _prevAnnealingExponent, _targetCoefficientofVariation)) + _targetCoefficientofVariation;
	}
	else if (xmin > _prevAnnealingExponent)
	{
	_annealingExponent      = xmin;
	_coefficientOfVariation = sqrt(fmin) + _targetCoefficientofVariation;
	}
	else
	{
	 koraliWarning(KORALI_DETAILED, "Annealing Step smaller than Min Rho Update, updating Annealing Exponent by %f (Min Rho Update). \n", _minRhoUpdate);
	_annealingExponent      = _prevAnnealingExponent + _minRhoUpdate;
	_coefficientOfVariation = sqrt(tmcmc_objlogp(_annealingExponent, &_sampleFitnessDatabase[0], _databaseEntryCount, _prevAnnealingExponent, _targetCoefficientofVariation)) + _targetCoefficientofVariation;
	}

	/* Compute weights and normalize*/

	for (size_t i = 0; i < _databaseEntryCount; i++) flcp[i] = _sampleFitnessDatabase[i]*(_annealingExponent-_prevAnnealingExponent);
	const double fjmax = gsl_stats_max(flcp.data(), 1, _databaseEntryCount);
	for (size_t i = 0; i < _databaseEntryCount; i++) weight[i] = exp( flcp[i] - fjmax );

	double sum_weight = std::accumulate(weight.begin(), weight.end(), 0.0);
	_logEvidence  += log(sum_weight) + fjmax - log(_databaseEntryCount);

	for (size_t i = 0; i < _databaseEntryCount; i++) q[i] = weight[i]/sum_weight;

	gsl_ran_multinomial(range, _databaseEntryCount, _populationSize, q.data(), nn.data());
	size_t zeroCount = 0;
	for (size_t i = 0; i < _databaseEntryCount; i++) { sel[i] = nn[i]; if ( nn[i] == 0 ) zeroCount++; }

	size_t uniqueSelections   = _databaseEntryCount - zeroCount;
	_proposalsAcceptanceRate  = (1.0*_acceptedSamplesCount)/_populationSize;
	_selectionAcceptanceRate = (1.0*uniqueSelections)/_populationSize;

	for (size_t i = 0; i < _k->N; i++)
	{
	_meanTheta[i] = 0;
	for (size_t j = 0; j < _databaseEntryCount; j++) _meanTheta[i] += _sampleParametersDatabase[j*_k->N + i]*q[j];
	}

	for (size_t i = 0; i < _k->N; i++) for (size_t j = i; j < _k->N; ++j)
	{
	double s = 0.0;
	for (size_t k = 0; k < _databaseEntryCount; ++k) s += q[k]*(_sampleParametersDatabase[k*_k->N+i]-_meanTheta[i])*(_sampleParametersDatabase[k*_k->N+j]-_meanTheta[j]);
	_covarianceMatrix[i*_k->N + j] = _covarianceMatrix[j*_k->N + i] = s*_covarianceScaling;
	}

	gsl_matrix_view sigma = gsl_matrix_view_array(&_covarianceMatrix[0], _k->N,_k->N);
	gsl_linalg_cholesky_decomp( &sigma.matrix );

    size_t leaderChainLen; 
	size_t leaderId = 0;
	for (size_t i = 0; i < _databaseEntryCount; i++) {
	 while (sel[i] != 0) {
		 for (size_t j = 0; j < _k->N ; j++) _chainLeaderParameters[leaderId*_k->N + j] = _sampleParametersDatabase[i*_k->N + j];
		 _chainLeadersLogLikelihoods[leaderId] = _sampleFitnessDatabase[i];
         
         if (sel[i] > _maxChainLength) leaderChainLen = _maxChainLength;
         else leaderChainLen = sel[i];
         
		 _chainLengths[leaderId] = leaderChainLen + _defaultBurnIn;
         sel[i] -= leaderChainLen;
		 leaderId++;
	 }
	}

	if (_useLocalCovariance) computeChainCovariances(_localCovarianceMatrices, uniqueSelections);

	_chainCount = uniqueSelections;
}


void Korali::Solver::TMCMC::computeChainCovariances(std::vector< std::vector<double> >& chain_cov, size_t newchains)
{
 // allocate space
 std::vector<size_t> nn_ind(newchains);
 std::vector<size_t> nn_count(newchains);
 std::vector<double> diam(_k->N);
 std::vector<double> chain_mean(_k->N);
 gsl_matrix* work = gsl_matrix_alloc(_k->N, _k->N);

 // find diameters
 for (size_t d = 0; d < _k->N; ++d) {
  double d_min = +1e6;
  double d_max = -1e6;
  for (size_t pos = 0; pos < _populationSize; ++pos) {
   double s = _sampleParametersDatabase[pos*_k->N+d];
   if (d_min > s) d_min = s;
   if (d_max < s) d_max = s;
  }
  diam[d] = d_max-d_min;
 }

 size_t idx, pos;
 int status = 0;
 double ds = 0.05;
 for (double scale = 0.1; scale <= 1.0; scale += ds) {
  // find neighbors in a rectangle - O(_populationSize^2)
  for (pos = 0; pos < newchains; ++pos) {
   nn_count[pos] = 0;
   double* curr = &_chainLeaderParameters[pos*_k->N];
   for (size_t i = 0; i < _populationSize; i++) {
    double* s = &_sampleParametersDatabase[i*_k->N];
    bool isInRectangle = true;
     for (size_t d = 0; d < _k->N; d++)  if (fabs(curr[d]-s[d]) > scale*diam[d]) isInRectangle = false;
     if (isInRectangle) {
      nn_ind[pos*_populationSize+nn_count[pos]] = i;
      nn_count[pos]++;
     }
   }
  }

  // compute the covariances
  for (pos = 0; pos < newchains; ++pos) {
   for (size_t d = 0; d < _k->N; ++d) {
    chain_mean[d] = 0;
    for (size_t k = 0; k < nn_count[pos]; ++k) {
     idx = nn_ind[pos*_populationSize+k];
     chain_mean[d] += _sampleParametersDatabase[idx*_k->N+d];
    }
    chain_mean[d] /= nn_count[pos];
   }

   for (size_t i = 0; i < _k->N; i++)
    for (size_t j = 0; j < _k->N; ++j) {
     double s = 0;
     for (size_t k = 0; k < nn_count[pos]; k++) {
      idx = nn_ind[pos*_populationSize+k];
      s  += (_sampleParametersDatabase[idx*_k->N+i]-chain_mean[i]) *
         (_sampleParametersDatabase[idx*_k->N+j]-chain_mean[j]);
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
   gsl_matrix_view sigma  = gsl_matrix_view_array(&chain_cov[pos][0], _k->N,_k->N);
   gsl_linalg_cholesky_decomp( &sigma.matrix );
 }

 if (status != GSL_SUCCESS) koraliError("TMCMC - GSL failed to create Chain Covariance Matrix.\n");

 // deallocate space
 gsl_matrix_free(work);
}

double Korali::Solver::TMCMC::tmcmc_objlogp(double x, const double *fj, size_t fn, double pj, double zero)
{
 std::vector<double> weight(fn);
 std::vector<double> q(fn);
 const double fjmax = gsl_stats_max(fj, 1, fn);

 for(size_t i = 0; i <fn; i++) weight[i] = exp((fj[i]-fjmax)*(x-pj));
 double sum_weight = std::accumulate(weight.begin(), weight.end(), 0.0);
 for(size_t i = 0; i < fn; i++) q[i] = weight[i]/sum_weight;

 double mean_q = gsl_stats_mean(q.data(), 1, fn);
 double std_q  = gsl_stats_sd_m(q.data(), 1, fn, mean_q);
 double cov2   = (std_q/mean_q-zero); 
 cov2 *= cov2;

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
 size_t MaxIter     = 1000;   /* Max number of search iterations */
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

 if(status == GSL_SUCCESS && s->fval >  Tol) koraliLog(KORALI_DETAILED, "Min Search converged but did not find minimum. \n");
 if(status != GSL_SUCCESS && s->fval <= Tol) koraliLog(KORALI_DETAILED, "Min Search did not converge but minimum found\n");
 if(status != GSL_SUCCESS && s->fval >  Tol) koraliLog(KORALI_DETAILED, "Min Search did not converge and did not find minimum\n");
 if(iter >= MaxIter) koraliLog(KORALI_DETAILED, "[Korali] Min Search MaxIter (%zu) reached\n", MaxIter);

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
 double clLogPrior = _k->_problem->evaluateLogPrior(&_chainLeaderParameters[c*_k->N]);
 if (clLogPrior > -INFINITY) return true;
 return false;
}

bool Korali::Solver::TMCMC::checkTermination()
{
 bool isFinished = false;

 if(_annealingExponent >= 1.0)
 {
  isFinished = true;
  koraliLog( KORALI_MINIMAL, "Annealing completed (1.0).\n");
 }

 return isFinished;

}

void Korali::Solver::TMCMC::finalize()
{

}

void Korali::Solver::TMCMC::printGeneration()
{
 koraliLog(KORALI_MINIMAL, "--------------------------------------------------------------------\n");
 koraliLog(KORALI_MINIMAL, "Generation %ld - Annealing Exponent:  %.3e.\n", _k->currentGeneration, _annealingExponent);

 koraliLog(KORALI_NORMAL, "Acceptance Rate (proposals / selections): (%.2f%% / %.2f%%)\n", 100*_proposalsAcceptanceRate, 100*_selectionAcceptanceRate);
 koraliLog(KORALI_NORMAL, "Coefficient of Variation: %.2f%%\n", 100.0*_coefficientOfVariation);

 koraliLog(KORALI_DETAILED, "Sample Mean:\n");
 for (size_t i = 0; i < _k->N; i++) koraliLogData(KORALI_DETAILED, " %s = %+6.3e\n", _k->_variables[i]->_name.c_str(), _meanTheta[i]);
 koraliLog(KORALI_DETAILED, "Sample Covariance:\n");
 for (size_t i = 0; i < _k->N; i++)
  {
	 koraliLogData(KORALI_DETAILED, "   | ");
   for (size_t j = 0; j < _k->N; j++)
    if(j <= i)  koraliLogData(KORALI_DETAILED, "%+6.3e  ",_covarianceMatrix[i*_k->N+j]);
    else        koraliLogData(KORALI_DETAILED, "     -      ");
   koraliLogData(KORALI_DETAILED, " |\n");
  }
}
