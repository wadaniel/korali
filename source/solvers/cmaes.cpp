#include "korali.h"

#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort

using namespace Korali::Solver;

CMAES::CMAES()
{
 // Initializing Gaussian Generator
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"] = "Gaussian";
 jsGaussian["Mean"] = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = new Variable();
 _gaussianGenerator->setDistribution(jsGaussian);

 _chiN = sqrt((double) _k->N) * (1. - 1./(4.*_k->N) + 1./(21.*_k->N*_k->N));

 // GSL Workspace
 gsl_eval  = gsl_vector_alloc(_k->N);
 gsl_evec  = gsl_matrix_alloc(_k->N, _k->N);
 gsl_work =  gsl_eigen_symmv_alloc(_k->N);
}


void CMAES::initialize()
{
 size_t s_max  = std::max(_sampleCount,  _viabilitySampleCount);
 size_t mu_max = std::max(_muValue, _viabilityMu);

 if(_isViabilityRegime) {
     _current_s  = _viabilitySampleCount;
     _current_mu = _viabilityMu;
 } else {
     _current_s  = _sampleCount;
     _current_mu = _muValue;
 }

 // Allocating Memory
 _samplePopulation.resize(_k->N*s_max);

 rgpc.resize(_k->N);
 rgps.resize(_k->N);
 rgdTmp.resize(_k->N);
 rgBDz.resize(_k->N);
 rgxmean.resize(_k->N);
 rgxold.resize(_k->N);
 rgxbestever.resize(_k->N);
 axisD.resize(_k->N);
 axisDtmp.resize(_k->N);
 curBestVector.resize(_k->N);

 index.resize(s_max);

 _initializedSample.resize(s_max);
 _fitnessVector.resize(s_max);

 // Init Generation
 _isFinished = false;

 C.resize(_k->N*_k->N);
 Ctmp.resize(_k->N*_k->N);
 B.resize(_k->N*_k->N);
 Btmp.resize(_k->N*_k->N);

 Z.resize(s_max*_k->N);
 BDZ.resize(s_max*_k->N);

 _transformedSamples.resize(s_max*_k->N);

 // Initailizing Mu
 _muWeights.resize(mu_max);

 _hasConstraints = (_k->_fconstraints.size() > 0);
 
 // CCMA-ES variables
 if (_hasConstraints)
 {
	if( (_normalVectorLearningRate <= 0.0) || (_normalVectorLearningRate > 1.0) ) 
        { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Normal Vector Learning Rate (%f), must be greater than 0.0 and less than 1.0\n", _normalVectorLearningRate); exit(-1); }
	if( (_globalSuccessLearningRate <= 0.0) || (_globalSuccessLearningRate > 1.0) ) { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Global Success Learning Rate (%f), must be greater than 0.0 and less than 1.0\n",  _globalSuccessLearningRate ); exit(-1); }
	if( (_targetSuccessRate <= 0.0) || (_targetSuccessRate > 1.0) )
		{ fprintf( stderr, "[Korali] CMA-ES Error: Invalid Target Success Rate (%f), must be greater than 0.0 and less than 1.0\n",  _targetSuccessRate ); exit(-1); }
	if(_covMatrixAdaptionStrength <= 0.0) { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Adaption Size (%f), must be greater than 0.0\n", _covMatrixAdaptionStrength ); exit(-1); }

	bestValidIdx     = -1;
	countcevals      = 0;
	correctionsC     = 0;
	maxnumviolations = 0;
	numviolations    = (size_t*) calloc (sizeof(size_t), _current_s);
	viabilityBounds  = (double*) calloc (sizeof(double), _k->_fconstraints.size());

	sucRates.resize(_k->_fconstraints.size());
	std::fill_n( std::begin(sucRates), _k->_fconstraints.size(), 0.5);

	viabilityImprovement  = (bool*) calloc (sizeof(bool), s_max);
	viabilityIndicator    = (bool**) calloc (sizeof(bool*), _k->_fconstraints.size());
	constraintEvaluations = (double**) calloc (sizeof(double*), _k->_fconstraints.size());

	for (size_t c = 0; c < _k->_fconstraints.size(); c++) viabilityIndicator[c]    = (bool*) calloc (sizeof(bool), s_max);
	for (size_t c = 0; c < _k->_fconstraints.size(); c++) constraintEvaluations[c] = (double*) calloc (sizeof(double), s_max);

	v = (double**) calloc (sizeof(double*), _k->_fconstraints.size());
	for (size_t i = 0; i < _k->_fconstraints.size(); i++) v[i] = (double*) calloc (sizeof(double), _k->N);

	besteverCeval = (double*) calloc (sizeof(double), _k->_fconstraints.size());
 }

 // Setting algorithm internal variables
 if (_hasConstraints) { initMuWeights(_viabilityMu); initCovCorrectionParams(); }
 else initMuWeights(_muValue);

 initCovariance();

 // Setting eigensystem evaluation Frequency
 //if( _covarianceEigenEvalFreq < 0.0) _covarianceEigenEvalFreq = 1.0/_covarianceMatrixLearningRate/((double)_k->N)/10.0;

 flgEigensysIsUptodate = true;

 countevals      = 0;
 countinfeasible = 0;
 resampled       = 0;
 
 bestEver = -std::numeric_limits<double>::max();
 psL2 = 0.0;

 /* set rgxmean */
 for (size_t i = 0; i < _k->N; ++i)
 {
	 if(_variableSettings[i].initialMean < _variableSettings[i].lowerBound || _variableSettings[i].initialMean > _variableSettings[i].upperBound) if(_k->_verbosity >= KORALI_MINIMAL)
		fprintf(stderr,"[Korali] Warning: Initial Value (%.4f) of variable \'%s\' is out of bounds (%.4f-%.4f).\n",
						_variableSettings[i].initialMean,
						_k->_variables[i]->_name.c_str(),
						_variableSettings[i].lowerBound,
						_variableSettings[i].upperBound);
	 rgxmean[i] = rgxold[i] = _variableSettings[i].initialMean;
 }

 if ( _hasConstraints ){ updateConstraints(); handleConstraints(); }
 updateDistribution();
 if ( _hasConstraints ) checkMeanAndSetRegime();
}

void CMAES::runGeneration()
{
 prepareGeneration();
 evaluateSamples();
 if ( _hasConstraints ){ updateConstraints(); handleConstraints(); }
 updateDistribution();
 if ( _hasConstraints ) checkMeanAndSetRegime();
}


void CMAES::evaluateSamples()
{
 for (size_t i = 0; i < _current_s; i++) for(size_t d = 0; d < _k->N; ++d)
  if(_k->_variables[d]->_isLogSpace == true)
	 _transformedSamples[i*_k->N+d] = std::exp(_samplePopulation[i*_k->N+d]);
	else
	 _transformedSamples[i*_k->N+d] = _samplePopulation[i*_k->N+d];

	while (_finishedSamples < _current_s)
	{
		for (size_t i = 0; i < _current_s; i++) if (_initializedSample[i] == false)
		{
			_initializedSample[i] = true;
			_k->_conduit->evaluateSample(&_transformedSamples[0], i); countevals++;
		}
		_k->_conduit->checkProgress();
	}
}

void CMAES::initMuWeights(size_t numsamplesmu)
{
 // Initializing Mu Weights
 if      (_muType == "Linear")       for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = numsamplesmu - i;
 else if (_muType == "Equal")        for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = 1.;
 else if (_muType == "Logarithmic")  for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = log(std::max( (double)numsamplesmu, 0.5*_current_s)+0.5)-log(i+1.);
 else  { fprintf( stderr, "[Korali] CMA-ES Error: Invalid setting of Mu Type (%s) (Linear, Equal, or Logarithmic accepted).",  _muType.c_str()); exit(-1); }

 // Normalize weights vector and set mueff
 double s1 = 0.0;
 double s2 = 0.0;

 for (size_t i = 0; i < numsamplesmu; i++)
 {
  s1 += _muWeights[i];
  s2 += _muWeights[i]*_muWeights[i];
 }
 _muEffective = s1*s1/s2;

 for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] /= s1;

 // Setting Mu Covariance
 //if (_muCovarianceIn < 1) _muCovariance = _muEffective;
 //else                     _muCovariance = _muCovarianceIn;
 
 /*
 // Setting Covariance Matrix Learning Rate
 double l1 = 2. / ((_k->N+1.4142)*(_k->N+1.4142));
 double l2 = (2.*_muEffective-1.) / ((_k->N+2.)*(_k->N+2.)+_muEffective);
 l2 = (l2 > 1) ? 1 : l2;
 l2 = (1./_muCovariance) * l1 + (1.-1./_muCovariance) * l2;

 _covarianceMatrixLearningRate = _covMatrixLearningRateIn;
 if (_covarianceMatrixLearningRate < 0 || _covarianceMatrixLearningRate > 1)  _covarianceMatrixLearningRate = l2;
 */

 // Setting Cumulative Covariancea
 if( (_initialCumulativeCovariance <= 0) || (_initialCumulativeCovariance > 1) ) _cumulativeCovariance = (4.0 + _muEffective/(1.0*_k->N)) / (_k->N+4.0 + 2.0*_muEffective/(1.0*_k->N));
 else _cumulativeCovariance = _initialCumulativeCovariance;

 // Setting Sigma Cumulation Factor
 _sigmaCumulationFactor = _initialSigmaCumulationFactor;
 if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor >= 1) _sigmaCumulationFactor = (_muEffective + 2.0) / (_k->N + _muEffective + 3.0);

 // Setting Damping Factor
 _dampFactor = _initialDampFactor;
 if (_dampFactor <= 0.0)
     _dampFactor = (1.0 + 2*std::max(0.0, sqrt((_muEffective-1.0)/(_k->N+1.0)) - 1))  /* basic factor */
        // * std::max(0.3, 1. - (double)_k->N / (1e-6+std::min(_termCondMaxGenerations, _termCondMaxFitnessEvaluations/_viabilitySampleCount))) /* modification for short runs */
        + _sigmaCumulationFactor; /* minor increment */

}

void CMAES::initCovCorrectionParams()
{
  // Setting beta
  _normalVectorLearningRate  = 1.0/(2.0+(double)_current_s);
  _beta = _covMatrixAdaptionStrength/(_current_s+2.);
}

void CMAES::initCovariance()
{
 
 // Setting Sigma
 _trace = 0.0;
 for (size_t i = 0; i < _k->N; ++i) _trace += _variableSettings[i].initialStdDev*_variableSettings[i].initialStdDev;
 sigma = sqrt(_trace/_k->N);

 // Setting B, C and axisD
 for (size_t i = 0; i < _k->N; ++i)
 {
  B[i*_k->N+i] = 1.0;
  C[i*_k->N+i] = axisD[i] = _variableSettings[i].initialStdDev * sqrt(_k->N / _trace);
  C[i*_k->N+i] *= C[i*_k->N+i];
 }

 minEW = *std::min_element(std::begin(axisD), std::end(axisD));
 maxEW = *std::max_element(std::begin(axisD), std::end(axisD));
 
 minEW = minEW * minEW;
 maxEW = maxEW * maxEW;

 maxdiagC=C[0]; for(size_t i=1;i<_k->N;++i) if(maxdiagC<C[i*_k->N+i]) maxdiagC=C[i*_k->N+i];
 mindiagC=C[0]; for(size_t i=1;i<_k->N;++i) if(mindiagC>C[i*_k->N+i]) mindiagC=C[i*_k->N+i];
}


void CMAES::processSample(size_t sampleId, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&_samplePopulation[sampleId*_k->N]);
 fitness = _fitnessSign * (logPrior+fitness);
 if(std::isfinite(fitness) == false)
 {
   fitness = _fitnessSign * std::numeric_limits<double>::max();
   printf("[Korali] Warning: sample %zu returned non finite fitness (set to %e)!\n", sampleId, fitness);
 }
 
 _fitnessVector[sampleId] = fitness;
 _finishedSamples++;
}


void CMAES::checkMeanAndSetRegime()
{
	if (_isViabilityRegime == false) return; /* mean already inside valid domain, no udpates */

	for (size_t c = 0; c < _k->_fconstraints.size(); c++){
		countcevals++;
		std::vector<double> sample(rgxmean);
		if ( _k->_fconstraints[c](sample) > 0.) return; /* mean violates constraint, do nothing */
	}

	/* mean inside domain, switch regime and update internal variables */
	_isViabilityRegime = false;

	for (size_t c = 0; c < _k->_fconstraints.size(); c++) { viabilityBounds[c] = 0; }
	_current_s  = _sampleCount;
	_current_mu = _muValue;

	bestEver = -std::numeric_limits<double>::max();
	initMuWeights(_current_mu);
	initCovCorrectionParams();
	initCovariance();
}


void CMAES::updateConstraints() //TODO: maybe parallelize constraint evaluations (DW)
{

 std::fill_n(numviolations, _current_s, 0);
 maxnumviolations = 0;

 for(size_t c = 0; c < _k->_fconstraints.size(); c++)
 {
  double maxviolation = 0.0;
  for(size_t i = 0; i < _current_s; ++i)
  {
    countcevals++;
    std::vector<double> sample(&_samplePopulation[i*_k->N], &_samplePopulation[(i+1)*_k->N]);

    constraintEvaluations[c][i] = _k->_fconstraints[c]( sample );

    if ( constraintEvaluations[c][i] > maxviolation ) maxviolation = constraintEvaluations[c][i];
    if ( _k->currentGeneration == 0 && _isViabilityRegime ) viabilityBounds[c] = maxviolation;

    if ( constraintEvaluations[c][i] > viabilityBounds[c] + 1e-12 ) numviolations[i]++;
    if ( numviolations[i] > maxnumviolations ) maxnumviolations = numviolations[i];

  }
 }

}


void CMAES::reEvaluateConstraints() //TODO: maybe we can parallelize constraint evaluations (DW)
{
  maxnumviolations = 0;
  for(size_t i = 0; i < _current_s; ++i) if(numviolations[i] > 0)
  {
    numviolations[i] = 0;
    for(size_t c = 0; c < _k->_fconstraints.size(); c++)
    {
      countcevals++;
      std::vector<double> sample(&_samplePopulation[i*_k->N], &_samplePopulation[(i+1)*_k->N]);

      constraintEvaluations[c][i] = _k->_fconstraints[c]( sample );

      if( constraintEvaluations[c][i] > viabilityBounds[c] + 1e-12 ) { viabilityIndicator[c][i] = true; numviolations[i]++; }
      else viabilityIndicator[c][i] = false;

    }
    if (numviolations[i] > maxnumviolations) maxnumviolations = numviolations[i];
  }
}


void CMAES::updateViabilityBoundaries()
{
 for(size_t c = 0; c < _k->_fconstraints.size(); c++)
 {
  double maxviolation = 0.0;
  for(size_t i = 0; i < _current_mu /* _current_s alternative */ ; ++i) if (constraintEvaluations[c][index[i]] > maxviolation)
    maxviolation = constraintEvaluations[c][index[i]];

  viabilityBounds[c] = std::max( 0.0, std::min(viabilityBounds[c], 0.5*(maxviolation + viabilityBounds[c])) );
 }
}


bool CMAES::isFeasible(size_t sampleIdx) const
{
 for (size_t d = 0; d < _k->N; ++d)
  if (_samplePopulation[ sampleIdx*_k->N+d ] < _variableSettings[d].lowerBound || _samplePopulation[ sampleIdx*_k->N+d ] > _variableSettings[d].upperBound) return false;
 return true;
}


void CMAES::prepareGeneration()
{

 /* calculate eigensystem */
 for (size_t d = 0; d < _k->N; ++d) Ctmp.assign(std::begin(C), std::end(C));
 updateEigensystem(Ctmp);

 for (size_t i = 0; i < _current_s; ++i)
 {
     size_t initial_infeasible = countinfeasible;
     sampleSingle(i);
     while( isFeasible( i ) == false )
     {
       countinfeasible++;
       sampleSingle(i);

       if ( _termCondMaxInfeasibleResamplingsEnabled )
       if ( (countinfeasible - initial_infeasible) > _termCondMaxInfeasibleResamplings )
       {
        if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: Exiting resampling loop (sample %zu), max resamplings (%zu) reached.\n", i, _termCondMaxInfeasibleResamplings);
        exit(-1);
       }
     }
 }

 _finishedSamples = 0;
 for (size_t i = 0; i < _current_s; i++) _initializedSample[i] = false;
}


void CMAES::sampleSingle(size_t sampleIdx)
{

  /* generate scaled random vector (D * z) */
  for (size_t d = 0; d < _k->N; ++d)
  {
   Z[sampleIdx*_k->N+d] = _gaussianGenerator->getRandomNumber();
   if (_isDiag) {
     BDZ[sampleIdx*_k->N+d] = axisD[d] * Z[sampleIdx*_k->N+d];
     _samplePopulation[sampleIdx * _k->N + d] = rgxmean[d] + sigma * BDZ[sampleIdx*_k->N+d];
   }
   else rgdTmp[d] = axisD[d] * Z[sampleIdx*_k->N+d];
  }

  if (!_isDiag)
   for (size_t d = 0; d < _k->N; ++d) {
    BDZ[sampleIdx*_k->N+d] = 0.0;
    for (size_t e = 0; e < _k->N; ++e) BDZ[sampleIdx*_k->N+d] += B[d*_k->N+e] * rgdTmp[e];
    _samplePopulation[sampleIdx * _k->N + d] = rgxmean[d] + sigma * BDZ[sampleIdx*_k->N+d];
  }
}


void CMAES::updateDistribution()
{

 /* Generate index */
 sort_index(_fitnessVector, index, _current_s);

 if( (_hasConstraints == false) ||  _isViabilityRegime )
  bestValidIdx = 0;
 else
 {
  bestValidIdx = -1;
  for (size_t i = 0; i < _current_s; i++) if(numviolations[index[i]] == 0) bestValidIdx = index[i];
  if(_k->_verbosity >= KORALI_DETAILED && bestValidIdx == -1) 
    { printf("[Korali] Warning: all samples violate constraints, no updates taking place.\n"); return; }
 }

 /* update function value history */
 prevFunctionValue = currentFunctionValue;

 /* update current best */
 currentFunctionValue = _fitnessVector[bestValidIdx];
 for (size_t d = 0; d < _k->N; ++d) curBestVector[d] = _samplePopulation[bestValidIdx*_k->N + d];

 /* update xbestever */
 if ( currentFunctionValue > bestEver )
 {
  prevBest = bestEver;
  bestEver = currentFunctionValue;
  for (size_t d = 0; d < _k->N; ++d) rgxbestever[d]   = curBestVector[d];
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) besteverCeval[c] = constraintEvaluations[c][bestValidIdx];
 }

 /* set weights */
 for (size_t d = 0; d < _k->N; ++d) {
   rgxold[d] = rgxmean[d];
   rgxmean[d] = 0.;
   for (size_t i = 0; i < _current_mu; ++i)
     rgxmean[d] += _muWeights[i] * _samplePopulation[index[i]*_k->N + d];

   rgBDz[d] = (rgxmean[d] - rgxold[d])/sigma;
 }

 /* calculate z := D^(-1) * B^(T) * rgBDz into rgdTmp */
 for (size_t d = 0; d < _k->N; ++d) {
  double sum = 0.0;
  if (_isDiag) sum = rgBDz[d];
  else for (size_t e = 0; e < _k->N; ++e) sum += B[e*_k->N+d] * rgBDz[e]; /* B^(T) * rgBDz ( iterating B[e][d] = B^(T) ) */

  rgdTmp[d] = sum / axisD[d]; /* D^(-1) * B^(T) * rgBDz */
 }

 psL2 = 0.0;

 /* cumulation for sigma (ps) using B*z */
 for (size_t d = 0; d < _k->N; ++d) {
    double sum = 0.0;
    if (_isDiag) sum = rgdTmp[d];
    else for (size_t e = 0; e < _k->N; ++e) sum += B[d*_k->N+e] * rgdTmp[e];

    rgps[d] = (1. - _sigmaCumulationFactor) * rgps[d] + sqrt(_sigmaCumulationFactor * (2. - _sigmaCumulationFactor) * _muEffective) * sum;

    /* calculate norm(ps)^2 */
    psL2 += rgps[d] * rgps[d];
 }

 int hsig = sqrt(psL2) / sqrt(1. - pow(1.-_sigmaCumulationFactor, 2.0*(1.0+_k->currentGeneration))) / _chiN  < 1.4 + 2./(_k->N+1);

 /* cumulation for covariance matrix (pc) using B*D*z~_k->N(0,C) */
 for (size_t d = 0; d < _k->N; ++d)
   rgpc[d] = (1. - _cumulativeCovariance) * rgpc[d] + hsig * sqrt( _cumulativeCovariance * (2. - _cumulativeCovariance) * _muEffective ) * rgBDz[d];

 /* update of C  */
 adaptC(hsig);

 double sigma_upper = sqrt(_trace/_k->N);
 // update sigma & viability boundaries
 if( _hasConstraints && (_isViabilityRegime == false) )
 {
   updateViabilityBoundaries();

   if ( prevBest == bestEver ) _globalSucRate = (1-_globalSuccessLearningRate)*_globalSucRate;
   else _globalSucRate = (1-_globalSuccessLearningRate)*_globalSucRate + _globalSuccessLearningRate;
   //else for(size_t c = 0; c < _k->_fconstraints.size(); c++) if( sucRates[c] < 0.5 ) { _globalSucRate = (1-_globalSuccessLearningRate)*_globalSucRate; break; }
   sigma *= exp(1.0/_dampFactor*(_globalSucRate-(_targetSuccessRate/(1.0-_targetSuccessRate))*(1-_globalSucRate)));
 }
 else
 {
   // sigma *= exp(((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor) (orig, alternative)
   sigma *= exp(std::min(1.0, ((sqrt(psL2)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor));
 }

 if(_k->_verbosity >= KORALI_DETAILED && sigma > sigma_upper) 
     printf("[Korali] Warning: Sigma exceeding inital value of sigma (%f > %f), increase Initial Standard Deviation of variables.\n", sigma, sigma_upper);
 
 /* upper bound for sigma */
 if( _isSigmaBounded && (sigma > sigma_upper) ) sigma = sigma_upper;

 /* numerical error management */

 //treat maximal standard deviations
 //TODO

 //treat minimal standard deviations
 for (size_t d = 0; d < _k->N; ++d) if (sigma * sqrt(C[d*_k->N+d]) < _variableSettings[d].minStdDevChange)
 {
   sigma = (_variableSettings[d].minStdDevChange)/sqrt(C[d*_k->N+d]) * exp(0.05+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) fprintf(stderr, "[Korali] Warning: Sigma increased due to minimal standard deviation.\n");
 }

 //too low coordinate axis deviations
 //TODO

 //treat numerical precision provblems
 //TODO

 /* Test if function values are identical, escape flat fitness */
 if (currentFunctionValue == _fitnessVector[index[(int)_current_mu]]) {
   sigma *= exp(0.2+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) {
     fprintf(stderr, "[Korali] Warning: Sigma increased due to equal function values.\n");
   }
 }

}

void CMAES::adaptC(int hsig)
{

 //if (_covarianceMatrixLearningRate != 0.0)
 if (true)
 {
  /* definitions for speeding up inner-most loop */
  //double ccov1  = std::min(_covarianceMatrixLearningRate * (1./_muCovariance) * (_isDiag ? (_k->N+1.5) / 3. : 1.), 1.); (orig, alternative)
  //double ccovmu = std::min(_covarianceMatrixLearningRate * (1-1./_muCovariance) * (_isDiag ? (_k->N+1.5) / 3. : 1.), 1.-ccov1); (orig, alternative)
  double ccov1  = 2.0 / (std::pow(_k->N+1.3,2)+_muEffective);
  double ccovmu = std::min(1.0-ccov1,  2.0 * (_muEffective-2+1/_muEffective) / (std::pow(_k->N+2.0,2)+_muEffective));
  double sigmasquare = sigma * sigma;

  flgEigensysIsUptodate = false;

  /* update covariance matrix */
  for (size_t d = 0; d < _k->N; ++d)
   for (size_t e = _isDiag ? d : 0; e <= d; ++e) {
     C[d*_k->N+e] = (1 - ccov1 - ccovmu) * C[d*_k->N+e] + ccov1 * (rgpc[d] * rgpc[e] + (1-hsig)*ccov1*_cumulativeCovariance*(2.-_cumulativeCovariance) * C[d*_k->N+e]);
     for (size_t k = 0; k < _current_mu; ++k) 
         C[d*_k->N+e] += ccovmu * _muWeights[k] * (_samplePopulation[index[k]*_k->N + d] - rgxold[d]) * (_samplePopulation[index[k]*_k->N + e] - rgxold[e]) / sigmasquare;
     if (e < d) C[e*_k->N+d] = C[d*_k->N+e];
   }

  /* update maximal and minimal diagonal value */
  maxdiagC = mindiagC = C[0];
  for (size_t d = 1; d < _k->N; ++d) {
  if (maxdiagC < C[d*_k->N+d]) maxdiagC = C[d*_k->N+d];
  else if (mindiagC > C[d*_k->N+d])  mindiagC = C[d*_k->N+d];
  }
 } /* if ccov... */
}

void CMAES::handleConstraints()
{
 size_t initial_resampled = resampled;
 size_t initial_corrections = correctionsC;

 while( maxnumviolations > 0 )
 {
  for (size_t i = 0; i < _k->N; i++) Ctmp.assign(std::begin(C), std::end(C));

  for(size_t i = 0; i < _current_s; ++i) if (numviolations[i] > 0)
  {
    //update v
    for( size_t c = 0; c < _k->_fconstraints.size(); c++ )
      if ( viabilityIndicator[c][i] == true )
      {
        correctionsC++;

        double v2 = 0;
        for( size_t d = 0; d < _k->N; ++d)
        {
            v[c][d] = (1.0-_normalVectorLearningRate)*v[c][d]+_normalVectorLearningRate*BDZ[i*_k->N+d];
            v2 += v[c][d]*v[c][d];
        }
        for( size_t d = 0; d < _k->N; ++d)
          for( size_t e = 0; e < _k->N; ++e)
            Ctmp[d*_k->N+e] = Ctmp[d*_k->N+e] - ((_beta * _beta * v[c][d]*v[c][e])/(v2*numviolations[i]*numviolations[i]));

        flgEigensysIsUptodate = false;
        sucRates[c] = (1.0-_globalSuccessLearningRate)*sucRates[c];
      }
      else
      {
        sucRates[c] = (1.0-_globalSuccessLearningRate)*sucRates[c]+_globalSuccessLearningRate/_current_s;
      }
   }

  updateEigensystem(Ctmp);

  /* in original some stopping criterion (TOLX) */
  // TODO

  //resample invalid points
  for(size_t i = 0; i < _current_s; ++i) if(numviolations[i] > 0)
  {
    do
    {
     resampled++;
     sampleSingle(i);

     if(_termCondMaxInfeasibleResamplingsEnabled)
     if(resampled-initial_resampled > _termCondMaxInfeasibleResamplings)
     {
        if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: Exiting resampling loop, max resamplings (%zu) reached.\n", _termCondMaxInfeasibleResamplings);
        reEvaluateConstraints();

        return;
     }

    }
    while( isFeasible(i) == false );
  }

  reEvaluateConstraints();

  if(correctionsC - initial_corrections > _maxCovMatrixCorrections)
  {
    if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: Exiting adaption loop, max adaptions (%zu) reached.\n", _maxCovMatrixCorrections);
    return;
  }

 }//while maxnumviolations > 0

}

bool CMAES::checkTermination()
{

 if ( _termCondMinFitnessEnabled && (_isViabilityRegime == false) && (_k->currentGeneration > 1) && (bestEver >= _termCondMinFitness) )
 {
  _isFinished = true;
  printf("Min fitness value (%+6.3e) > (%+6.3e)",  bestEver, _termCondMinFitness);
 }
 
 if ( _termCondMaxFitnessEnabled && (_isViabilityRegime == false) && (_k->currentGeneration > 1) && (bestEver >= _termCondMaxFitness) )
 {
  _isFinished = true;
  printf("Max fitness value (%+6.3e) > (%+6.3e)",  bestEver, _termCondMaxFitness);
 }

 double range = fabs(currentFunctionValue - prevFunctionValue);
 if ( _termCondMinFitnessDiffThresholdEnabled && (_k->currentGeneration > 1) && (range <= _termCondMinFitnessDiffThreshold) )
 {
  _isFinished = true;
  printf("Function value differences (%+6.3e) < (%+6.3e)\n",  range, _termCondMinFitnessDiffThreshold);
 }

 size_t idx;
 
 if ( _termCondMinStandardDeviationEnabled )
 {
  size_t cTemp = 0;
  for(idx = 0; idx <_k->N; ++idx )
   cTemp += (sigma * sqrt(C[idx*_k->N+idx]) < _termCondMinStandardDeviation * _variableSettings[idx].initialStdDev) ? 1 : 0;
  
  if (cTemp == _k->N) {
   _isFinished = true;
   printf("Object variable changes < %+6.3e", _termCondMinStandardDeviation * _variableSettings[idx].initialStdDev);
  }

  for(idx = 0; idx <_k->N; ++idx )
   if ( _termCondMaxStandardDeviationEnabled && (sigma * sqrt(C[idx*_k->N+idx]) > _termCondMaxStandardDeviation * _variableSettings[idx].initialStdDev) )
   {
    _isFinished = true;
    printf("Standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n", _termCondMaxStandardDeviation * _variableSettings[idx].initialStdDev);
    break;
   }
 }

 if ( _termCondMaxCovMatrixConditionEnabled && (maxEW >= minEW * _termCondMaxCovMatrixCondition) )
 {
   _isFinished = true;
   printf("Maximal condition number %7.2e reached. maxEW=%7.2e, minEig=%7.2e, maxdiagC=%7.2e, mindiagC=%7.2e\n",
                                _termCondMaxCovMatrixCondition, maxEW, minEW, maxdiagC, mindiagC);
 }

 double fac;
 size_t iAchse = 0;
 size_t iKoo = 0;
 /* Component of rgxmean is not changed anymore */
 if( _termCondMinStandardDeviationStepFactorEnabled)
 if (!_isDiag )
 {
    for (iAchse = 0; iAchse < _k->N; ++iAchse)
    {
    fac = _termCondMinStandardDeviationStepFactor * sigma * axisD[iAchse];
    for (iKoo = 0; iKoo < _k->N; ++iKoo){
      if (rgxmean[iKoo] != rgxmean[iKoo] + fac * B[iKoo*_k->N+iAchse])
      break;
    }
    if (iKoo == _k->N)
    {
      _isFinished = true;
      printf("Standard deviation %f*%7.2e in principal axis %ld without effect.", _termCondMinStandardDeviationStepFactor, sigma*axisD[iAchse], iAchse);
      break;
    }
  }
 }

 /* Component of rgxmean is not changed anymore */
 if( _termCondMinStandardDeviationStepFactorEnabled )
 for (iKoo = 0; iKoo < _k->N; ++iKoo)
 {
  if (rgxmean[iKoo] == rgxmean[iKoo] + _termCondMinStandardDeviationStepFactor*sigma*sqrt(C[iKoo*_k->N+iKoo]) )
  {
   _isFinished = true;
   printf("Standard deviation %f*%7.2e in coordinate %ld without effect.", _termCondMinStandardDeviationStepFactor, sigma*sqrt(C[iKoo*_k->N+iKoo]), iKoo);
   break;
  }

 }

 return _isFinished;
}


void CMAES::updateEigensystem(std::vector<double>& M, int flgforce)
{
 if(flgforce == 0 && flgEigensysIsUptodate) return;
 /* if(_k->currentGeneration % _covarianceEigenEvalFreq == 0) return; */

 eigen(_k->N, M, axisDtmp, Btmp);
 
 /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
 double minEWtmp = *std::min_element(std::begin(axisDtmp), std::end(axisDtmp));
 double maxEWtmp = *std::max_element(std::begin(axisDtmp), std::end(axisDtmp));

 if (minEWtmp <= 0.0) 
 { if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: Min Eigenvalue smaller or equal 0.0 (%+6.3e) after Eigen decomp (no update possible).\n", minEWtmp ); return; }

 for (size_t d = 0; d < _k->N; ++d) 
 {
     axisDtmp[d] = sqrt(axisDtmp[d]); 
     if (std::isfinite(axisDtmp[d]) == false)
     {
       if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: Could not calculate root of Eigenvalue (%+6.3e) after Eigen decomp (no update possible).\n", axisDtmp[d] ); 
       return; 
     }
    for (size_t e = 0; e < _k->N; ++e) if (std::isfinite(B[d*_k->N+e]) == false)
    {
       if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: Non finite value detected in B (no update possible).\n"); 
       return;
    }
 }
 
 /* write back */
 for (size_t d = 0; d < _k->N; ++d) axisD[d] = axisDtmp[d];
 B.assign(std::begin(Btmp), std::end(Btmp));

 minEW = minEWtmp;
 maxEW = maxEWtmp;

 flgEigensysIsUptodate = true;
}


/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/


void CMAES::eigen(size_t size, std::vector<double>& M,  std::vector<double>& diag, std::vector<double>& Q) const
{
 double* data = (double*) malloc (sizeof(double) * size * size);

 for (size_t i = 0; i <  size; i++)
 for (size_t j = 0; j <= i; j++)
 {
  data[i*size + j] = M[i*_k->N+j];
  data[j*size + i] = M[i*_k->N+j];
 }

 gsl_matrix_view m = gsl_matrix_view_array (data, size, size);

 gsl_eigen_symmv (&m.matrix, gsl_eval, gsl_evec, gsl_work);
 gsl_eigen_symmv_sort (gsl_eval, gsl_evec, GSL_EIGEN_SORT_ABS_ASC);

 for (size_t i = 0; i < size; i++)
 {
  gsl_vector_view gsl_evec_i = gsl_matrix_column (gsl_evec, i);
  for (size_t j = 0; j < size; j++) Q[j*_k->N+i] = gsl_vector_get (&gsl_evec_i.vector, j);
 }

 for (size_t i = 0; i < size; i++) diag[i] = gsl_vector_get (gsl_eval, i);

 free(data);
}


void CMAES::sort_index(const std::vector<double>& vec, std::vector<size_t>& index, size_t n) const
{
  // initialize original index locations
  std::iota(std::begin(index), std::end(index), (size_t) 0);

  // sort indexes based on comparing values in v
  std::sort(std::begin(index), std::end(index), [vec](size_t i1, size_t i2) {return vec[i1] > vec[i2];} );

}


void CMAES::printGeneration()
{
 if (_k->currentGeneration % _terminalOutputFrequency != 0) return;

 if ( _hasConstraints && _isViabilityRegime && _k->_verbosity >= KORALI_NORMAL)
 {
   printf("\n[Korali] CCMA-ES searching start (MeanX violates constraints) .. \n");
   printf("[Korali] Viability Bounds:\n");
   for (size_t c = 0; c < _k->_fconstraints.size(); c++)  printf("         %s = (%+6.3e)\n", _k->_variables[c]->_name.c_str(), viabilityBounds[c]);
   printf("\n");
 }

 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Sigma:                        %+6.3e\n", sigma);
  printf("[Korali] Current Function Value: Max = %+6.3e - Best = %+6.3e\n", currentFunctionValue, bestEver);
  printf("[Korali] Diagonal Covariance:    Min = %+6.3e -  Max = %+6.3e\n", mindiagC, maxdiagC);
  printf("[Korali] Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", minEW, maxEW);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), rgxmean[d], rgxbestever[d]);

  printf("[Korali] Constraint Evaluation at Current Function Value:\n");
  if ( _hasConstraints )
  if ( bestValidIdx >= 0 )
  {
      for (size_t c = 0; c < _k->_fconstraints.size(); c++) printf("         ( %+6.3e )\n", constraintEvaluations[c][bestValidIdx]);
  }
  else
  {
      for (size_t c = 0; c < _k->_fconstraints.size(); c++) printf("         ( %+6.3e )\n", constraintEvaluations[c][0]);
  }

  printf("[Korali] Covariance Matrix:\n");
  for (size_t d = 0; d < _k->N; d++)
  {
   for (size_t e = 0; e <= d; e++) printf("   %+6.3e  ",C[d*_k->N+e]);
   printf("\n");
  }

  printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
  printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
  if ( _hasConstraints ) { printf("[Korali] Number of Constraint Evaluations: %zu\n", countcevals); printf("[Korali] Number of Matrix Corrections: %zu\n", correctionsC ); }
 }

 if (_k->_verbosity >= KORALI_NORMAL)
   printf("--------------------------------------------------------------------\n");
}

void CMAES::finalize()
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] CMA-ES Finished\n");
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), bestEver);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t d = 0; d < _k->N; ++d) printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), rgxbestever[d]);
    if ( _hasConstraints ) { printf("[Korali] Constraint Evaluation at Optimum:\n"); for (size_t c = 0; c < _k->_fconstraints.size(); c++) printf("         ( %+6.3e )\n", besteverCeval[c]); }
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
    printf("--------------------------------------------------------------------\n");
 }
}
