#include "korali.hpp"

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

 _uniformGenerator = nullptr;
}

CMAES::~CMAES()
{
  delete _gaussianGenerator;
}


void CMAES::initialize()
{
 // Checking for accepted problem types
 std::string pName = _k->_js["Problem"]["Type"];
 bool acceptableProblem = false;
 if (pName == "Optimization")  acceptableProblem = true;
 if (pName == "Bayesian Inference")  acceptableProblem = true;
 if (acceptableProblem == false) koraliError("CMAES cannot solve problems of type: '%s'.\n", pName.c_str());

 _chiN = sqrt((double) _k->N) * (1. - 1./(4.*_k->N) + 1./(21.*_k->N*_k->N));
 _chiS = sqrt((double) _k->N) * (1. - 1./(4.*_k->N) + 1./(21.*_k->N*_k->N));

 // Allocating Memory
 _samplePopulation.resize(_k->N*_sampleCount);

 _evolutionPath.resize(_k->N);
 _conjugateEvolutionPath.resize(_k->N);
 _BDZtmp.resize(_k->N);
 _y.resize(_k->N);
 _mean.resize(_k->N);
 _previousMean.resize(_k->N);
 _bestEverSample.resize(_k->N);
 _axisD.resize(_k->N);
 _axisDtmp.resize(_k->N);
 _currentBestSample.resize(_k->N);

 _sortingIndex.resize(_sampleCount);

 _isInitializedSample.resize(_sampleCount);
 _fitnessVector.resize(_sampleCount);

 _C.resize(_k->N*_k->N);
 _Ctmp.resize(_k->N*_k->N);
 _B.resize(_k->N*_k->N);
 _Btmp.resize(_k->N*_k->N);

 _Z.resize(_sampleCount*_k->N);
 _BDZ.resize(_sampleCount*_k->N);

 _granularity.resize(_k->N);
 _maskingMatrix.resize(_k->N);
 _maskingMatrixSigma.resize(_k->N);
 _maskingMatrixEntries = 0;

 // Initailizing Mu
 _muWeights.resize(_muValue);
 
 // Setting algorithm internal variables
 initMuWeights(_muValue);

 initCovariance();

 _isEigenSystemUpdate = true;
 _infeasibleSampleCount = 0;
 
 _bestEverValue = -std::numeric_limits<double>::max();
 _conjugateEvolutionPathL2Norm = 0.0;

 /* set _mean */
 for (size_t i = 0; i < _k->N; ++i)
 {
   if(_variableSettings[i].initialMean < _variableSettings[i].lowerBound || _variableSettings[i].initialMean > _variableSettings[i].upperBound)
   koraliWarning(KORALI_MINIMAL, "Initial Value (%.4f) of variable \'%s\' is out of bounds (%.4f-%.4f).\n",
            _variableSettings[i].initialMean,
            _k->_variables[i]->_name.c_str(),
            _variableSettings[i].lowerBound,
            _variableSettings[i].upperBound);
   _mean[i] = _previousMean[i] = _variableSettings[i].initialMean;
 }
 
 /* set _granularity for discrete variables */
 size_t numDiscretes = 0;
 for (size_t i = 0; i < _k->N; ++i)
 {
   if( (_variableSettings[i].discrete == true) && _variableSettings[i].granularity == 0.0) 
       koraliError("Granularity not set for discrete variable \'%s\'.\n", _k->_variables[i]->_name.c_str());
   if (_variableSettings[i].discrete == true) numDiscretes++;
    _granularity[i] = _variableSettings[i].granularity;
 }

 _hasDiscreteVariables = (numDiscretes > 0);
 
 if (_hasDiscreteVariables)
 {
   auto jsUniform = nlohmann::json();
   jsUniform["Type"] = "Uniform";
   jsUniform["Minimum"] = 0.0;
   jsUniform["Maximum"] = 1.0;
   jsUniform["Seed"] = _k->_seed++;
   _uniformGenerator = std::make_shared<Variable>();
   _uniformGenerator->setDistribution(jsUniform);
 }
 
 _numDiscreteMutations = 0; // updated each generation
 _discreteMutations.resize(_k->N*_sampleCount);
 std::fill( std::begin(_discreteMutations), std::end(_discreteMutations), 0.0);


}


void CMAES::runGeneration()
{
 prepareGeneration();
 evaluateSamples();
 updateDistribution();
}


void CMAES::evaluateSamples()
{
  while (_finishedSampleCount < _sampleCount)
  {
    for (size_t i = 0; i < _sampleCount; i++) if (_isInitializedSample[i] == false)
    {
     std::vector<double> _logTransformedSample(_k->N);

     for(size_t d = 0; d<_k->N; ++d)
       if (_k->_variables[d]->_isLogSpace == true)
           _logTransformedSample[d] = std::exp(_samplePopulation[i*_k->N+d]);
       else
           _logTransformedSample[d] = _samplePopulation[i*_k->N+d];

      _isInitializedSample[i] = true;
      _k->_conduit->evaluateSample(_logTransformedSample.data(), i);
    }
    _k->_conduit->checkProgress();
  }
}

void CMAES::initMuWeights(size_t numsamplesmu)
{
 // Initializing Mu Weights
 if      (_muType == "Linear")       for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = numsamplesmu - i;
 else if (_muType == "Equal")        for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = 1.;
 else if (_muType == "Logarithmic")  for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = log(std::max( (double)numsamplesmu, 0.5*_sampleCount)+0.5)-log(i+1.);
 else  koraliError("CMA-ES - Invalid setting of Mu Type (%s) (Linear, Equal, or Logarithmic accepted).",  _muType.c_str());

 // Normalize weights vector and set mueff
 double s1 = 0.0;
 double s2 = 0.0;

 for (size_t i = 0; i < numsamplesmu; i++)
 {
  s1 += _muWeights[i];
  s2 += _muWeights[i]*_muWeights[i];
 }
 _effectiveMu = s1*s1/s2;

 for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] /= s1;
 
 // Setting Cumulative Covariancea
 if( (_initialCumulativeCovariance <= 0) || (_initialCumulativeCovariance > 1) ) _cumulativeCovariance = (4.0 + _effectiveMu/(1.0*_k->N)) / (_k->N+4.0 + 2.0*_effectiveMu/(1.0*_k->N));
 else _cumulativeCovariance = _initialCumulativeCovariance;

 // Setting Sigma Cumulation Factor
 _sigmaCumulationFactor = _initialSigmaCumulationFactor;
 if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor >= 1) 
   _sigmaCumulationFactor = (_effectiveMu + 2.0) / (_k->N + _effectiveMu + 3.0);

 // Setting Damping Factor
 _dampFactor = _initialDampFactor;
 if (_dampFactor <= 0.0)
     _dampFactor = (1.0 + 2*std::max(0.0, sqrt((_effectiveMu-1.0)/(_k->N+1.0)) - 1)) + _sigmaCumulationFactor;

}


void CMAES::initCovariance()
{

 for(size_t d = 0; d < _k->N; ++d)
 {
    if ( _variableSettings[d].initialStdDev<0 )
    if ( std::isfinite(_variableSettings[d].lowerBound) && std::isfinite(_variableSettings[d].upperBound ) )
            _variableSettings[d].initialStdDev = 0.3*(_variableSettings[d].upperBound-_variableSettings[d].lowerBound);
    else 
        koraliError("Lower/Upper Bound not defined, and Initial Standard Dev not defined for variable '%s'\n", _k->_variables[d]->_name.c_str());
 }
 
 // Setting Sigma
 _trace = 0.0;
 for (size_t i = 0; i < _k->N; ++i) _trace += _variableSettings[i].initialStdDev*_variableSettings[i].initialStdDev;
 _sigma = sqrt(_trace/_k->N);

 // Setting B, C and _axisD
 for (size_t i = 0; i < _k->N; ++i)
 {
  _B[i*_k->N+i] = 1.0;
  _C[i*_k->N+i] = _axisD[i] = _variableSettings[i].initialStdDev * sqrt(_k->N / _trace);
  _C[i*_k->N+i] *= _C[i*_k->N+i];
 }

 _minCovarianceEigenvalue = *std::min_element(std::begin(_axisD), std::end(_axisD));
 _maxCovarianceEigenvalue = *std::max_element(std::begin(_axisD), std::end(_axisD));
 
 _minCovarianceEigenvalue = _minCovarianceEigenvalue * _minCovarianceEigenvalue;
 _maxCovarianceEigenvalue = _maxCovarianceEigenvalue * _maxCovarianceEigenvalue;

 _maxDiagCElement=_C[0]; for(size_t i=1;i<_k->N;++i) if(_maxDiagCElement<_C[i*_k->N+i]) _maxDiagCElement=_C[i*_k->N+i];
 _minDiagCElement=_C[0]; for(size_t i=1;i<_k->N;++i) if(_minDiagCElement>_C[i*_k->N+i]) _minDiagCElement=_C[i*_k->N+i];
}


void CMAES::processSample(size_t sampleId, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&_samplePopulation[sampleId*_k->N]);
 fitness = logPrior+fitness;
 if(std::isfinite(fitness) == false)  koraliError("Sample %zu returned non finite fitness (set to %e)!\n", sampleId, fitness);
 _fitnessVector[sampleId] = fitness;
 _finishedSampleCount++;
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
 for (size_t d = 0; d < _k->N; ++d) _Ctmp.assign(std::begin(_C), std::end(_C));
 updateEigensystem(_Ctmp);

 for (size_t i = 0; i < _sampleCount; ++i)
 {
     size_t initial_infeasible = _infeasibleSampleCount;
     sampleSingle(i);
     while( isFeasible( i ) == false )
     {
       _infeasibleSampleCount++;
       sampleSingle(i);

       if ( _termCondMaxInfeasibleResamplingsEnabled )
       if ( (_infeasibleSampleCount - initial_infeasible) > _termCondMaxInfeasibleResamplings )
        koraliError("Exiting resampling loop (sample %zu), max resamplings (%zu) reached.\n", i, _termCondMaxInfeasibleResamplings);
     }
 }

 _finishedSampleCount = 0;
 for (size_t i = 0; i < _sampleCount; i++) _isInitializedSample[i] = false;
}


void CMAES::sampleSingle(size_t sampleIdx)
{
  for (size_t d = 0; d < _k->N; ++d)
  {
   _Z[sampleIdx*_k->N+d] = _gaussianGenerator->getRandomNumber();
   if (_isDiag) {
     _BDZ[sampleIdx*_k->N+d] = _axisD[d] * _Z[sampleIdx*_k->N+d];
     _samplePopulation[sampleIdx * _k->N + d] = _mean[d] + _sigma * _BDZ[sampleIdx*_k->N+d];
   }
   else _BDZtmp[d] = _axisD[d] * _Z[sampleIdx*_k->N+d];
  }

  if (!_isDiag)
   for (size_t d = 0; d < _k->N; ++d) {
    _BDZ[sampleIdx*_k->N+d] = 0.0;
    for (size_t e = 0; e < _k->N; ++e) _BDZ[sampleIdx*_k->N+d] += _B[d*_k->N+e] * _BDZtmp[e];
    _samplePopulation[sampleIdx * _k->N + d] = _mean[d] + _sigma * _BDZ[sampleIdx*_k->N+d];
  }

  if(_hasDiscreteVariables)
  {
    if ( (sampleIdx+1) < _numDiscreteMutations )
    {
      double p_geom = std::pow(0.7, 1.0/_maskingMatrixEntries);
      size_t select = std::floor(_uniformGenerator->getRandomNumber() * _maskingMatrixEntries);

      for(size_t d = 0; d < _k->N; ++d) if( (_maskingMatrix[d] == 1.0) && (select-- == 0) )
      {
        double dmutation = 1.0;
        while(  _uniformGenerator->getRandomNumber() > p_geom ) dmutation += 1.0;
        dmutation *= _granularity[d];

        if( _uniformGenerator->getRandomNumber() > 0.5 ) dmutation*=-1.0;
        _discreteMutations[sampleIdx*_k->N+d] = dmutation; 
        _samplePopulation[sampleIdx*_k->N+d] += dmutation;
      }
    }
    else if ( (sampleIdx+1) == _numDiscreteMutations )
    {
      for(size_t d = 0; d < _k->N; ++d) if( _granularity[d] != 0.0 )
      {
        double dmutation = std::round(_bestEverSample[d]/_granularity[d]) * _granularity[d] - _samplePopulation[sampleIdx*_k->N+d];
        _discreteMutations[sampleIdx*_k->N+d] = dmutation;
        _samplePopulation[sampleIdx*_k->N+d] += dmutation;
      }
    }
  }

}


void CMAES::updateDistribution()
{

 /* Generate _sortingIndex */
 sort_index(_fitnessVector, _sortingIndex, _sampleCount);

 /* update function value history */
 _previousBestValue = _currentBestValue;

 /* update current best */
 _currentBestValue = _fitnessVector[0];
 for (size_t d = 0; d < _k->N; ++d) _currentBestSample[d] = _samplePopulation[d];

 /* update xbestever */
 if ( _currentBestValue > _bestEverValue )
 {
  _previousBestEverValue = _bestEverValue;
  _bestEverValue = _currentBestValue;
  for (size_t d = 0; d < _k->N; ++d) _bestEverSample[d] = _currentBestSample[d];
 }

 /* set weights */
 for (size_t d = 0; d < _k->N; ++d) {
   _previousMean[d] = _mean[d];
   _mean[d] = 0.;
   for (size_t i = 0; i < _muValue; ++i) _mean[d] += _muWeights[i] * _samplePopulation[_sortingIndex[i]*_k->N + d];

   _y[d] = (_mean[d] - _previousMean[d])/_sigma;
 }

 /* calculate z := D^(-1) * B^(T) * _y into _BDZtmp */
 for (size_t d = 0; d < _k->N; ++d) {
  double sum = 0.0;
  if (_isDiag) sum = _y[d];
  else for (size_t e = 0; e < _k->N; ++e) sum += _B[e*_k->N+d] * _y[e]; /* B^(T) * _y ( iterating B[e][d] = B^(T) ) */

  _BDZtmp[d] = sum / _axisD[d]; /* D^(-1) * B^(T) * _y */
 }

 _conjugateEvolutionPathL2Norm = 0.0;

 /* cumulation for conjugate evolution path  */
 for (size_t d = 0; d < _k->N; ++d) {
    double sum = 0.0;
    if (_isDiag) sum = _BDZtmp[d];
    else for (size_t e = 0; e < _k->N; ++e) sum += _B[d*_k->N+e] * _BDZtmp[e];

    _conjugateEvolutionPath[d] = (1. - _sigmaCumulationFactor) * _conjugateEvolutionPath[d] + sqrt(_sigmaCumulationFactor * (2. - _sigmaCumulationFactor) * _effectiveMu) * sum;

    /* calculate norm(ps)^2 */
    _conjugateEvolutionPathL2Norm += _conjugateEvolutionPath[d] * _conjugateEvolutionPath[d];
 }

 int hsig = sqrt(_conjugateEvolutionPathL2Norm) / sqrt(1. - pow(1.-_sigmaCumulationFactor, 2.0*(1.0+_k->currentGeneration))) / _chiN  < 1.4 + 2./(_k->N+1);

 /* cumulation for covariance matrix (pc) using B*D*z~_k->N(0,C) */
 for (size_t d = 0; d < _k->N; ++d)
   _evolutionPath[d] = (1. - _cumulativeCovariance) * _evolutionPath[d] + hsig * sqrt( _cumulativeCovariance * (2. - _cumulativeCovariance) * _effectiveMu ) * _y[d];

 /* update of C  */
 adaptC(hsig);

 /* update masking matrix */
 if(_hasDiscreteVariables) updateDiscreteMutationMatrix();

 /* update of sigma */
 adaptSigma();

 /* numerical error management */

 //treat maximal standard deviations
 //TODO

 //treat minimal standard deviations
 for (size_t d = 0; d < _k->N; ++d) if (_sigma * sqrt(_C[d*_k->N+d]) < _variableSettings[d].minStdDevChange)
 {
   _sigma = (_variableSettings[d].minStdDevChange)/sqrt(_C[d*_k->N+d]) * exp(0.05+_sigmaCumulationFactor/_dampFactor);
   koraliWarning(KORALI_DETAILED, "Sigma increased due to minimal standard deviation.\n");
 }

 //too low coordinate axis deviations
 //TODO

 //treat numerical precision provblems
 //TODO

 /* Test if function values are identical, escape flat fitness */
 if (_currentBestValue == _fitnessVector[_sortingIndex[(int)_muValue]]) {
   _sigma *= exp(0.2+_sigmaCumulationFactor/_dampFactor);
   koraliWarning(KORALI_DETAILED, "Sigma increased due to equal function values.\n");
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
  double ccov1  = 2.0 / (std::pow(_k->N+1.3,2)+_effectiveMu);
  double ccovmu = std::min(1.0-ccov1,  2.0 * (_effectiveMu-2+1/_effectiveMu) / (std::pow(_k->N+2.0,2)+_effectiveMu));
  double sigmasquare = _sigma * _sigma;

  _isEigenSystemUpdate = false;

  /* update covariance matrix */
  for (size_t d = 0; d < _k->N; ++d)
   for (size_t e = _isDiag ? d : 0; e <= d; ++e) {
     _C[d*_k->N+e] = (1 - ccov1 - ccovmu) * _C[d*_k->N+e] 
                        + ccov1 * (_evolutionPath[d] * _evolutionPath[e] 
                        + (1-hsig)*ccov1*_cumulativeCovariance*(2.-_cumulativeCovariance) * _C[d*_k->N+e]);
     for (size_t k = 0; k < _muValue; ++k)
         _C[d*_k->N+e] += ccovmu * _muWeights[k] 
                        * (_samplePopulation[_sortingIndex[k]*_k->N + d] - _discreteMutations[_sortingIndex[k]*_k->N+d] - _previousMean[d]) 
                        * (_samplePopulation[_sortingIndex[k]*_k->N + e] - _discreteMutations[_sortingIndex[k]*_k->N+e] - _previousMean[e])  
                        / sigmasquare;
     if (e < d) _C[e*_k->N+d] = _C[d*_k->N+e];
   }

  /* update maximal and minimal diagonal value */
  _maxDiagCElement = _minDiagCElement = _C[0];
  for (size_t d = 1; d < _k->N; ++d) {
  if (_maxDiagCElement < _C[d*_k->N+d]) _maxDiagCElement = _C[d*_k->N+d];
  else if (_minDiagCElement > _C[d*_k->N+d]) _minDiagCElement = _C[d*_k->N+d];
  }
 } /* if ccov... */
}


void CMAES::adaptSigma()
{
 double sigma_upper = sqrt(_trace/_k->N);
 
 if(_hasDiscreteVariables)
 {
   double pathL2 = 0.0;
   for(size_t d = 0; d < _k->N; ++d) pathL2 += _maskingMatrixSigma[d]*_conjugateEvolutionPath[d]*_conjugateEvolutionPath[d];
   _sigma *= exp(_sigmaCumulationFactor/_dampFactor*(sqrt(pathL2)/_chiS-1.));
 }
 else
 {
   // _sigma *= exp(min(1.0, _sigmaCumulationFactor/_dampFactor*((sqrt(_conjugateEvolutionPathL2Norm)/_chiN)-1.))); (alternative)
   _sigma *= exp(_sigmaCumulationFactor/_dampFactor*(sqrt(_conjugateEvolutionPathL2Norm)/_chiN-1.));
 }
 if(_sigma > sigma_upper)
     koraliLog(KORALI_DETAILED, "Sigma exceeding inital value of _sigma (%f > %f), increase Initial Standard Deviation of variables.\n", _sigma, sigma_upper);
 
 /* upper bound for _sigma */
 if( _isSigmaBounded && (_sigma > sigma_upper) ) _sigma = sigma_upper;
}


void CMAES::updateDiscreteMutationMatrix()
{
  // implemented based on 'A CMA-ES for Mixed-Integer Nonlinear Optimization' by
  // Hansen2011
  
  size_t entries = _k->N + 1; // +1 to prevent 0-ness
  std::fill( std::begin(_maskingMatrixSigma), std::end(_maskingMatrixSigma), 1.0);
  for(size_t d = 0; d < _k->N; ++d) if(_sigma*std::sqrt(_C[d*_k->N+d])/std::sqrt(_sigmaCumulationFactor) < 0.2*_granularity[d]) { _maskingMatrixSigma[d] = 0.0; entries--; }
  _chiS = sqrt((double) entries) * (1. - 1./(4.*entries) + 1./(21.*entries*entries));
  
  _maskingMatrixEntries = 0;
  std::fill( std::begin(_maskingMatrix), std::end(_maskingMatrix), 0.0);
  for(size_t d = 0; d < _k->N; ++d) if(2.0*_sigma*std::sqrt(_C[d*_k->N+d]) < _granularity[d]) { _maskingMatrix[d] = 1.0; _maskingMatrixEntries++; }
 
  _numDiscreteMutations = std::min( std::round(_sampleCount/10.0 + _maskingMatrixEntries + 1) , std::floor(_sampleCount/2.0) - 1);
  std::fill( std::begin(_discreteMutations), std::end(_discreteMutations), 0.0);
  
}


bool CMAES::checkTermination()
{
 
 bool isFinished = false;
 if ( _termCondMinFitnessEnabled && (_k->currentGeneration > 1) && (_bestEverValue >= _termCondMinFitness) )
 {
  isFinished = true;
  koraliLog(KORALI_MINIMAL, "Min fitness value (%+6.3e) > (%+6.3e).\n",  _bestEverValue, _termCondMinFitness);
 }
 
 if ( _termCondMaxFitnessEnabled&& (_k->currentGeneration > 1) && (_bestEverValue >= _termCondMaxFitness) )
 {
  isFinished = true;
  koraliLog(KORALI_MINIMAL, "Max fitness value (%+6.3e) > (%+6.3e)\n",  _bestEverValue, _termCondMaxFitness);
 }

 double range = fabs(_currentBestValue - _previousBestValue);
 if ( _termCondMinFitnessDiffThresholdEnabled && (_k->currentGeneration > 1) && (range <= _termCondMinFitnessDiffThreshold) )
 {
  isFinished = true;
  koraliLog(KORALI_MINIMAL, "Function value differences (%+6.3e) < (%+6.3e)\n",  range, _termCondMinFitnessDiffThreshold);
 }

 size_t idx;
 
 if ( _termCondMinStandardDeviationEnabled )
 {
  size_t cTemp = 0;
  for(idx = 0; idx <_k->N; ++idx )
   cTemp += (_sigma * sqrt(_C[idx*_k->N+idx]) < _termCondMinStandardDeviation * _variableSettings[idx].initialStdDev) ? 1 : 0;
  
  if (cTemp == _k->N) {
   isFinished = true;
   koraliLog(KORALI_MINIMAL, "Object variable changes < %+6.3e\n", _termCondMinStandardDeviation * _variableSettings[idx].initialStdDev);
  }

  for(idx = 0; idx <_k->N; ++idx )
   if ( _termCondMaxStandardDeviationEnabled && (_sigma * sqrt(_C[idx*_k->N+idx]) > _termCondMaxStandardDeviation * _variableSettings[idx].initialStdDev) )
   {
    isFinished = true;
    koraliLog(KORALI_MINIMAL, "Standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n", 
            _termCondMaxStandardDeviation * _variableSettings[idx].initialStdDev);
    break;
   }
 }

 if ( _termCondMaxCovMatrixConditionEnabled && (_maxCovarianceEigenvalue >= _minCovarianceEigenvalue * _termCondMaxCovMatrixCondition) )
 {
   isFinished = true;
   koraliLog(KORALI_MINIMAL, "Maximal condition number %7.2e reached. _maxCovarianceEigenvalue=%7.2e, minEig=%7.2e, _maxDiagCElement=%7.2e, _minDiagCElement=%7.2e\n",
                                _termCondMaxCovMatrixCondition, _maxCovarianceEigenvalue, _minCovarianceEigenvalue, _maxDiagCElement, _minDiagCElement);
 }

 double fac;
 size_t iAchse = 0;
 size_t iKoo = 0;
 /* Component of _mean is not changed anymore */
 if( _termCondMinStandardDeviationStepFactorEnabled)
 if (!_isDiag )
 {
    for (iAchse = 0; iAchse < _k->N; ++iAchse)
    {
    fac = _termCondMinStandardDeviationStepFactor * _sigma * _axisD[iAchse];
    for (iKoo = 0; iKoo < _k->N; ++iKoo){
      if (_mean[iKoo] != _mean[iKoo] + fac * _B[iKoo*_k->N+iAchse])
      break;
    }
    if (iKoo == _k->N)
    {
      isFinished = true;
      koraliLog(KORALI_MINIMAL, "Standard deviation %f*%7.2e in principal axis %ld without effect.\n", _termCondMinStandardDeviationStepFactor, _sigma*_axisD[iAchse], iAchse);
      break;
    }
  }
 }

 /* Component of _mean is not changed anymore */
 if( _termCondMinStandardDeviationStepFactorEnabled )
 for (iKoo = 0; iKoo < _k->N; ++iKoo)
 {
  if (_mean[iKoo] == _mean[iKoo] + _termCondMinStandardDeviationStepFactor*_sigma*sqrt(_C[iKoo*_k->N+iKoo]) )
  {
   isFinished = true;
   koraliLog(KORALI_MINIMAL, "Standard deviation %f*%7.2e in coordinate %ld without effect.\n", _termCondMinStandardDeviationStepFactor, _sigma*sqrt(_C[iKoo*_k->N+iKoo]), iKoo);
   break;
  }

 }

 return isFinished;
}


void CMAES::updateEigensystem(std::vector<double>& M, int flgforce)
{
 if(flgforce == 0 && _isEigenSystemUpdate) return;
 /* if(_k->currentGeneration % _covarianceEigenEvalFreq == 0) return; */

 eigen(_k->N, M, _axisDtmp, _Btmp);
 
 /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
 double minEWtmp = *std::min_element(std::begin(_axisDtmp), std::end(_axisDtmp));
 double maxEWtmp = *std::max_element(std::begin(_axisDtmp), std::end(_axisDtmp));

 if (minEWtmp <= 0.0) 
 { koraliWarning(KORALI_NORMAL, "Min Eigenvalue smaller or equal 0.0 (%+6.3e) after Eigen decomp (no update possible).\n", minEWtmp ); return; }

 for (size_t d = 0; d < _k->N; ++d) 
 {
     _axisDtmp[d] = sqrt(_axisDtmp[d]); 
     if (std::isfinite(_axisDtmp[d]) == false)
     {
    	 koraliWarning(KORALI_NORMAL, "Could not calculate root of Eigenvalue (%+6.3e) after Eigen decomp (no update possible).\n", _axisDtmp[d] );
       return; 
     }
    for (size_t e = 0; e < _k->N; ++e) if (std::isfinite(_B[d*_k->N+e]) == false)
    {
    	koraliWarning(KORALI_NORMAL, "Non finite value detected in B (no update possible).\n");
       return;
    }
 }
 
 /* write back */
 for (size_t d = 0; d < _k->N; ++d) _axisD[d] = _axisDtmp[d];
 _B.assign(std::begin(_Btmp), std::end(_Btmp));

 _minCovarianceEigenvalue = minEWtmp;
 _maxCovarianceEigenvalue = maxEWtmp;

 _isEigenSystemUpdate = true;
}


/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/


void CMAES::eigen(size_t size, std::vector<double>& M,  std::vector<double>& diag, std::vector<double>& Q) const
{
 std::vector<double> data(size * size);

 for (size_t i = 0; i <  size; i++)
 for (size_t j = 0; j <= i; j++)
 {
  data[i*size + j] = M[i*_k->N+j];
  data[j*size + i] = M[i*_k->N+j];
 }

 // GSL Workspace
 gsl_vector* gsl_eval = gsl_vector_alloc(_k->N);
 gsl_matrix* gsl_evec = gsl_matrix_alloc(_k->N, _k->N);
 gsl_eigen_symmv_workspace* gsl_work =  gsl_eigen_symmv_alloc(_k->N);

 gsl_matrix_view m = gsl_matrix_view_array (data.data(), size, size);

 gsl_eigen_symmv (&m.matrix, gsl_eval, gsl_evec, gsl_work);
 gsl_eigen_symmv_sort (gsl_eval, gsl_evec, GSL_EIGEN_SORT_ABS_ASC);

 for (size_t i = 0; i < size; i++)
 {
  gsl_vector_view gsl_evec_i = gsl_matrix_column (gsl_evec, i);
  for (size_t j = 0; j < size; j++) Q[j*_k->N+i] = gsl_vector_get (&gsl_evec_i.vector, j);
 }

 for (size_t i = 0; i < size; i++) diag[i] = gsl_vector_get (gsl_eval, i);

 gsl_vector_free(gsl_eval);
 gsl_matrix_free(gsl_evec);
 gsl_eigen_symmv_free(gsl_work);
}


void CMAES::sort_index(const std::vector<double>& vec, std::vector<size_t>& _sortingIndex, size_t n) const
{
  // initialize original _sortingIndex locations
  std::iota(std::begin(_sortingIndex), std::begin(_sortingIndex)+n, (size_t) 0);

  // sort indexes based on comparing values in vec
  std::sort(std::begin(_sortingIndex), std::begin(_sortingIndex)+n, [vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; } );

}


void CMAES::printGeneration()
{
 koraliLog(KORALI_MINIMAL, "CMA-ES Generation %zu\n", _k->currentGeneration);

 koraliLog(KORALI_NORMAL, "Sigma:                        %+6.3e\n", _sigma);
 koraliLog(KORALI_NORMAL, "Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
 koraliLog(KORALI_NORMAL, "Diagonal Covariance:    Min = %+6.3e -  Max = %+6.3e\n", _minDiagCElement, _maxDiagCElement);
 koraliLog(KORALI_NORMAL, "Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", _minCovarianceEigenvalue, _maxCovarianceEigenvalue);

	koraliLog(KORALI_DETAILED, "Variable = (MeanX, BestX):\n");
	for (size_t d = 0; d < _k->N; d++) koraliLogData(KORALI_DETAILED, "         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _mean[d], _bestEverSample[d]);

	koraliLog(KORALI_DETAILED, "Covariance Matrix:\n");
	for (size_t d = 0; d < _k->N; d++)
	{
	 for (size_t e = 0; e <= d; e++) koraliLogData(KORALI_DETAILED, "   %+6.3e  ",_C[d*_k->N+e]);
	 koraliLog(KORALI_DETAILED, "\n");
	}

	koraliLog(KORALI_DETAILED, "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);

	koraliLog(KORALI_NORMAL, "--------------------------------------------------------------------\n");
}

void CMAES::finalize()
{
	koraliLog(KORALI_MINIMAL, "CMA-ES Finished\n");
	koraliLog(KORALI_MINIMAL, "Optimum found: %e\n", _bestEverValue);
	koraliLog(KORALI_MINIMAL, "Optimum found at:\n");
	for (size_t d = 0; d < _k->N; ++d) koraliLogData(KORALI_MINIMAL, "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverSample[d]);
	koraliLog(KORALI_MINIMAL, "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
	koraliLog(KORALI_MINIMAL, "--------------------------------------------------------------------\n");
}
