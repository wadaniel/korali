#include "korali.hpp"

#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort

using namespace Korali::Solver;

CCMAES::CCMAES()
{
 // Initializing Gaussian Generator
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"] = "Gaussian";
 jsGaussian["Mean"] = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = std::make_shared<Variable>();
 _gaussianGenerator->setDistribution(jsGaussian);
 
 _uniformGenerator = nullptr;
}

CCMAES::~CCMAES()
{
}


void CCMAES::initialize()
{
 // Checking for accepted problem types
 std::string pName = _k->_js["Problem"]["Type"];
 bool acceptableProblem = false;
 if (pName == "Optimization")  acceptableProblem = true;
 if (pName == "Constrained Optimization")  acceptableProblem = true;
 if (pName == "Bayesian Inference")  acceptableProblem = true;
 if (acceptableProblem == false) koraliError("CCMAES cannot solve problems of type: '%s'.\n", pName.c_str());

 size_t s_max  = std::max(_sampleCount,  _viabilitySampleCount);
 size_t mu_max = std::max(_muValue, _viabilityMu);

 _chiSquareNumber = sqrt((double) _k->N) * (1. - 1./(4.*_k->N) + 1./(21.*_k->N*_k->N));
 _chiSquareNumberDiscreteMutations = sqrt((double) _k->N) * (1. - 1./(4.*_k->N) + 1./(21.*_k->N*_k->N));

 _constraintsDefined = (_k->_constraints.size() > 0);
 if(_constraintsDefined) { _isViabilityRegime = true; }
 else                    _isViabilityRegime = false;


 if(_isViabilityRegime) {
     _currentSampleCount  = _viabilitySampleCount;
     _currentSampleMu = _viabilityMu;
 } else {
     _currentSampleCount  = _sampleCount;
     _currentSampleMu = _muValue;
 }

 // Allocating Memory
 _samplePopulation.resize(_k->N*s_max);

 _evolutionPath.resize(_k->N);
 _conjugateEvolutionPath.resize(_k->N);
 _temporaryBDZMatrix.resize(_k->N);
 _meanUpdate.resize(_k->N);
 _meanUpdate.resize(_k->N);
 _previousMean.resize(_k->N);
 _bestEverCoordinates.resize(_k->N);
 _axisLengths.resize(_k->N);
 _temporaryAxisLengths.resize(_k->N);
 _currentBestCoordinates.resize(_k->N);

 _sortingIndex.resize(s_max);

 _isInitializedSample.resize(s_max);
 _fitnessVector.resize(s_max);

 _covarianceMatrix.resize(_k->N*_k->N);
 _auxiliarCovarianceMatrix.resize(_k->N*_k->N);
 _covarianceEigenvectorMatrix.resize(_k->N*_k->N);
 _auxiliarCovarianceEigenvectorMatrix.resize(_k->N*_k->N);

 _randomNumberStorage.resize(s_max*_k->N);
 _bDZMatrix.resize(s_max*_k->N);
 
 _granularity.resize(_k->N);
 _maskingMatrix.resize(_k->N);
 _maskingMatrixSigma.resize(_k->N);
 _discreteMutations.resize(_k->N*_sampleCount);
 std::fill( std::begin(_discreteMutations), std::end(_discreteMutations), 0.0);

 _numberMaskingMatrixEntries = 0;
 _numberOfDiscreteMutations = 0;


 // Initailizing Mu
 _muWeights.resize(mu_max);

 // CCMAES variables
 if (_constraintsDefined)
 {
  if( (_globalSuccessLearningRate <= 0.0) || (_globalSuccessLearningRate > 1.0) ) 
    koraliError("CCMAES Error: Invalid Global Success Learning Rate (%f), must be greater than 0.0 and less than 1.0\n",  _globalSuccessLearningRate );
  if( (_targetSuccessRate <= 0.0) || (_targetSuccessRate > 1.0) )
    koraliError("CCMAES Error: Invalid Target Success Rate (%f), must be greater than 0.0 and less than 1.0\n",  _targetSuccessRate );
  if(_covarianceMatrixAdaptionStrength <= 0.0) 
    koraliError(" CCMAES Error: Invalid Adaption Size (%f), must be greater than 0.0\n", _covarianceMatrixAdaptionStrength );

  _globalSuccessRate = 0.5;
  _bestValidSample   = -1;
  _sampleConstraintViolationCounts.resize(s_max);
  _viabilityBoundaries.resize(_k->_constraints.size());

  _viabilityImprovement.resize(s_max);
  _viabilityIndicator.resize(_k->_constraints.size());
  _constraintEvaluations.resize(_k->_constraints.size());

  for (size_t c = 0; c < _k->_constraints.size(); c++) _viabilityIndicator[c].resize(s_max);
  for (size_t c = 0; c < _k->_constraints.size(); c++) _constraintEvaluations[c].resize(s_max);

  _normalConstraintApproximation.resize(_k->_constraints.size());
  for (size_t i = 0; i < _k->_constraints.size(); i++) _normalConstraintApproximation[i].resize(_k->N);

  _bestConstraintEvaluations.resize(_k->_constraints.size());

  _normalVectorLearningRate = 1.0/(2.0+_k->N);
  _covarianceMatrixAdaptionFactor = _covarianceMatrixAdaptionStrength/(_k->N+2.);

 }
 else
 {
  _globalSuccessRate = -1.0;
  _covarianceMatrixAdaptionFactor              = -1.0;
  _bestValidSample   = 0;
 }
  
 _constraintEvaluationCount = 0;
 _covarianceMatrixAdaptationCount           = 0;
 _maximumViolationCount         = 0;

 // Setting algorithm internal variables
 if (_constraintsDefined) { initMuWeights(_viabilityMu); }
 else initMuWeights(_muValue);

 initCovariance();

 _infeasibleSampleCount = 0;
 _resampledParameterCount       = 0;
 
 _bestEverValue = -std::numeric_limits<double>::max();
 _conjugateEvolutionPathL2Norm = 0.0;

 /* set _meanUpdate */
 for (size_t i = 0; i < _k->N; ++i)
 {
   if(_k->_variables[i]->_initialMean < _k->_variables[i]->_lowerBound || _k->_variables[i]->_initialMean > _k->_variables[i]->_upperBound)
   koraliWarning(KORALI_MINIMAL, "Initial Value (%.4f) of variable \'%s\' is out of bounds (%.4f-%.4f).\n",
            _k->_variables[i]->_initialMean,
            _k->_variables[i]->_name.c_str(),
            _k->_variables[i]->_lowerBound,
            _k->_variables[i]->_upperBound);
   _meanUpdate[i] = _previousMean[i] = _k->_variables[i]->_initialMean;
 }

 /* set _granularity for discrete variables */
 size_t numDiscretes = 0;
 for (size_t i = 0; i < _k->N; ++i)
 {
   if( (_k->_variables[i]->_isDiscrete == true) && _k->_variables[i]->_granularity == 0.0)
       koraliError("Granularity not set for discrete variable \'%s\'.\n", _k->_variables[i]->_name.c_str());
   if (_k->_variables[i]->_isDiscrete == true) numDiscretes++;
    _granularity[i] = _k->_variables[i]->_granularity;
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
 


}


void CCMAES::runGeneration()
{
 if ( _constraintsDefined ) checkMeanAndSetRegime();
 prepareGeneration();
 if ( _constraintsDefined ){ updateConstraints(); handleConstraints(); }
 evaluateSamples();
 updateDistribution();
}


void CCMAES::evaluateSamples()
{
  while (_finishedSampleCount < _currentSampleCount)
  {
    for (size_t i = 0; i < _currentSampleCount; i++) if (_isInitializedSample[i] == false)
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

void CCMAES::initMuWeights(size_t numsamplesmu)
{
 // Initializing Mu Weights
 if      (_muType == "Linear")       for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = numsamplesmu - i;
 else if (_muType == "Equal")        for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = 1.;
 else if (_muType == "Logarithmic")  for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = log(std::max( (double)numsamplesmu, 0.5*_currentSampleCount)+0.5)-log(i+1.);
 else  koraliError("CCMAES - Invalid setting of Mu Type (%s) (Linear, Equal, or Logarithmic accepted).",  _muType.c_str());

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
    if (_constraintsDefined) _sigmaCumulationFactor = sqrt(_effectiveMu) / ( sqrt(_effectiveMu) + sqrt(_k->N) );
    else                     _sigmaCumulationFactor = (_effectiveMu + 2.0) / (_k->N + _effectiveMu + 3.0);

 // Setting Damping Factor
 _dampFactor = _initialDampFactor;
 if (_dampFactor <= 0.0)
     _dampFactor = (1.0 + 2*std::max(0.0, sqrt((_effectiveMu-1.0)/(_k->N+1.0)) - 1)) + _sigmaCumulationFactor;

}


void CCMAES::initCovariance()
{

 for(size_t d = 0; d < _k->N; ++d)
 {
    if ( _k->_variables[d]->_initialStandardDeviation<0 )
    if ( std::isfinite(_k->_variables[d]->_lowerBound) && std::isfinite(_k->_variables[d]->_upperBound ) )
            _k->_variables[d]->_initialStandardDeviation = 0.3*(_k->_variables[d]->_upperBound-_k->_variables[d]->_lowerBound);
    else 
        koraliError("Lower/Upper Bound not defined, and Initial Standard Dev not defined for variable '%s'\n", _k->_variables[d]->_name.c_str());
 }
 
 // Setting Sigma
 _trace = 0.0;
 for (size_t i = 0; i < _k->N; ++i) _trace += _k->_variables[i]->_initialStandardDeviation*_k->_variables[i]->_initialStandardDeviation;
 _sigma = sqrt(_trace/_k->N);

 // Setting B, C and _axisD
 for (size_t i = 0; i < _k->N; ++i)
 {
  _covarianceEigenvectorMatrix[i*_k->N+i] = 1.0;
  _covarianceMatrix[i*_k->N+i] = _axisLengths[i] = _k->_variables[i]->_initialStandardDeviation * sqrt(_k->N / _trace);
  _covarianceMatrix[i*_k->N+i] *= _covarianceMatrix[i*_k->N+i];
 }

 _minimumCovarianceEigenvalue = *std::min_element(std::begin(_axisLengths), std::end(_axisLengths));
 _maximumCovarianceEigenvalue = *std::max_element(std::begin(_axisLengths), std::end(_axisLengths));
 
 _minimumCovarianceEigenvalue = _minimumCovarianceEigenvalue * _minimumCovarianceEigenvalue;
 _maximumCovarianceEigenvalue = _maximumCovarianceEigenvalue * _maximumCovarianceEigenvalue;

 _maximumDiagonalCovarianceMatrixElement=_covarianceMatrix[0]; for(size_t i=1;i<_k->N;++i) if(_maximumDiagonalCovarianceMatrixElement<_covarianceMatrix[i*_k->N+i]) _maximumDiagonalCovarianceMatrixElement=_covarianceMatrix[i*_k->N+i];
 _minimumDiagonalCovarianceMatrixElement=_covarianceMatrix[0]; for(size_t i=1;i<_k->N;++i) if(_minimumDiagonalCovarianceMatrixElement>_covarianceMatrix[i*_k->N+i]) _minimumDiagonalCovarianceMatrixElement=_covarianceMatrix[i*_k->N+i];
}


void CCMAES::processSample(size_t sampleId, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&_samplePopulation[sampleId*_k->N]);
 fitness = logPrior+fitness;
 if(std::isfinite(fitness) == false) 
 {
   fitness = -1.0 * std::numeric_limits<double>::max();
   koraliWarning(KORALI_DETAILED,"Sample %zu returned non finite fitness (set to %e)!\n", sampleId, fitness);
 }
 _fitnessVector[sampleId] = fitness;
 _finishedSampleCount++;
}


void CCMAES::checkMeanAndSetRegime()
{
  if (_isViabilityRegime == false) return; /* mean already inside valid domain, no udpates */

  Korali::Model data;
  for (size_t i = 0; i < _k->N; i++) data._computationalVariables.push_back(_meanUpdate[i]);
  for (size_t c = 0; c < _k->_constraints.size(); c++)
  {
    _constraintEvaluationCount++;
    data._results.clear();
    _k->_constraints[c](data);
    if (data._results.size() != 1)  koraliError("Constraints require exactly a 1-element result array. Provided: %lu.\n", data._results.size() );
    if (data._results[0] > 0.0) return; /* mean violates constraint, do nothing */
  }

  /* mean inside domain, switch regime and update internal variables */
  _isViabilityRegime = false;

  for (size_t c = 0; c < _k->_constraints.size(); c++) { _viabilityBoundaries[c] = 0; }
  _currentSampleCount = _sampleCount;
  _currentSampleMu = _muValue;

  _bestEverValue = -std::numeric_limits<double>::max();
  initMuWeights(_currentSampleMu);
  initCovariance();
}


void CCMAES::updateConstraints() //TODO: maybe parallelize constraint evaluations (DW)
{

 for(size_t i = 0; i < _currentSampleCount; i++) _sampleConstraintViolationCounts[i] = 0;
 _maximumViolationCount = 0;

 for(size_t c = 0; c < _k->_constraints.size(); c++)
 {
  double maxviolation = 0.0;
  for(size_t i = 0; i < _currentSampleCount; ++i)
  {
    _constraintEvaluationCount++;
    Korali::Model data;
    for (size_t j = 0; j < _k->N; j++)
     data._computationalVariables.push_back(_samplePopulation[i*_k->N + j]);

    _k->_constraints[c](data);
    if (data._results.size() != 1)  koraliError("Constraints require exactly a 1-element result array. Provided: %lu.\n", data._results.size() );
    _constraintEvaluations[c][i] = data._results[0];

    if ( _constraintEvaluations[c][i] > maxviolation ) maxviolation = _constraintEvaluations[c][i];
    if ( _k->currentGeneration == 0 && _isViabilityRegime ) _viabilityBoundaries[c] = maxviolation;

    if ( _constraintEvaluations[c][i] > _viabilityBoundaries[c] + 1e-12 ) _sampleConstraintViolationCounts[i]++;
    if ( _sampleConstraintViolationCounts[i] > _maximumViolationCount ) _maximumViolationCount = _sampleConstraintViolationCounts[i];

  }
 }

}


void CCMAES::reEvaluateConstraints() //TODO: maybe we can parallelize constraint evaluations (DW)
{
  _maximumViolationCount = 0;
  for(size_t i = 0; i < _currentSampleCount; ++i) if(_sampleConstraintViolationCounts[i] > 0)
  {
    _sampleConstraintViolationCounts[i] = 0;
    for(size_t c = 0; c < _k->_constraints.size(); c++)
    {
     _constraintEvaluationCount++;
     Korali::Model data;
     for (size_t j = 0; j < _k->N; j++)
      data._computationalVariables.push_back(_samplePopulation[i*_k->N + j]);

     _k->_constraints[c](data);
     if (data._results.size() != 1)  koraliError("Constraints require exactly a 1-element result array. Provided: %lu.\n", data._results.size() );
     _constraintEvaluations[c][i] = data._results[0];

      if( _constraintEvaluations[c][i] > _viabilityBoundaries[c] + 1e-12 ) { _viabilityIndicator[c][i] = true; _sampleConstraintViolationCounts[i]++; }
      else _viabilityIndicator[c][i] = false;

    }
    if (_sampleConstraintViolationCounts[i] > _maximumViolationCount) _maximumViolationCount = _sampleConstraintViolationCounts[i];
  }
}


void CCMAES::updateViabilityBoundaries()
{
 for(size_t c = 0; c < _k->_constraints.size(); c++)
 {
  double maxviolation = 0.0;
  for(size_t i = 0; i < _currentSampleMu /* _currentSampleCount alternative */ ; ++i) if (_constraintEvaluations[c][_sortingIndex[i]] > maxviolation)
    maxviolation = _constraintEvaluations[c][_sortingIndex[i]];

  _viabilityBoundaries[c] = std::max( 0.0, std::min(_viabilityBoundaries[c], 0.5*(maxviolation + _viabilityBoundaries[c])) );
 }
}


bool CCMAES::isFeasible(size_t sampleIdx) const
{
 for (size_t d = 0; d < _k->N; ++d)
  if (_samplePopulation[ sampleIdx*_k->N+d ] < _k->_variables[d]->_lowerBound || _samplePopulation[ sampleIdx*_k->N+d ] > _k->_variables[d]->_upperBound) return false;
 return true;
}


void CCMAES::prepareGeneration()
{

 /* calculate eigensystem */
 for (size_t d = 0; d < _k->N; ++d) _auxiliarCovarianceMatrix.assign(std::begin(_covarianceMatrix), std::end(_covarianceMatrix));
 updateEigensystem(_auxiliarCovarianceMatrix);

 for (size_t i = 0; i < _currentSampleCount; ++i)
 {
     size_t initial_infeasible = _infeasibleSampleCount;
     sampleSingle(i);
     while( isFeasible( i ) == false )
     {
       _infeasibleSampleCount++;
       sampleSingle(i);

       if ( _maxInfeasibleResamplings_enabled )
       if ( (_infeasibleSampleCount - initial_infeasible) > _maxInfeasibleResamplings )
        koraliError("Exiting resampling loop (sample %zu), max resamplings (%zu) reached.\n", i, _maxInfeasibleResamplings);
     }
 }

 _finishedSampleCount = 0;
 for (size_t i = 0; i < _currentSampleCount; i++) _isInitializedSample[i] = false;
}


void CCMAES::sampleSingle(size_t sampleIdx)
{
  for (size_t d = 0; d < _k->N; ++d)
  {
   _randomNumberStorage[sampleIdx*_k->N+d] = _gaussianGenerator->getRandomNumber();
   if (_isDiagonal) {
     _bDZMatrix[sampleIdx*_k->N+d] = _axisLengths[d] * _randomNumberStorage[sampleIdx*_k->N+d];
     _samplePopulation[sampleIdx * _k->N + d] = _meanUpdate[d] + _sigma * _bDZMatrix[sampleIdx*_k->N+d];
   }
   else _temporaryBDZMatrix[d] = _axisLengths[d] * _randomNumberStorage[sampleIdx*_k->N+d];
  }

  if (!_isDiagonal)
   for (size_t d = 0; d < _k->N; ++d) {
    _bDZMatrix[sampleIdx*_k->N+d] = 0.0;
    for (size_t e = 0; e < _k->N; ++e) _bDZMatrix[sampleIdx*_k->N+d] += _covarianceEigenvectorMatrix[d*_k->N+e] * _temporaryBDZMatrix[e];
    _samplePopulation[sampleIdx * _k->N + d] = _meanUpdate[d] + _sigma * _bDZMatrix[sampleIdx*_k->N+d];
  }
  
  if(_hasDiscreteVariables)
  {
    if ( (sampleIdx+1) < _numberOfDiscreteMutations )
    {
      double p_geom = std::pow(0.7, 1.0/_numberMaskingMatrixEntries);
      size_t select = std::floor(_uniformGenerator->getRandomNumber() * _numberMaskingMatrixEntries);

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
    else if ( (sampleIdx+1) == _numberOfDiscreteMutations )
    {
      for(size_t d = 0; d < _k->N; ++d) if( _granularity[d] != 0.0 )
      {
        double dmutation = std::round(_bestEverCoordinates[d]/_granularity[d]) * _granularity[d] - _samplePopulation[sampleIdx*_k->N+d];
        _discreteMutations[sampleIdx*_k->N+d] = dmutation;
        _samplePopulation[sampleIdx*_k->N+d] += dmutation;
      }
    }
  }

}


void CCMAES::updateDistribution()
{

 /* Generate _sortingIndex */
 sort_index(_fitnessVector, _sortingIndex, _currentSampleCount);

 if( (_constraintsDefined == false) ||  _isViabilityRegime )
  _bestValidSample = 0;
 else
 {
  _bestValidSample = -1;
  for (size_t i = 0; i < _currentSampleCount; i++) if(_sampleConstraintViolationCounts[_sortingIndex[i]] == 0) _bestValidSample = _sortingIndex[i];
  if(_bestValidSample == -1) { koraliWarning(KORALI_DETAILED, "All samples violate constraints, no updates taking place.\n"); return; }
 }

 /* update function value history */
 _previousBestValue = _currentBestValue;

 /* update current best */
 _currentBestValue = _fitnessVector[_bestValidSample];
 for (size_t d = 0; d < _k->N; ++d) _currentBestCoordinates[d] = _samplePopulation[_bestValidSample*_k->N + d];

 /* update xbestever */
 if ( _currentBestValue > _bestEverValue )
 {
  _previousBestEverValue = _bestEverValue;
  _bestEverValue = _currentBestValue;
  for (size_t d = 0; d < _k->N; ++d) _bestEverCoordinates[d]   = _currentBestCoordinates[d];
  for (size_t c = 0; c < _k->_constraints.size(); c++) _bestConstraintEvaluations[c] = _constraintEvaluations[c][_bestValidSample];
 }

 /* set weights */
 for (size_t d = 0; d < _k->N; ++d) {
   _previousMean[d] = _meanUpdate[d];
   _meanUpdate[d] = 0.;
   for (size_t i = 0; i < _currentSampleMu; ++i)
     _meanUpdate[d] += _muWeights[i] * _samplePopulation[_sortingIndex[i]*_k->N + d];

   _meanUpdate[d] = (_meanUpdate[d] - _previousMean[d])/_sigma;
 }

 /* calculate z := D^(-1) * B^(T) * _meanUpdate into _temporaryBDZMatrix */
 for (size_t d = 0; d < _k->N; ++d) {
  double sum = 0.0;
  if (_isDiagonal) sum = _meanUpdate[d];
  else for (size_t e = 0; e < _k->N; ++e) sum += _covarianceEigenvectorMatrix[e*_k->N+d] * _meanUpdate[e]; /* B^(T) * _meanUpdate ( iterating B[e][d] = B^(T) ) */

  _temporaryBDZMatrix[d] = sum / _axisLengths[d]; /* D^(-1) * B^(T) * _meanUpdate */
 }

 _conjugateEvolutionPathL2Norm = 0.0;

 /* cumulation for _sigma (ps) using B*z */
 for (size_t d = 0; d < _k->N; ++d) {
    double sum = 0.0;
    if (_isDiagonal) sum = _temporaryBDZMatrix[d];
    else for (size_t e = 0; e < _k->N; ++e) sum += _covarianceEigenvectorMatrix[d*_k->N+e] * _temporaryBDZMatrix[e];

    _conjugateEvolutionPath[d] = (1. - _sigmaCumulationFactor) * _conjugateEvolutionPath[d] + sqrt(_sigmaCumulationFactor * (2. - _sigmaCumulationFactor) * _effectiveMu) * sum;

    /* calculate norm(ps)^2 */
    _conjugateEvolutionPathL2Norm += _conjugateEvolutionPath[d] * _conjugateEvolutionPath[d];
 }

 int hsig = (1.4 + 2.0/(_k->N+1) > sqrt(_conjugateEvolutionPathL2Norm) / sqrt(1. - pow(1.-_sigmaCumulationFactor, 2.0*(1.0+_k->currentGeneration))) / _chiSquareNumber);

 /* cumulation for covariance matrix (pc) using B*D*z~_k->N(0,C) */
 for (size_t d = 0; d < _k->N; ++d) 
    _evolutionPath[d] = (1. - _cumulativeCovariance) * _evolutionPath[d] + hsig * sqrt( _cumulativeCovariance * (2. - _cumulativeCovariance) * _effectiveMu ) * _meanUpdate[d];

 /* update covariance matrix  */
 adaptC(hsig);

 /* update masking matrix */
 if(_hasDiscreteVariables) updateDiscreteMutationMatrix();
 
 /* update viability bounds */
 if( _constraintsDefined && (_isViabilityRegime == false) ) updateViabilityBoundaries();

 /* update sigma */
 updateSigma();

 /* numerical error management */
 numericalErrorTreatment();

}

void CCMAES::adaptC(int hsig)
{
  /* definitions for speeding up inner-most loop */
  //double ccov1  = std::min(_covarianceMatrixLearningRate * (1./_muCovariance) * (_isDiagonal ? (_k->N+1.5) / 3. : 1.), 1.); (orig, alternative)
  //double ccovmu = std::min(_covarianceMatrixLearningRate * (1-1./_muCovariance) * (_isDiagonal ? (_k->N+1.5) / 3. : 1.), 1.-ccov1); (orig, alternative)
  double ccov1  = 2.0 / (std::pow(_k->N+1.3,2)+_effectiveMu);
  double ccovmu = std::min(1.0-ccov1,  2.0 * (_effectiveMu-2+1/_effectiveMu) / (std::pow(_k->N+2.0,2)+_effectiveMu));
  double sigmasquare = _sigma * _sigma;

  /* update covariance matrix */
  for (size_t d = 0; d < _k->N; ++d)
   for (size_t e = _isDiagonal ? d : 0; e <= d; ++e) 
   {
     _covarianceMatrix[d*_k->N+e] = (1 - ccov1 - ccovmu) * _covarianceMatrix[d*_k->N+e]
                        + ccov1 * (_evolutionPath[d] * _evolutionPath[e] 
                        + (1-hsig)*ccov1*_cumulativeCovariance*(2.-_cumulativeCovariance) * _covarianceMatrix[d*_k->N+e]);
     
     for (size_t k = 0; k < _currentSampleMu; ++k)
         _covarianceMatrix[d*_k->N+e] += ccovmu * _muWeights[k] * (_samplePopulation[_sortingIndex[k]*_k->N + d]
                            - _previousMean[d]) * (_samplePopulation[_sortingIndex[k]*_k->N + e] - _previousMean[e]) / sigmasquare;
     
     if (e < d) _covarianceMatrix[e*_k->N+d] = _covarianceMatrix[d*_k->N+e];
   }

  /* update maximal and minimal diagonal value */
  _maximumDiagonalCovarianceMatrixElement = _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[0];
  for (size_t d = 1; d < _k->N; ++d) {
  if (_maximumDiagonalCovarianceMatrixElement < _covarianceMatrix[d*_k->N+d]) _maximumDiagonalCovarianceMatrixElement = _covarianceMatrix[d*_k->N+d];
  else if (_minimumDiagonalCovarianceMatrixElement > _covarianceMatrix[d*_k->N+d])  _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[d*_k->N+d];
  }
}


void CCMAES::updateSigma()
{
 /* update for non-viable region */
 if( _constraintsDefined && (_isViabilityRegime == false) )
 {
   _globalSuccessRate = (1-_globalSuccessLearningRate)*_globalSuccessRate;
   if ( _previousBestEverValue != _bestEverValue ) _globalSuccessRate += _globalSuccessLearningRate;

   _sigma *= exp((_globalSuccessRate-(_targetSuccessRate/(1.0-_targetSuccessRate))*(1-_globalSuccessRate))/_dampFactor);
 }
 /* update for discrte variables */
 else if(_hasDiscreteVariables)
 {
   double pathL2 = 0.0;
   for(size_t d = 0; d < _k->N; ++d) pathL2 += _maskingMatrixSigma[d]*_conjugateEvolutionPath[d]*_conjugateEvolutionPath[d];
   _sigma *= exp(_sigmaCumulationFactor/_dampFactor*(sqrt(pathL2)/_chiSquareNumberDiscreteMutations-1.));
 }
 /* standard update */
 else
 {
   // _sigma *= exp(min(1.0, _sigmaCumulationFactor/_dampFactor*((sqrt(_conjugateEvolutionPathL2Norm)/_chiSquareNumber)-1.))); (alternative)
   _sigma *= exp(_sigmaCumulationFactor/_dampFactor*(sqrt(_conjugateEvolutionPathL2Norm)/_chiSquareNumber-1.));
 }

 /* escape flat fitness */
 if (_currentBestValue == _fitnessVector[_sortingIndex[(int)_currentSampleMu]]) {
   _sigma *= exp(0.2+_sigmaCumulationFactor/_dampFactor);
   koraliWarning(KORALI_DETAILED, "Sigma increased due to equal function values.\n");
 }

 /* upper bound check for _sigma */
 double upperBound = sqrt(_trace/_k->N);
 
 if(_sigma > upperBound) 
 {
  koraliLog(KORALI_DETAILED, "Sigma exceeding inital value of _sigma (%f > %f), increase Initial Standard Deviation of variables.\n", _sigma, upperBound);
  if( _isSigmaBounded ) _sigma = upperBound;
 }

}


void CCMAES::numericalErrorTreatment()
{
 //treat maximal standard deviations
 //TODO

 //treat minimal standard deviations
 for (size_t d = 0; d < _k->N; ++d) if (_sigma * sqrt(_covarianceMatrix[d*_k->N+d]) < _k->_variables[d]->_minimumStandardDeviationChanges)
 {
   _sigma = (_k->_variables[d]->_minimumStandardDeviationChanges)/sqrt(_covarianceMatrix[d*_k->N+d]) * exp(0.05+_sigmaCumulationFactor/_dampFactor);
   koraliWarning(KORALI_DETAILED, "Sigma increased due to minimal standard deviation.\n");
 }

 //too low coordinate axis deviations
 //TODO

 //treat numerical precision provblems
 //TODO
}


void CCMAES::handleConstraints()
{
 size_t initial_resampled = _resampledParameterCount;
 size_t initial_corrections = _covarianceMatrixAdaptationCount;

 while( _maximumViolationCount > 0 )
 {
  for (size_t i = 0; i < _k->N; i++) _auxiliarCovarianceMatrix.assign(std::begin(_covarianceMatrix), std::end(_covarianceMatrix));

  for(size_t i = 0; i < _currentSampleCount; ++i) if (_sampleConstraintViolationCounts[i] > 0)
  {
    //update constraint normal
    for( size_t c = 0; c < _k->_constraints.size(); c++ ) if ( _viabilityIndicator[c][i] == true )
    {
        _covarianceMatrixAdaptationCount++;

        double v2 = 0;
        for( size_t d = 0; d < _k->N; ++d)
        {
            _normalConstraintApproximation[c][d] = (1.0-_normalVectorLearningRate)*_normalConstraintApproximation[c][d]+_normalVectorLearningRate*_bDZMatrix[i*_k->N+d];
            v2 += _normalConstraintApproximation[c][d]*_normalConstraintApproximation[c][d];
        }
        for( size_t d = 0; d < _k->N; ++d)
          for( size_t e = 0; e < _k->N; ++e)
            _auxiliarCovarianceMatrix[d*_k->N+e] = _auxiliarCovarianceMatrix[d*_k->N+e] - ((_covarianceMatrixAdaptionFactor * _covarianceMatrixAdaptionFactor * _normalConstraintApproximation[c][d]*_normalConstraintApproximation[c][e])/(v2*_sampleConstraintViolationCounts[i]*_sampleConstraintViolationCounts[i]));
    }
   }

  updateEigensystem(_auxiliarCovarianceMatrix);

  /* in original some stopping criterion (TOLX) */
  // TODO

  //resample invalid points
  for(size_t i = 0; i < _currentSampleCount; ++i) if(_sampleConstraintViolationCounts[i] > 0)
  {
    do
    {
     _resampledParameterCount++;
     sampleSingle(i);

     if(_maxInfeasibleResamplings_enabled)
     if(_resampledParameterCount-initial_resampled > _maxInfeasibleResamplings)
     {
        koraliWarning(KORALI_DETAILED, "Exiting resampling loop, max resamplings (%zu) reached.\n", _maxInfeasibleResamplings);
        reEvaluateConstraints();
        return;
     }

    }
    while( isFeasible(i) == false );
  }

  reEvaluateConstraints();

  if(_covarianceMatrixAdaptationCount - initial_corrections > _maxCovarianceMatrixCorrections)
  {
    koraliWarning(KORALI_DETAILED, "Exiting adaption loop, max adaptions (%zu) reached.\n", _maxCovarianceMatrixCorrections);
    return;
  }

 }//while _maximumViolationCount > 0

}


void CCMAES::updateDiscreteMutationMatrix()
{
  // implemented based on 'A CMA-ES for Mixed-Integer Nonlinear Optimization' by
  // Hansen2011
  
  size_t entries = _k->N + 1; // +1 to prevent 0-ness
  std::fill( std::begin(_maskingMatrixSigma), std::end(_maskingMatrixSigma), 1.0);
  for(size_t d = 0; d < _k->N; ++d) if(_sigma*std::sqrt(_covarianceMatrix[d*_k->N+d])/std::sqrt(_sigmaCumulationFactor) < 0.2*_granularity[d]) { _maskingMatrixSigma[d] = 0.0; entries--; }
  _chiSquareNumberDiscreteMutations = sqrt((double) entries) * (1. - 1./(4.*entries) + 1./(21.*entries*entries));
  
  _numberMaskingMatrixEntries = 0;
  std::fill( std::begin(_maskingMatrix), std::end(_maskingMatrix), 0.0);
  for(size_t d = 0; d < _k->N; ++d) if(2.0*_sigma*std::sqrt(_covarianceMatrix[d*_k->N+d]) < _granularity[d]) { _maskingMatrix[d] = 1.0; _numberMaskingMatrixEntries++; }
 
  _numberOfDiscreteMutations = std::min( std::round(_sampleCount/10.0 + _numberMaskingMatrixEntries + 1) , std::floor(_sampleCount/2.0) - 1);
  std::fill( std::begin(_discreteMutations), std::end(_discreteMutations), 0.0);
  
}


bool CCMAES::checkTermination()
{
 
 bool isFinished = false;
 if ( _minFitness_enabled && (_isViabilityRegime == false) && (_k->currentGeneration > 1) && (_bestEverValue >= _minFitness) )
 {
  isFinished = true;
  koraliLog(KORALI_MINIMAL, "Min fitness value (%+6.3e) > (%+6.3e).\n",  _bestEverValue, _minFitness);
 }
 
 if ( _maxFitness_enabled && (_isViabilityRegime == false) && (_k->currentGeneration > 1) && (_bestEverValue >= _maxFitness) )
 {
  isFinished = true;
  koraliLog(KORALI_MINIMAL, "Max fitness value (%+6.3e) > (%+6.3e)\n",  _bestEverValue, _maxFitness);
 }

 double range = fabs(_currentBestValue - _previousBestValue);
 if ( _minFitnessDiffThreshold_enabled && (_k->currentGeneration > 1) && (range <= _minFitnessDiffThreshold) )
 {
  isFinished = true;
  koraliLog(KORALI_MINIMAL, "Function value differences (%+6.3e) < (%+6.3e)\n",  range, _minFitnessDiffThreshold);
 }

 size_t idx;
 
 if ( _minStandardDeviation_enabled )
 {
  size_t cTemp = 0;
  for(idx = 0; idx <_k->N; ++idx )
   cTemp += (_sigma * sqrt(_covarianceMatrix[idx*_k->N+idx]) < _minStandardDeviation * _k->_variables[idx]->_initialStandardDeviation) ? 1 : 0;
  
  if (cTemp == _k->N) {
   isFinished = true;
   koraliLog(KORALI_MINIMAL, "Object variable changes < %+6.3e\n", _minStandardDeviation * _k->_variables[idx]->_initialStandardDeviation);
  }

  for(idx = 0; idx <_k->N; ++idx )
   if ( _maxStandardDeviation_enabled && (_sigma * sqrt(_covarianceMatrix[idx*_k->N+idx]) > _maxStandardDeviation * _k->_variables[idx]->_initialStandardDeviation) )
   {
    isFinished = true;
    koraliLog(KORALI_MINIMAL, "Standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n", 
            _maxStandardDeviation * _k->_variables[idx]->_initialStandardDeviation);
    break;
   }
 }

 if ( _maxConditionCovarianceMatrix_enabled && (_maximumCovarianceEigenvalue >= _minimumCovarianceEigenvalue * _maxConditionCovarianceMatrix) )
 {
   isFinished = true;
   koraliLog(KORALI_MINIMAL, "Maximal condition number %7.2e reached. _maximumCovarianceEigenvalue=%7.2e, minEig=%7.2e, _maximumDiagonalCovarianceMatrixElement=%7.2e, _minimumDiagonalCovarianceMatrixElement=%7.2e\n",
                                _maxConditionCovarianceMatrix, _maximumCovarianceEigenvalue, _minimumCovarianceEigenvalue, _maximumDiagonalCovarianceMatrixElement, _minimumDiagonalCovarianceMatrixElement);
 }

 double fac;
 size_t iAchse = 0;
 size_t iKoo = 0;
 /* Component of _meanUpdate is not changed anymore */
 if( _minStandardDeviationStepFactor_enabled)
 if (!_isDiagonal )
 {
    for (iAchse = 0; iAchse < _k->N; ++iAchse)
    {
    fac = _minStandardDeviationStepFactor * _sigma * _axisLengths[iAchse];
    for (iKoo = 0; iKoo < _k->N; ++iKoo){
      if (_meanUpdate[iKoo] != _meanUpdate[iKoo] + fac * _covarianceEigenvectorMatrix[iKoo*_k->N+iAchse])
      break;
    }
    if (iKoo == _k->N)
    {
      isFinished = true;
      koraliLog(KORALI_MINIMAL, "Standard deviation %f*%7.2e in principal axis %ld without effect.\n", _minStandardDeviationStepFactor, _sigma*_axisLengths[iAchse], iAchse);
      break;
    }
  }
 }

 /* Component of _meanUpdate is not changed anymore */
 if( _minStandardDeviationStepFactor_enabled )
 for (iKoo = 0; iKoo < _k->N; ++iKoo)
 {
  if (_meanUpdate[iKoo] == _meanUpdate[iKoo] + _minStandardDeviationStepFactor*_sigma*sqrt(_covarianceMatrix[iKoo*_k->N+iKoo]) )
  {
   isFinished = true;
   koraliLog(KORALI_MINIMAL, "Standard deviation %f*%7.2e in coordinate %ld without effect.\n", _minStandardDeviationStepFactor, _sigma*sqrt(_covarianceMatrix[iKoo*_k->N+iKoo]), iKoo);
   break;
  }
 }

 return isFinished;
}


void CCMAES::updateEigensystem(std::vector<double>& M)
{
 eigen(_k->N, M, _temporaryAxisLengths, _auxiliarCovarianceEigenvectorMatrix);
 
 /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
 double minEWtmp = *std::min_element(std::begin(_temporaryAxisLengths), std::end(_temporaryAxisLengths));
 double maxEWtmp = *std::max_element(std::begin(_temporaryAxisLengths), std::end(_temporaryAxisLengths));

 if (minEWtmp <= 0.0) 
 { koraliWarning(KORALI_DETAILED, "Min Eigenvalue smaller or equal 0.0 (%+6.3e) after Eigen decomp (no update possible).\n", minEWtmp ); return; }

 for (size_t d = 0; d < _k->N; ++d) 
 {
     _temporaryAxisLengths[d] = sqrt(_temporaryAxisLengths[d]);
     if (std::isfinite(_temporaryAxisLengths[d]) == false)
     {
       koraliWarning(KORALI_DETAILED, "Could not calculate root of Eigenvalue (%+6.3e) after Eigen decomp (no update possible).\n", _temporaryAxisLengths[d] );
       return; 
     }
    for (size_t e = 0; e < _k->N; ++e) if (std::isfinite(_covarianceEigenvectorMatrix[d*_k->N+e]) == false)
    {
      koraliWarning(KORALI_DETAILED, "Non finite value detected in B (no update possible).\n");
       return;
    }
 }
 
 /* write back */
 for (size_t d = 0; d < _k->N; ++d) _axisLengths[d] = _temporaryAxisLengths[d];
 _covarianceEigenvectorMatrix.assign(std::begin(_auxiliarCovarianceEigenvectorMatrix), std::end(_auxiliarCovarianceEigenvectorMatrix));

 _minimumCovarianceEigenvalue = minEWtmp;
 _maximumCovarianceEigenvalue = maxEWtmp;
}


/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/


void CCMAES::eigen(size_t size, std::vector<double>& M,  std::vector<double>& diag, std::vector<double>& Q) const
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


void CCMAES::sort_index(const std::vector<double>& vec, std::vector<size_t>& _sortingIndex, size_t n) const
{
  // initialize original _sortingIndex locations
  std::iota(std::begin(_sortingIndex), std::begin(_sortingIndex)+n, (size_t) 0);

  // sort indexes based on comparing values in vec
  std::sort(std::begin(_sortingIndex), std::begin(_sortingIndex)+n, [vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; } );

}


void CCMAES::printGeneration()
{
 koraliLog(KORALI_MINIMAL, "CCMAES Generation %zu\n", _k->currentGeneration);

 if ( _constraintsDefined && _isViabilityRegime)
 {
   koraliLog(KORALI_NORMAL, "Searching start (MeanX violates constraints) .. \n");
   koraliLog(KORALI_NORMAL, "Viability Bounds:\n");
   for (size_t c = 0; c < _k->_constraints.size(); c++) koraliLogData(KORALI_NORMAL, "         %s = (%+6.3e)\n", _k->_variables[c]->_name.c_str(), _viabilityBoundaries[c]);
   koraliLog(KORALI_NORMAL, "\n");
 }

 koraliLog(KORALI_NORMAL, "Sigma:                        %+6.3e\n", _sigma);
 koraliLog(KORALI_NORMAL, "Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
 koraliLog(KORALI_NORMAL, "Diagonal Covariance:    Min = %+6.3e -  Max = %+6.3e\n", _minimumDiagonalCovarianceMatrixElement, _maximumDiagonalCovarianceMatrixElement);
 koraliLog(KORALI_NORMAL, "Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", _minimumCovarianceEigenvalue, _maximumCovarianceEigenvalue);

  koraliLog(KORALI_DETAILED, "Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _k->N; d++) koraliLogData(KORALI_DETAILED, "         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _meanUpdate[d], _bestEverCoordinates[d]);

  koraliLog(KORALI_DETAILED, "Constraint Evaluation at Current Function Value:\n");
  if ( _constraintsDefined )
  {
   if ( _bestValidSample >= 0 )
      for (size_t c = 0; c < _k->_constraints.size(); c++) koraliLogData(KORALI_DETAILED, "         ( %+6.3e )\n", _constraintEvaluations[c][_bestValidSample]);
   else
      for (size_t c = 0; c < _k->_constraints.size(); c++) koraliLogData(KORALI_DETAILED, "         ( %+6.3e )\n", _constraintEvaluations[c][0]);
  }

  koraliLog(KORALI_DETAILED, "Covariance Matrix:\n");
  for (size_t d = 0; d < _k->N; d++)
  {
   for (size_t e = 0; e <= d; e++) koraliLogData(KORALI_DETAILED, "   %+6.3e  ",_covarianceMatrix[d*_k->N+e]);
   koraliLog(KORALI_DETAILED, "\n");
  }

  koraliLog(KORALI_DETAILED, "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
  if ( _constraintsDefined )
  {
    koraliLog(KORALI_DETAILED, "Number of Constraint Evaluations: %zu\n", _constraintEvaluationCount);
    koraliLog(KORALI_DETAILED, "Number of Matrix Corrections: %zu\n", _covarianceMatrixAdaptationCount );
  }

  koraliLog(KORALI_NORMAL, "--------------------------------------------------------------------\n");
}

void CCMAES::finalize()
{
  koraliLog(KORALI_MINIMAL, "CCMAES Finished\n");
  koraliLog(KORALI_MINIMAL, "Optimum found: %e\n", _bestEverValue);
  koraliLog(KORALI_MINIMAL, "Optimum found at:\n");
  for (size_t d = 0; d < _k->N; ++d) koraliLogData(KORALI_MINIMAL, "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverCoordinates[d]);
  if ( _constraintsDefined )
  {
    koraliLog(KORALI_MINIMAL, "Constraint Evaluation at Optimum:\n");
    for (size_t c = 0; c < _k->_constraints.size(); c++)
      koraliLogData(KORALI_MINIMAL, "         ( %+6.3e )\n", _bestConstraintEvaluations[c]);
  }
  koraliLog(KORALI_MINIMAL, "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
  koraliLog(KORALI_MINIMAL, "--------------------------------------------------------------------\n");
}
