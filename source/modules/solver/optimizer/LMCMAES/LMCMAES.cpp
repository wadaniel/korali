#include "engine.hpp"
#include "modules/solver/optimizer/LMCMAES/LMCMAES.hpp"
#include "sample/sample.hpp"

#include <algorithm> // std::sort
#include <chrono>
#include <numeric> // std::iota
#include <stdio.h>
#include <unistd.h>

namespace korali
{
namespace solver
{
namespace optimizer
{


void LMCMAES::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  // Establishing optimization goal
  _bestEverValue = -std::numeric_limits<double>::infinity();

  _previousBestValue = _bestEverValue;
  _currentBestValue = _bestEverValue;

  if (_targetDistanceCoefficients.size() == 0) _targetDistanceCoefficients = {double(_variableCount), 0.0, 0.0};

  if (_targetDistanceCoefficients.size() != 3)
    KORALI_LOG_ERROR("LMCMAES requires 3 parameters for 'Target Distance Coefficients' (%zu provided).\n", _targetDistanceCoefficients.size());

  if (_muValue == 0) _muValue = _populationSize / 2;
  if (_subsetSize == 0) _subsetSize = 4 + std::floor(3 * std::log(double(_variableCount)));
  if (_cumulativeCovariance == 0.0) _cumulativeCovariance = 1.0 / ((double)_subsetSize);
  if (_choleskyMatrixLearningRate == 0.0) _choleskyMatrixLearningRate = 1.0 / (10.0 * std::log((double)_variableCount + 1.0));
  if (_targetDistanceCoefficients.empty()) _targetDistanceCoefficients = {(double)_variableCount, 0.0, 0.0};
  if (_setUpdateInterval == 0) _setUpdateInterval = std::max(std::floor(std::log(_variableCount)), 1.0);

  _chiSquareNumber = sqrt((double)_variableCount) * (1. - 1. / (4. * _variableCount) + 1. / (21. * _variableCount * _variableCount));
  _sigmaExponentFactor = 0.0;
  _conjugateEvolutionPathL2Norm = 0.0;

  // Allocating Memory
  _samplePopulation.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _samplePopulation[i].resize(_variableCount);

  _evolutionPath.resize(_variableCount);
  _meanUpdate.resize(_variableCount);
  _currentMean.resize(_variableCount);
  _previousMean.resize(_variableCount);
  _bestEverVariables.resize(_variableCount);
  _currentBestVariables.resize(_variableCount);
  _randomVector.resize(_variableCount);
  _choleskyFactorVectorProduct.resize(_variableCount);
  _standardDeviation.resize(_variableCount);

  _muWeights.resize(_muValue);

  _sortingIndex.resize(_populationSize);
  _valueVector.resize(_populationSize);

  _evolutionPathWeights.resize(_subsetSize);
  _subsetHistory.resize(_subsetSize);
  _subsetUpdateTimes.resize(_subsetSize);

  std::fill(_evolutionPathWeights.begin(), _evolutionPathWeights.end(), 0.0);
  std::fill(_subsetHistory.begin(), _subsetHistory.end(), 0);
  std::fill(_subsetUpdateTimes.begin(), _subsetUpdateTimes.end(), 0);

  _inverseVectors.resize(_subsetSize);
  _evolutionPathHistory.resize(_subsetSize);
  for (size_t i = 0; i < _subsetSize; ++i)
  {
    _inverseVectors[i].resize(_variableCount);
    _evolutionPathHistory[i].resize(_variableCount);
    std::fill(_inverseVectors[i].begin(), _inverseVectors[i].end(), 0.0);
    std::fill(_evolutionPathHistory[i].begin(), _evolutionPathHistory[i].end(), 0.0);
  }

  // Initializing variable defaults
  for (size_t i = 0; i < _variableCount; i++)
  {
    /* init mean if not defined */
    if (std::isfinite(_k->_variables[i]->_initialValue) == false)
    {
      if (std::isfinite(_k->_variables[i]->_lowerBound) == false) KORALI_LOG_ERROR("Initial (Mean) Value of variable \'%s\' not defined, and cannot be inferred because variable lower bound is not finite.\n", _k->_variables[i]->_name.c_str());
      if (std::isfinite(_k->_variables[i]->_upperBound) == false) KORALI_LOG_ERROR("Initial (Mean) Value of variable \'%s\' not defined, and cannot be inferred because variable upper bound is not finite.\n", _k->_variables[i]->_name.c_str());
      _k->_variables[i]->_initialValue = (_k->_variables[i]->_upperBound + _k->_variables[i]->_lowerBound) * 0.5;
    }

    /* calculate stddevs */
    if (std::isfinite(_k->_variables[i]->_initialStandardDeviation) == false)
    {
      if ((std::isfinite(_k->_variables[i]->_lowerBound) && std::isfinite(_k->_variables[i]->_upperBound)) == false)
        KORALI_LOG_ERROR("Either Lower/Upper Bound or Initial Value of variable \'%s\' must be defined.\n", _k->_variables[i]->_name.c_str());
      _standardDeviation[i] = 0.3 * (_k->_variables[i]->_upperBound - _k->_variables[i]->_lowerBound);
    }
    else
      _standardDeviation[i] = _k->_variables[i]->_initialStandardDeviation;
  }

  _sigma = _initialSigma;

  if (_muType == "Linear")
    for (size_t i = 0; i < _muValue; i++) _muWeights[i] = _muValue - i;
  else if (_muType == "Equal")
    for (size_t i = 0; i < _muValue; i++) _muWeights[i] = 1.;
  else if (_muType == "Logarithmic")
    for (size_t i = 0; i < _muValue; i++) _muWeights[i] = log(std::max((double)_muValue, 0.5 * _populationSize) + 0.5) - log(i + 1.);
  else
    KORALI_LOG_ERROR("Invalid setting of Mu Type (%s) (Linear, Equal, and Logarithmic accepted).", _muType.c_str());

  if ((_randomNumberDistribution != "Normal") && (_randomNumberDistribution != "Uniform")) KORALI_LOG_ERROR("Invalid setting of Random Number Distribution (%s) (Normal and Uniform accepted).", _randomNumberDistribution.c_str());

  // Normalize weights vector and set mueff
  double s1 = 0.0;
  double s2 = 0.0;

  for (size_t i = 0; i < _muValue; i++)
  {
    s1 += _muWeights[i];
    s2 += _muWeights[i] * _muWeights[i];
  }
  _effectiveMu = s1 * s1 / s2;

  for (size_t i = 0; i < _muValue; i++) _muWeights[i] /= s1;

  if (_initialSigma <= 0.0)
    KORALI_LOG_ERROR("Invalid Initial Sigma (must be greater 0.0, is %lf).", _initialSigma);
  if (_cumulativeCovariance <= 0.0)
    KORALI_LOG_ERROR("Invalid Initial Cumulative Covariance (must be greater 0.0).");
  if (_sigmaCumulationFactor <= 0.0)
    KORALI_LOG_ERROR("Invalid Sigma Cumulative Covariance (must be greater 0.0).");
  if (_dampFactor <= 0.0)
    KORALI_LOG_ERROR("Invalid Damp Factor (must be greater 0.0).");
  if (_choleskyMatrixLearningRate <= 0.0 || _choleskyMatrixLearningRate > 1.0)
    KORALI_LOG_ERROR("Invalid Cholesky Matrix Learning Rate (must be in (0, 1], is %lf).", _choleskyMatrixLearningRate);

  _infeasibleSampleCount = 0;
  _sqrtInverseCholeskyRate = std::sqrt(1.0 - _choleskyMatrixLearningRate);

  for (size_t i = 0; i < _variableCount; i++) _currentMean[i] = _previousMean[i] = _k->_variables[i]->_initialValue;
}

void LMCMAES::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  prepareGeneration();

  // Initializing Sample Evaluation
  std::vector<Sample> samples(_populationSize);
  for (size_t i = 0; i < _populationSize; i++)
  {
    samples[i]["Module"] = "Problem";
    samples[i]["Operation"] = "Evaluate";
    samples[i]["Parameters"] = _samplePopulation[i];
    samples[i]["Sample Id"] = i;
    _modelEvaluationCount++;
    KORALI_START(samples[i]);
  }

  // Waiting for samples to finish
  KORALI_WAITALL(samples);
  updateDistribution(samples);
}

void LMCMAES::prepareGeneration()
{
  for (size_t i = 0; i < _populationSize; ++i)
  {
    bool isFeasible;
    do
    {
      sampleSingle(i);
      isFeasible = isSampleFeasible(_samplePopulation[i]);

      if (isFeasible == false) _infeasibleSampleCount++;

    } while (isFeasible == false);
  }
}

void LMCMAES::sampleSingle(size_t sampleIdx)
{
  if (_symmetricSampling || (sampleIdx % 2) == 0)
  {
    choleskyFactorUpdate(sampleIdx);
    for (size_t d = 0; d < _variableCount; ++d)
    {
      _samplePopulation[sampleIdx][d] = _currentMean[d] + _sigma * _standardDeviation[d] * _choleskyFactorVectorProduct[d];
    }
  }
  else
  {
    for (size_t d = 0; d < _variableCount; ++d)
      //_samplePopulation[sampleIdx][d] = 2.0*_currentMean[d] - _samplePopulation[sampleIdx-1][d]; // version from [Loshchilov2015]
      _samplePopulation[sampleIdx][d] = _currentMean[d] - _sigma * _standardDeviation[d] * _choleskyFactorVectorProduct[d]; // version from Loshchilov's code
  }
}

void LMCMAES::updateDistribution(std::vector<Sample> &samples)
{
  // Processing results
  for (size_t i = 0; i < _populationSize; i++)
    _valueVector[i] = KORALI_GET(double, samples[i], "F(x)");

  /* Generate _sortingIndex */
  sort_index(_valueVector, _sortingIndex);

  /* update current best */
  _previousBestValue = _currentBestValue;
  _currentBestValue = _valueVector[0];
  for (size_t d = 0; d < _variableCount; ++d) _currentBestVariables[d] = _samplePopulation[_sortingIndex[0]][d];

  /* update xbestever */
  if (_currentBestValue > _bestEverValue)
  {
    _bestEverValue = _currentBestValue;

    for (size_t d = 0; d < _variableCount; ++d) _bestEverVariables[d] = _currentBestVariables[d];
  }

  /* set weights */
  for (size_t d = 0; d < _variableCount; ++d)
  {
    _previousMean[d] = _currentMean[d];
    _currentMean[d] = 0.;
    for (size_t i = 0; i < _muValue; ++i)
      _currentMean[d] += _muWeights[i] * _samplePopulation[_sortingIndex[i]][d];

    _meanUpdate[d] = (_currentMean[d] - _previousMean[d]) / (_sigma * _standardDeviation[d]);
  }

  /* update evolution path */
  _conjugateEvolutionPathL2Norm = 0.0;
  for (size_t d = 0; d < _variableCount; ++d)
  {
    _evolutionPath[d] = (1. - _cumulativeCovariance) * _evolutionPath[d] + sqrt(_cumulativeCovariance * (2. - _cumulativeCovariance) * _effectiveMu) * _meanUpdate[d];
    _conjugateEvolutionPathL2Norm += std::pow(_evolutionPath[d], 2);
  }
  _conjugateEvolutionPathL2Norm = std::sqrt(_conjugateEvolutionPathL2Norm);

  /* update stored paths */
  if ((_k->_currentGeneration - 1) % _setUpdateInterval == 0)
  {
    updateSet();
    updateInverseVectors();
  }

  /* update sigma */
  updateSigma();

  /* numerical error management */
  numericalErrorTreatment();
}

void LMCMAES::choleskyFactorUpdate(size_t sampleIdx)
{
  /* randomly select subsetStartIndex */
  double ms = 4.0;
  if (sampleIdx == 0) ms *= 10;
  size_t subsetStartIndex = _subsetSize - std::min((size_t)std::floor(ms * std::abs(_normalGenerator->getRandomNumber())) + 1, _subsetSize);

  if (_randomNumberDistribution == "Normal")
    for (size_t d = 0; d < _variableCount; ++d) _randomVector[d] = _normalGenerator->getRandomNumber();
  else /* (_randomNumberDistribution == "Uniform" */
    for (size_t d = 0; d < _variableCount; ++d) _randomVector[d] = 2 * _uniformGenerator->getRandomNumber() - 1.0;

  for (size_t d = 0; d < _variableCount; ++d) _choleskyFactorVectorProduct[d] = _randomVector[d];

  for (size_t i = subsetStartIndex; i < _subsetSize; ++i)
  {
    size_t idx = _subsetHistory[i];

    double k = 0.0;
    for (size_t d = 0; d < _variableCount; ++d) k += _inverseVectors[idx][d] * _randomVector[d];
    k *= _evolutionPathWeights[idx];

    _minCholeskyFactorVectorProductEntry = std::numeric_limits<double>::infinity();
    _maxCholeskyFactorVectorProductEntry = -std::numeric_limits<double>::infinity();
    for (size_t d = 0; d < _variableCount; ++d)
    {
      _choleskyFactorVectorProduct[d] = _sqrtInverseCholeskyRate * _choleskyFactorVectorProduct[d] + k * _evolutionPathHistory[idx][d];
      if (_choleskyFactorVectorProduct[d] < _minCholeskyFactorVectorProductEntry) _minCholeskyFactorVectorProductEntry = _choleskyFactorVectorProduct[d];
      if (_choleskyFactorVectorProduct[d] > _maxCholeskyFactorVectorProductEntry) _maxCholeskyFactorVectorProductEntry = _choleskyFactorVectorProduct[d];
    }
  }
}

void LMCMAES::updateSet()
{
  size_t t = std::floor(double(_k->_currentGeneration - 1.0) / double(_setUpdateInterval));

  if (t < _subsetSize)
  {
    _replacementIndex = t;
    _subsetHistory[t] = t;
    _subsetUpdateTimes[t] = t * _setUpdateInterval + 1;
  }
  else
  {
    double tmparg = 0.0, minarg, target;
    minarg = std::numeric_limits<double>::max();
    for (size_t i = 1; i < _subsetSize; ++i)
    {
      /* `target` by default equals _variableCount */
      target = _targetDistanceCoefficients[0] + _targetDistanceCoefficients[1] * std::pow(double(i + 1.) / double(_subsetSize), _targetDistanceCoefficients[2]);
      tmparg = _subsetUpdateTimes[_subsetHistory[i]] - _subsetUpdateTimes[_subsetHistory[i - 1]] - target;
      if (tmparg < minarg)
      {
        minarg = tmparg;
        _replacementIndex = i;
      }
    }
    if (tmparg > 0) _replacementIndex = 0; /* if all evolution paths at a distance of `target` or larger, update oldest */
    size_t jtmp = _subsetHistory[_replacementIndex];
    for (size_t i = _replacementIndex; i < _subsetSize - 1; ++i) _subsetHistory[i] = _subsetHistory[i + 1];

    _subsetHistory[_subsetSize - 1] = jtmp;
    _subsetUpdateTimes[jtmp] = t * _setUpdateInterval + 1;
  }

  /* insert new evolution path */
  std::copy(_evolutionPath.begin(), _evolutionPath.end(), _evolutionPathHistory[_subsetHistory[_replacementIndex]].begin());
}

void LMCMAES::updateInverseVectors()
{
  double djt, k;
  double fac = std::sqrt(1.0 + _choleskyMatrixLearningRate / (1.0 - _choleskyMatrixLearningRate));

  /* update all inverse vectors and evolution path weights onwards from replacement index */
  for (size_t i = _replacementIndex; i < _subsetSize; ++i)
  {
    size_t idx = _subsetHistory[i];

    double v2L2 = 0.0;
    for (size_t d = 0; d < _variableCount; ++d) v2L2 += _inverseVectors[idx][d] * _inverseVectors[idx][d];

    k = 0.0;
    if (v2L2 > 0.0)
    {
      djt = _sqrtInverseCholeskyRate / v2L2 * (1.0 - 1.0 / (fac * std::sqrt(v2L2)));

      k = 0.0;
      for (size_t d = 0; d < _variableCount; ++d) k += _inverseVectors[idx][d] * _evolutionPathHistory[idx][d];
      k *= djt;

      _evolutionPathWeights[idx] = _sqrtInverseCholeskyRate / v2L2 * (std::sqrt(1.0 + _choleskyMatrixLearningRate / (1.0 - _choleskyMatrixLearningRate) * v2L2) - 1.0);
    }

    for (size_t d = 0; d < _variableCount; ++d)
      _inverseVectors[idx][d] = _sqrtInverseCholeskyRate * _evolutionPathHistory[idx][d] - k * _inverseVectors[idx][d];
  }
}

void LMCMAES::updateSigma()
{
  _sigma *= exp(_sigmaCumulationFactor / _dampFactor * (_conjugateEvolutionPathL2Norm / _chiSquareNumber - 1.));

  /* escape flat evaluation */
  if (_currentBestValue == _valueVector[_sortingIndex[(int)_muValue]])
  {
    _sigma *= exp(0.2 + _sigmaCumulationFactor / _dampFactor);
    _k->_logger->logWarning("Detailed", "Sigma increased due to equal function values.\n");
  }

  /* upper bound check for _sigma */
  if (_sigma > 2.0 * _initialSigma)
  {
    _k->_logger->logInfo("Detailed", "Sigma exceeding initial sigma by a factor of two (%f > %f), increase value of Initial Sigma.\n", _sigma, 2.0 * _initialSigma);
    if (_isSigmaBounded)
    {
      _sigma = 2.0 * _initialSigma;
      _k->_logger->logInfo("Detailed", "Sigma set to upper bound (%f) due to solver configuration 'Is Sigma Bounded' = 'true'.\n", _sigma);
    }
  }
}

void LMCMAES::numericalErrorTreatment()
{
  //treat numerical precision provblems
  //TODO
}

/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/

void LMCMAES::sort_index(const std::vector<double> &vec, std::vector<size_t> &sortingIndex) const
{
  // initialize original _sortingIndex locations
  std::iota(std::begin(sortingIndex), std::end(sortingIndex), (size_t)0);

  // sort indexes based on comparing values in vec
  std::sort(std::begin(sortingIndex), std::end(sortingIndex), [vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; });
}

void LMCMAES::printGenerationBefore() { return; }

void LMCMAES::printGenerationAfter()
{
  _k->_logger->logInfo("Normal", "Sigma:                        %+6.3e\n", _sigma);
  _k->_logger->logInfo("Normal", "Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
  _k->_logger->logInfo("Normal", "Cholesky Factor:        Min = %+6.3e -  Max = %+6.3e\n", _minCholeskyFactorVectorProductEntry, _maxCholeskyFactorVectorProductEntry);
  _k->_logger->logInfo("Normal", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);

  _k->_logger->logInfo("Detailed", "Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _variableCount; d++) _k->_logger->logData("Detailed", "         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _currentMean[d], _bestEverVariables[d]);

  _k->_logger->logInfo("Detailed", "Covariance Matrix:\n");
}

void LMCMAES::finalize()
{
  // Updating Results
  (*_k)["Results"]["Best Sample"]["F(x)"] = _bestEverValue;
  (*_k)["Results"]["Best Sample"]["Parameters"] = _bestEverVariables;

  _k->_logger->logInfo("Minimal", "Optimum found: %e\n", _bestEverValue);
  _k->_logger->logInfo("Minimal", "Optimum found at:\n");
  for (size_t d = 0; d < _variableCount; ++d) _k->_logger->logData("Minimal", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverVariables[d]);
  _k->_logger->logInfo("Minimal", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
}

void LMCMAES::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Normal Generator"))
 {
 _normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(js["Normal Generator"], _k));
 _normalGenerator->applyVariableDefaults();
 _normalGenerator->applyModuleDefaults(js["Normal Generator"]);
 _normalGenerator->setConfiguration(js["Normal Generator"]);
   eraseValue(js, "Normal Generator");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Value Vector"))
 {
 try { _valueVector = js["Value Vector"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Value Vector']\n%s", e.what()); } 
   eraseValue(js, "Value Vector");
 }

 if (isDefined(js, "Mu Weights"))
 {
 try { _muWeights = js["Mu Weights"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Mu Weights']\n%s", e.what()); } 
   eraseValue(js, "Mu Weights");
 }

 if (isDefined(js, "Effective Mu"))
 {
 try { _effectiveMu = js["Effective Mu"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Effective Mu']\n%s", e.what()); } 
   eraseValue(js, "Effective Mu");
 }

 if (isDefined(js, "Sigma Exponent Factor"))
 {
 try { _sigmaExponentFactor = js["Sigma Exponent Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Sigma Exponent Factor']\n%s", e.what()); } 
   eraseValue(js, "Sigma Exponent Factor");
 }

 if (isDefined(js, "Sigma"))
 {
 try { _sigma = js["Sigma"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Sigma']\n%s", e.what()); } 
   eraseValue(js, "Sigma");
 }

 if (isDefined(js, "Sample Population"))
 {
 try { _samplePopulation = js["Sample Population"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Sample Population']\n%s", e.what()); } 
   eraseValue(js, "Sample Population");
 }

 if (isDefined(js, "Finished Sample Count"))
 {
 try { _finishedSampleCount = js["Finished Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Finished Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Finished Sample Count");
 }

 if (isDefined(js, "Previous Best Value"))
 {
 try { _previousBestValue = js["Previous Best Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Previous Best Value']\n%s", e.what()); } 
   eraseValue(js, "Previous Best Value");
 }

 if (isDefined(js, "Current Best Variables"))
 {
 try { _currentBestVariables = js["Current Best Variables"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Current Best Variables']\n%s", e.what()); } 
   eraseValue(js, "Current Best Variables");
 }

 if (isDefined(js, "Best Sample Index"))
 {
 try { _bestSampleIndex = js["Best Sample Index"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Best Sample Index']\n%s", e.what()); } 
   eraseValue(js, "Best Sample Index");
 }

 if (isDefined(js, "Sorting Index"))
 {
 try { _sortingIndex = js["Sorting Index"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Sorting Index']\n%s", e.what()); } 
   eraseValue(js, "Sorting Index");
 }

 if (isDefined(js, "Random Vector"))
 {
 try { _randomVector = js["Random Vector"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Random Vector']\n%s", e.what()); } 
   eraseValue(js, "Random Vector");
 }

 if (isDefined(js, "Replacement Index"))
 {
 try { _replacementIndex = js["Replacement Index"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Replacement Index']\n%s", e.what()); } 
   eraseValue(js, "Replacement Index");
 }

 if (isDefined(js, "Subset History"))
 {
 try { _subsetHistory = js["Subset History"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Subset History']\n%s", e.what()); } 
   eraseValue(js, "Subset History");
 }

 if (isDefined(js, "Subset Update Times"))
 {
 try { _subsetUpdateTimes = js["Subset Update Times"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Subset Update Times']\n%s", e.what()); } 
   eraseValue(js, "Subset Update Times");
 }

 if (isDefined(js, "Cholesky Factor Vector Product"))
 {
 try { _choleskyFactorVectorProduct = js["Cholesky Factor Vector Product"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Cholesky Factor Vector Product']\n%s", e.what()); } 
   eraseValue(js, "Cholesky Factor Vector Product");
 }

 if (isDefined(js, "Min Cholesky Factor Vector Product Entry"))
 {
 try { _minCholeskyFactorVectorProductEntry = js["Min Cholesky Factor Vector Product Entry"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Min Cholesky Factor Vector Product Entry']\n%s", e.what()); } 
   eraseValue(js, "Min Cholesky Factor Vector Product Entry");
 }

 if (isDefined(js, "Max Cholesky Factor Vector Product Entry"))
 {
 try { _maxCholeskyFactorVectorProductEntry = js["Max Cholesky Factor Vector Product Entry"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Max Cholesky Factor Vector Product Entry']\n%s", e.what()); } 
   eraseValue(js, "Max Cholesky Factor Vector Product Entry");
 }

 if (isDefined(js, "Evolution Path History"))
 {
 try { _evolutionPathHistory = js["Evolution Path History"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Evolution Path History']\n%s", e.what()); } 
   eraseValue(js, "Evolution Path History");
 }

 if (isDefined(js, "Inverse Vectors"))
 {
 try { _inverseVectors = js["Inverse Vectors"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Inverse Vectors']\n%s", e.what()); } 
   eraseValue(js, "Inverse Vectors");
 }

 if (isDefined(js, "Current Mean"))
 {
 try { _currentMean = js["Current Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Current Mean']\n%s", e.what()); } 
   eraseValue(js, "Current Mean");
 }

 if (isDefined(js, "Previous Mean"))
 {
 try { _previousMean = js["Previous Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Previous Mean']\n%s", e.what()); } 
   eraseValue(js, "Previous Mean");
 }

 if (isDefined(js, "Mean Update"))
 {
 try { _meanUpdate = js["Mean Update"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Mean Update']\n%s", e.what()); } 
   eraseValue(js, "Mean Update");
 }

 if (isDefined(js, "Evolution Path"))
 {
 try { _evolutionPath = js["Evolution Path"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Evolution Path']\n%s", e.what()); } 
   eraseValue(js, "Evolution Path");
 }

 if (isDefined(js, "Evolution Path Weights"))
 {
 try { _evolutionPathWeights = js["Evolution Path Weights"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Evolution Path Weights']\n%s", e.what()); } 
   eraseValue(js, "Evolution Path Weights");
 }

 if (isDefined(js, "Conjugate Evolution Path L2 Norm"))
 {
 try { _conjugateEvolutionPathL2Norm = js["Conjugate Evolution Path L2 Norm"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Conjugate Evolution Path L2 Norm']\n%s", e.what()); } 
   eraseValue(js, "Conjugate Evolution Path L2 Norm");
 }

 if (isDefined(js, "Infeasible Sample Count"))
 {
 try { _infeasibleSampleCount = js["Infeasible Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Infeasible Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Infeasible Sample Count");
 }

 if (isDefined(js, "Sqrt Inverse Cholesky Rate"))
 {
 try { _sqrtInverseCholeskyRate = js["Sqrt Inverse Cholesky Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Sqrt Inverse Cholesky Rate']\n%s", e.what()); } 
   eraseValue(js, "Sqrt Inverse Cholesky Rate");
 }

 if (isDefined(js, "Chi Square Number"))
 {
 try { _chiSquareNumber = js["Chi Square Number"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Chi Square Number']\n%s", e.what()); } 
   eraseValue(js, "Chi Square Number");
 }

 if (isDefined(js, "Standard Deviation"))
 {
 try { _standardDeviation = js["Standard Deviation"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Standard Deviation']\n%s", e.what()); } 
   eraseValue(js, "Standard Deviation");
 }

 if (isDefined(js, "Population Size"))
 {
 try { _populationSize = js["Population Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Population Size']\n%s", e.what()); } 
   eraseValue(js, "Population Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Population Size'] required by LMCMAES.\n"); 

 if (isDefined(js, "Mu Value"))
 {
 try { _muValue = js["Mu Value"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Mu Value']\n%s", e.what()); } 
   eraseValue(js, "Mu Value");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mu Value'] required by LMCMAES.\n"); 

 if (isDefined(js, "Mu Type"))
 {
 try { _muType = js["Mu Type"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Mu Type']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_muType == "Linear") validOption = true; 
 if (_muType == "Equal") validOption = true; 
 if (_muType == "Logarithmic") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mu Type'] required by LMCMAES.\n", _muType.c_str()); 
}
   eraseValue(js, "Mu Type");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mu Type'] required by LMCMAES.\n"); 

 if (isDefined(js, "Initial Sigma"))
 {
 try { _initialSigma = js["Initial Sigma"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Initial Sigma']\n%s", e.what()); } 
   eraseValue(js, "Initial Sigma");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Sigma'] required by LMCMAES.\n"); 

 if (isDefined(js, "Random Number Distribution"))
 {
 try { _randomNumberDistribution = js["Random Number Distribution"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Random Number Distribution']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_randomNumberDistribution == "Normal") validOption = true; 
 if (_randomNumberDistribution == "Uniform") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Random Number Distribution'] required by LMCMAES.\n", _randomNumberDistribution.c_str()); 
}
   eraseValue(js, "Random Number Distribution");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Random Number Distribution'] required by LMCMAES.\n"); 

 if (isDefined(js, "Symmetric Sampling"))
 {
 try { _symmetricSampling = js["Symmetric Sampling"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Symmetric Sampling']\n%s", e.what()); } 
   eraseValue(js, "Symmetric Sampling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Symmetric Sampling'] required by LMCMAES.\n"); 

 if (isDefined(js, "Sigma Cumulation Factor"))
 {
 try { _sigmaCumulationFactor = js["Sigma Cumulation Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Sigma Cumulation Factor']\n%s", e.what()); } 
   eraseValue(js, "Sigma Cumulation Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Sigma Cumulation Factor'] required by LMCMAES.\n"); 

 if (isDefined(js, "Damp Factor"))
 {
 try { _dampFactor = js["Damp Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Damp Factor']\n%s", e.what()); } 
   eraseValue(js, "Damp Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Damp Factor'] required by LMCMAES.\n"); 

 if (isDefined(js, "Is Sigma Bounded"))
 {
 try { _isSigmaBounded = js["Is Sigma Bounded"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Is Sigma Bounded']\n%s", e.what()); } 
   eraseValue(js, "Is Sigma Bounded");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Is Sigma Bounded'] required by LMCMAES.\n"); 

 if (isDefined(js, "Cumulative Covariance"))
 {
 try { _cumulativeCovariance = js["Cumulative Covariance"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Cumulative Covariance']\n%s", e.what()); } 
   eraseValue(js, "Cumulative Covariance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Cumulative Covariance'] required by LMCMAES.\n"); 

 if (isDefined(js, "Cholesky Matrix Learning Rate"))
 {
 try { _choleskyMatrixLearningRate = js["Cholesky Matrix Learning Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Cholesky Matrix Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Cholesky Matrix Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Cholesky Matrix Learning Rate'] required by LMCMAES.\n"); 

 if (isDefined(js, "Target Distance Coefficients"))
 {
 try { _targetDistanceCoefficients = js["Target Distance Coefficients"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Target Distance Coefficients']\n%s", e.what()); } 
   eraseValue(js, "Target Distance Coefficients");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Target Distance Coefficients'] required by LMCMAES.\n"); 

 if (isDefined(js, "Target Success Rate"))
 {
 try { _targetSuccessRate = js["Target Success Rate"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Target Success Rate']\n%s", e.what()); } 
   eraseValue(js, "Target Success Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Target Success Rate'] required by LMCMAES.\n"); 

 if (isDefined(js, "Set Update Interval"))
 {
 try { _setUpdateInterval = js["Set Update Interval"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Set Update Interval']\n%s", e.what()); } 
   eraseValue(js, "Set Update Interval");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Set Update Interval'] required by LMCMAES.\n"); 

 if (isDefined(js, "Subset Size"))
 {
 try { _subsetSize = js["Subset Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Subset Size']\n%s", e.what()); } 
   eraseValue(js, "Subset Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Subset Size'] required by LMCMAES.\n"); 

 if (isDefined(js, "Termination Criteria", "Min Value"))
 {
 try { _minValue = js["Termination Criteria"]["Min Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ LMCMAES ] \n + Key:    ['Termination Criteria']['Min Value']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Min Value");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Min Value'] required by LMCMAES.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Optimizer::setConfiguration(js);
 _type = "optimizer/LMCMAES";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: LMCMAES: \n%s\n", js.dump(2).c_str());
} 

void LMCMAES::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Population Size"] = _populationSize;
   js["Mu Value"] = _muValue;
   js["Mu Type"] = _muType;
   js["Initial Sigma"] = _initialSigma;
   js["Random Number Distribution"] = _randomNumberDistribution;
   js["Symmetric Sampling"] = _symmetricSampling;
   js["Sigma Cumulation Factor"] = _sigmaCumulationFactor;
   js["Damp Factor"] = _dampFactor;
   js["Is Sigma Bounded"] = _isSigmaBounded;
   js["Cumulative Covariance"] = _cumulativeCovariance;
   js["Cholesky Matrix Learning Rate"] = _choleskyMatrixLearningRate;
   js["Target Distance Coefficients"] = _targetDistanceCoefficients;
   js["Target Success Rate"] = _targetSuccessRate;
   js["Set Update Interval"] = _setUpdateInterval;
   js["Subset Size"] = _subsetSize;
   js["Termination Criteria"]["Min Value"] = _minValue;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Value Vector"] = _valueVector;
   js["Mu Weights"] = _muWeights;
   js["Effective Mu"] = _effectiveMu;
   js["Sigma Exponent Factor"] = _sigmaExponentFactor;
   js["Sigma"] = _sigma;
   js["Sample Population"] = _samplePopulation;
   js["Finished Sample Count"] = _finishedSampleCount;
   js["Previous Best Value"] = _previousBestValue;
   js["Current Best Variables"] = _currentBestVariables;
   js["Best Sample Index"] = _bestSampleIndex;
   js["Sorting Index"] = _sortingIndex;
   js["Random Vector"] = _randomVector;
   js["Replacement Index"] = _replacementIndex;
   js["Subset History"] = _subsetHistory;
   js["Subset Update Times"] = _subsetUpdateTimes;
   js["Cholesky Factor Vector Product"] = _choleskyFactorVectorProduct;
   js["Min Cholesky Factor Vector Product Entry"] = _minCholeskyFactorVectorProductEntry;
   js["Max Cholesky Factor Vector Product Entry"] = _maxCholeskyFactorVectorProductEntry;
   js["Evolution Path History"] = _evolutionPathHistory;
   js["Inverse Vectors"] = _inverseVectors;
   js["Current Mean"] = _currentMean;
   js["Previous Mean"] = _previousMean;
   js["Mean Update"] = _meanUpdate;
   js["Evolution Path"] = _evolutionPath;
   js["Evolution Path Weights"] = _evolutionPathWeights;
   js["Conjugate Evolution Path L2 Norm"] = _conjugateEvolutionPathL2Norm;
   js["Infeasible Sample Count"] = _infeasibleSampleCount;
   js["Sqrt Inverse Cholesky Rate"] = _sqrtInverseCholeskyRate;
   js["Chi Square Number"] = _chiSquareNumber;
   js["Standard Deviation"] = _standardDeviation;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Optimizer::getConfiguration(js);
} 

void LMCMAES::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Mu Value\": 0, \"Mu Type\": \"Logarithmic\", \"Initial Sigma\": 1.0, \"Random Number Distribution\": \"Normal\", \"Symmetric Sampling\": true, \"Sigma Cumulation Factor\": 0.3, \"Damp Factor\": 1.0, \"Is Sigma Bounded\": false, \"Cumulative Covariance\": 0.0, \"Cholesky Matrix Learning Rate\": 0.0, \"Target Distance Coefficients\": [], \"Target Success Rate\": 0.25, \"Set Update Interval\": 0, \"Subset Size\": 0, \"Termination Criteria\": {\"Min Value\": -Infinity}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Optimizer::applyModuleDefaults(js);
} 

void LMCMAES::applyVariableDefaults() 
{

 std::string defaultString = "{\"Initial Standard Deviation\": -Infinity, \"Minimum Standard Deviation Update\": 0.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Optimizer::applyVariableDefaults();
} 

bool LMCMAES::checkTermination()
{
 bool hasFinished = false;

 if (-_bestEverValue < _minValue)
 {
  _terminationCriteria.push_back("LMCMAES['Min Value'] = " + std::to_string(_minValue) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Optimizer::checkTermination();
 return hasFinished;
}



} //optimizer
} //solver
} //korali

