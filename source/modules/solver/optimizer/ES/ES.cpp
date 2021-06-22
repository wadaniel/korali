#include "engine.hpp"
#include "modules/solver/optimizer/ES/ES.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace optimizer
{


void ES::setInitialConfiguration()
{
  knlohmann::json problemConfig = (*_k)["Problem"];
  _variableCount = _k->_variables.size();

  // Establishing optimization goal
  _bestEverValue = -std::numeric_limits<double>::infinity();

  _previousBestEverValue = _bestEverValue;
  _previousBestValue = _bestEverValue;
  _currentBestValue = _bestEverValue;

  if (_populationSize == 0) _populationSize = ceil(4.0 + floor(3 * log((double)_variableCount)));

  // Allocating Memory
  _samplePopulation.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _samplePopulation[i].resize(_variableCount);

  _currentMean.resize(_variableCount);
  _previousMean.resize(_variableCount);
  _bestEverVariables.resize(_variableCount);
  _currentBestVariables.resize(_variableCount);
  _valueVector.resize(_populationSize);

  if(_mirroredSampling)
  {
    if(_populationSize % 2 == 1) KORALI_LOG_ERROR("Mirrored Sampling can only be applied with an even Sample Population (is %zu)", _populationSize);
  }

  _covarianceMatrix.resize(_variableCount * _variableCount);

  // Initializing variable defaults
  for (size_t i = 0; i < _variableCount; ++i)
  {
    if (std::isfinite(_k->_variables[i]->_initialValue) == false)
    {
      if (std::isfinite(_k->_variables[i]->_lowerBound) == false) KORALI_LOG_ERROR("Initial (Mean) Value of variable \'%s\' not defined, and cannot be inferred because variable lower bound is not finite.\n", _k->_variables[i]->_name.c_str());
      if (std::isfinite(_k->_variables[i]->_upperBound) == false) KORALI_LOG_ERROR("Initial (Mean) Value of variable \'%s\' not defined, and cannot be inferred because variable upper bound is not finite.\n", _k->_variables[i]->_name.c_str());
      _k->_variables[i]->_initialValue = (_k->_variables[i]->_upperBound + _k->_variables[i]->_lowerBound) * 0.5;
    }

    if (std::isfinite(_k->_variables[i]->_initialStandardDeviation) == false)
    {
      if (std::isfinite(_k->_variables[i]->_lowerBound) == false) KORALI_LOG_ERROR("Initial (Mean) Value of variable \'%s\' not defined, and cannot be inferred because variable lower bound is not finite.\n", _k->_variables[i]->_name.c_str());
      if (std::isfinite(_k->_variables[i]->_upperBound) == false) KORALI_LOG_ERROR("Initial Standard Deviation \'%s\' not defined, and cannot be inferred because variable upper bound is not finite.\n", _k->_variables[i]->_name.c_str());
      _k->_variables[i]->_initialStandardDeviation = (_k->_variables[i]->_upperBound - _k->_variables[i]->_lowerBound) * 0.3;
    }
  }

  _infeasibleSampleCount = 0;
}

void ES::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

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

  // Gathering evaluations
  for (size_t i = 0; i < _populationSize; i++)
    _valueVector[i] = KORALI_GET(double, samples[i], "F(x)");

  updateDistribution();
}

void ES::prepareGeneration()
{
  for (size_t i = 0; i < _populationSize; ++i)
  {
    bool isFeasible;
    do
    {
      std::vector<double> rands(_variableCount);
      for(size_t d = 0; d < _variableCount; ++d) rands[d] = _normalGenerator->getRandomNumber();
      sampleSingle(i, rands);

      isFeasible = isSampleFeasible(_samplePopulation[i]);

      _infeasibleSampleCount += isFeasible ? 0 : 1;

    } while (isFeasible == false && (_infeasibleSampleCount < _maxInfeasibleResamplings));
  }
}

void ES::sampleSingle(size_t sampleIdx, const std::vector<double>& randomNumbers)
{
    // TODO
}

void ES::updateDistribution()
{
  /* Generate _sortingIndex */
  sort_index(_valueVector, _sortingIndex, _populationSize);

  /* update function value history */
  _previousBestValue = _currentBestValue;

  /* update current best */
  _currentBestValue = _valueVector[_sortingIndex[0]];

  _currentBestVariables = _samplePopulation[_sortingIndex[0]];

  /* update xbestever */
  if (_currentBestValue > _bestEverValue || _k->_currentGeneration == 1)
  {
    _previousBestEverValue = _bestEverValue;
    _bestEverValue = _currentBestValue;
    _bestEverVariables = _currentBestVariables;

  }

  // TODO
}

void ES::sort_index(const std::vector<double> &vec, std::vector<size_t> &sortingIndex, size_t N) const
{
  // initialize original sortingIndex locations
  std::iota(std::begin(sortingIndex), std::begin(sortingIndex) + N, (size_t)0);

  // sort indexes based on comparing values in vec
  std::sort(std::begin(sortingIndex), std::begin(sortingIndex) + N, [vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; });
}

void ES::printGenerationBefore() { return; }

void ES::printGenerationAfter()
{
  _k->_logger->logInfo("Normal", "Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
  _k->_logger->logInfo("Normal", "Diagonal Covariance:    Min = %+6.3e -  Max = %+6.3e\n", _minimumDiagonalCovarianceMatrixElement, _maximumDiagonalCovarianceMatrixElement);

  _k->_logger->logInfo("Detailed", "Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _variableCount; d++) _k->_logger->logData("Detailed", "         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _currentMean[d], _bestEverVariables[d]);

  _k->_logger->logInfo("Detailed", "Covariance Matrix:\n");
  for (size_t d = 0; d < _variableCount; d++)
  {
    for (size_t e = 0; e <= d; e++) _k->_logger->logData("Detailed", "   %+6.3e  ", _covarianceMatrix[d * _variableCount + e]);
    _k->_logger->logInfo("Detailed", "\n");
  }

  _k->_logger->logInfo("Detailed", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
}

void ES::finalize()
{
  // Updating Results
  (*_k)["Results"]["Best Sample"]["F(x)"] = _bestEverValue;
  (*_k)["Results"]["Best Sample"]["Parameters"] = _bestEverVariables;

  _k->_logger->logInfo("Minimal", "Optimum found at:\n");
  for (size_t d = 0; d < _variableCount; ++d) _k->_logger->logData("Minimal", "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverVariables[d]);
  _k->_logger->logInfo("Minimal", "Optimum found: %e\n", _bestEverValue);
  _k->_logger->logInfo("Minimal", "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
}

void ES::setConfiguration(knlohmann::json& js) 
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

 if (isDefined(js, "Sorting Index"))
 {
 try { _sortingIndex = js["Sorting Index"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Sorting Index']\n%s", e.what()); } 
   eraseValue(js, "Sorting Index");
 }

 if (isDefined(js, "Value Vector"))
 {
 try { _valueVector = js["Value Vector"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Value Vector']\n%s", e.what()); } 
   eraseValue(js, "Value Vector");
 }

 if (isDefined(js, "Sample Population"))
 {
 try { _samplePopulation = js["Sample Population"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Sample Population']\n%s", e.what()); } 
   eraseValue(js, "Sample Population");
 }

 if (isDefined(js, "Finished Sample Count"))
 {
 try { _finishedSampleCount = js["Finished Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Finished Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Finished Sample Count");
 }

 if (isDefined(js, "Current Best Variables"))
 {
 try { _currentBestVariables = js["Current Best Variables"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Current Best Variables']\n%s", e.what()); } 
   eraseValue(js, "Current Best Variables");
 }

 if (isDefined(js, "Best Ever Variables"))
 {
 try { _bestEverVariables = js["Best Ever Variables"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Best Ever Variables']\n%s", e.what()); } 
   eraseValue(js, "Best Ever Variables");
 }

 if (isDefined(js, "Previous Best Value"))
 {
 try { _previousBestValue = js["Previous Best Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Previous Best Value']\n%s", e.what()); } 
   eraseValue(js, "Previous Best Value");
 }

 if (isDefined(js, "Best Sample Index"))
 {
 try { _bestSampleIndex = js["Best Sample Index"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Best Sample Index']\n%s", e.what()); } 
   eraseValue(js, "Best Sample Index");
 }

 if (isDefined(js, "Previous Best Ever Value"))
 {
 try { _previousBestEverValue = js["Previous Best Ever Value"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Previous Best Ever Value']\n%s", e.what()); } 
   eraseValue(js, "Previous Best Ever Value");
 }

 if (isDefined(js, "Covariance Matrix"))
 {
 try { _covarianceMatrix = js["Covariance Matrix"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Covariance Matrix']\n%s", e.what()); } 
   eraseValue(js, "Covariance Matrix");
 }

 if (isDefined(js, "Current Mean"))
 {
 try { _currentMean = js["Current Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Current Mean']\n%s", e.what()); } 
   eraseValue(js, "Current Mean");
 }

 if (isDefined(js, "Previous Mean"))
 {
 try { _previousMean = js["Previous Mean"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Previous Mean']\n%s", e.what()); } 
   eraseValue(js, "Previous Mean");
 }

 if (isDefined(js, "Infeasible Sample Count"))
 {
 try { _infeasibleSampleCount = js["Infeasible Sample Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Infeasible Sample Count']\n%s", e.what()); } 
   eraseValue(js, "Infeasible Sample Count");
 }

 if (isDefined(js, "Maximum Diagonal Covariance Matrix Element"))
 {
 try { _maximumDiagonalCovarianceMatrixElement = js["Maximum Diagonal Covariance Matrix Element"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Maximum Diagonal Covariance Matrix Element']\n%s", e.what()); } 
   eraseValue(js, "Maximum Diagonal Covariance Matrix Element");
 }

 if (isDefined(js, "Minimum Diagonal Covariance Matrix Element"))
 {
 try { _minimumDiagonalCovarianceMatrixElement = js["Minimum Diagonal Covariance Matrix Element"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Minimum Diagonal Covariance Matrix Element']\n%s", e.what()); } 
   eraseValue(js, "Minimum Diagonal Covariance Matrix Element");
 }

 if (isDefined(js, "Population Size"))
 {
 try { _populationSize = js["Population Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Population Size']\n%s", e.what()); } 
   eraseValue(js, "Population Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Population Size'] required by ES.\n"); 

 if (isDefined(js, "Diagonal Covariance"))
 {
 try { _diagonalCovariance = js["Diagonal Covariance"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Diagonal Covariance']\n%s", e.what()); } 
   eraseValue(js, "Diagonal Covariance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Diagonal Covariance'] required by ES.\n"); 

 if (isDefined(js, "Mirrored Sampling"))
 {
 try { _mirroredSampling = js["Mirrored Sampling"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Mirrored Sampling']\n%s", e.what()); } 
   eraseValue(js, "Mirrored Sampling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mirrored Sampling'] required by ES.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Infeasible Resamplings"))
 {
 try { _maxInfeasibleResamplings = js["Termination Criteria"]["Max Infeasible Resamplings"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Termination Criteria']['Max Infeasible Resamplings']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Infeasible Resamplings");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Infeasible Resamplings'] required by ES.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Optimizer::setConfiguration(js);
 _type = "optimizer/ES";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: ES: \n%s\n", js.dump(2).c_str());
} 

void ES::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Population Size"] = _populationSize;
   js["Diagonal Covariance"] = _diagonalCovariance;
   js["Mirrored Sampling"] = _mirroredSampling;
   js["Termination Criteria"]["Max Infeasible Resamplings"] = _maxInfeasibleResamplings;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Sorting Index"] = _sortingIndex;
   js["Value Vector"] = _valueVector;
   js["Sample Population"] = _samplePopulation;
   js["Finished Sample Count"] = _finishedSampleCount;
   js["Current Best Variables"] = _currentBestVariables;
   js["Best Ever Variables"] = _bestEverVariables;
   js["Previous Best Value"] = _previousBestValue;
   js["Best Sample Index"] = _bestSampleIndex;
   js["Previous Best Ever Value"] = _previousBestEverValue;
   js["Covariance Matrix"] = _covarianceMatrix;
   js["Current Mean"] = _currentMean;
   js["Previous Mean"] = _previousMean;
   js["Infeasible Sample Count"] = _infeasibleSampleCount;
   js["Maximum Diagonal Covariance Matrix Element"] = _maximumDiagonalCovarianceMatrixElement;
   js["Minimum Diagonal Covariance Matrix Element"] = _minimumDiagonalCovarianceMatrixElement;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Optimizer::getConfiguration(js);
} 

void ES::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Population Size\": 0, \"Diagonal Covariance\": false, \"Mirrored Sampling\": false, \"Termination Criteria\": {\"Max Infeasible Resamplings\": 10000, \"Max Condition Covariance Matrix\": Infinity, \"Min Standard Deviation\": -Infinity, \"Max Standard Deviation\": Infinity}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}, \"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}, \"Best Ever Value\": -Infinity, \"Current Min Standard Deviation\": Infinity, \"Current Max Standard Deviation\": -Infinity}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Optimizer::applyModuleDefaults(js);
} 

void ES::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Optimizer::applyVariableDefaults();
} 

bool ES::checkTermination()
{
 bool hasFinished = false;

 if (_k->_currentGeneration > 1 && ((_maxInfeasibleResamplings > 0) && (_infeasibleSampleCount >= _maxInfeasibleResamplings)))
 {
  _terminationCriteria.push_back("ES['Max Infeasible Resamplings'] = " + std::to_string(_maxInfeasibleResamplings) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Optimizer::checkTermination();
 return hasFinished;
}



} //optimizer
} //solver
} //korali

