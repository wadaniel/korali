#include "engine.hpp"
#include "modules/solver/optimizer/ES/ES.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace optimizer
{
;

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
  _randomVector.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _randomVector[i].resize(_variableCount);


  _currentMean.resize(_variableCount);
  _previousMean.resize(_variableCount);
  _bestEverVariables.resize(_variableCount);
  _currentBestVariables.resize(_variableCount);
  _valueVector.resize(_populationSize);
  _weightVector.resize(_populationSize);

  _covarianceMatrix.resize(_variableCount * _variableCount);
  
  // Version specific checks
  if (_version == "ES-EM")
  {
    _versionId = 0;
  }
  else if (_version == "ES-EM-SGA-v1")
  {
    _versionId = 1;
  }
  else if (_version == "ES-EM-SGA-v2")
  {
    _versionId = 2;
  }
  else if (_version == "ES-EM-SGA-v3")
  {
    if(_populationSize % 2 == 1) KORALI_LOG_ERROR("Mirrored Sampling can only be applied with an even Sample Population (is %zu)", _populationSize);
    _versionId = 3;
  }
  else if (_version == "ES-EM-C")
  {
    _versionId = 4;
  }
  else
    KORALI_LOG_ERROR("Version '%s' not recognized. Available options are 'ES-EM', 'ES-SGA-v1', 'ES-SGA-v2', 'ES-SGA-v3' and 'ES-EM-C'", _version);


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

  // Set initial mean and covariance
  for(size_t d = 0; d < _variableCount; ++d) _currentMean[d] = _k->_variables[d]->_initialValue;
  for(size_t d = 0; d < _variableCount; ++d) _covarianceMatrix[d*_variableCount+d] = _k->_variables[d]->_initialStandardDeviation;
  
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
  std::vector<double> rands(_variableCount);
  for (size_t i = 0; i < _populationSize; ++i)
  {
      for(size_t d = 0; d < _variableCount; ++d) rands[d] = _normalGenerator->getRandomNumber();
      sampleSingle(i, rands);
  }
}

void ES::sampleSingle(size_t sampleIdx, const std::vector<double>& randomNumbers)
{
    // TODO
}

void ES::updateDistribution()
{
  // Update sorting index based on value vector
  sort_index(_valueVector, _sortingIndex, _populationSize);

  // Update current best value and variable
  _previousBestValue = _currentBestValue;
  _currentBestValue = _valueVector[_sortingIndex[0]];
  _currentBestVariables = _samplePopulation[_sortingIndex[0]];

  // Update best ever variables
  if (_currentBestValue > _bestEverValue || _k->_currentGeneration == 1)
  {
    _previousBestEverValue = _bestEverValue;
    _bestEverValue = _currentBestValue;
    _bestEverVariables = _currentBestVariables;
  }


  // Set weight vector
  if(_versionId == 0) // ES-EM
  {
    double sumOfValues = std::accumulate(_valueVector.begin(), _valueVector.end(), 0.);
    for(size_t i = 0; i < _populationSize; ++i)
        _weightVector[i] = _valueVector[i]/sumOfValues;
  }
  else if(_versionId == 1) // ES-SGA-v1
  {
    double sumOfValues = std::accumulate(_valueVector.begin(), _valueVector.end(), 0.);
    for(size_t i = 0; i < _populationSize; ++i)
        _weightVector[i] = _valueVector[i]/sumOfValues;
  }
  else if(_versionId == 2) // ES-SGA-v2
  {
    for(size_t i = 1; i < _populationSize; ++i)
        _weightVector[i] = _valueVector[i]-_valueVector[0];
  }
  else if(_versionId == 3) // ES-SGA-v3
  {
    for(size_t i = 0; i < _populationSize; i+=2)
        _weightVector[i] = 0.5*(_valueVector[i]-_valueVector[i+1]);
  }
  else if(_versionId == 4) // ES-EM-C
  {
    double sumOfValues = std::accumulate(_valueVector.begin(), _valueVector.end(), 0.);
    for(size_t i = 0; i < _populationSize; ++i)
        _weightVector[i] = _valueVector[i]/sumOfValues;
  }
  
  // Reset mean and covariance
  _previousMean = _currentMean;
  std::fill(_currentMean.begin(), _currentMean.end(), 0.f);
  std::fill(_covarianceMatrix.begin(), _covarianceMatrix.end(), 0.f);
  
  // Update mean and covariance
  for(size_t i = 0; i < _populationSize; ++i)
  {

    for(size_t d = 0; d < _variableCount; ++d)
    {
      _currentMean[d] += _weightVector[i]*_samplePopulation[i][d];
      for(size_t e = 0; e < d; ++e)
      {
        _covarianceMatrix[d*_variableCount+e] = _weightVector[i]*_samplePopulation[i][d]*_samplePopulation[i][e];
        _covarianceMatrix[e*_variableCount+d] = _covarianceMatrix[d*_variableCount+e];
      }
    _covarianceMatrix[d*_variableCount+d] = _weightVector[i]*_samplePopulation[i][d]*_samplePopulation[i][d];
    }
  }
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

 if (isDefined(js, "Weight Vector"))
 {
 try { _weightVector = js["Weight Vector"].get<std::vector<double>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Weight Vector']\n%s", e.what()); } 
   eraseValue(js, "Weight Vector");
 }

 if (isDefined(js, "Sample Population"))
 {
 try { _samplePopulation = js["Sample Population"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Sample Population']\n%s", e.what()); } 
   eraseValue(js, "Sample Population");
 }

 if (isDefined(js, "Random Vector"))
 {
 try { _randomVector = js["Random Vector"].get<std::vector<std::vector<double>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Random Vector']\n%s", e.what()); } 
   eraseValue(js, "Random Vector");
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

 if (isDefined(js, "Version Id"))
 {
 try { _versionId = js["Version Id"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Version Id']\n%s", e.what()); } 
   eraseValue(js, "Version Id");
 }

 if (isDefined(js, "Population Size"))
 {
 try { _populationSize = js["Population Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Population Size']\n%s", e.what()); } 
   eraseValue(js, "Population Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Population Size'] required by ES.\n"); 

 if (isDefined(js, "Version"))
 {
 try { _version = js["Version"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ ES ] \n + Key:    ['Version']\n%s", e.what()); } 
   eraseValue(js, "Version");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Version'] required by ES.\n"); 

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
   js["Version"] = _version;
   js["Termination Criteria"]["Max Infeasible Resamplings"] = _maxInfeasibleResamplings;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Sorting Index"] = _sortingIndex;
   js["Value Vector"] = _valueVector;
   js["Weight Vector"] = _weightVector;
   js["Sample Population"] = _samplePopulation;
   js["Random Vector"] = _randomVector;
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
   js["Version Id"] = _versionId;
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

;

} //optimizer
} //solver
} //korali
;
