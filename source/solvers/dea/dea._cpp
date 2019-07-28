#include "korali.hpp"
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort
using namespace Korali::Solver;

constexpr size_t str2int(const char* str, int h = 0) { return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h]; }

DEA::DEA()
{
 // Initializing Generators
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"]  = "Gaussian";
 jsGaussian["Mean"]  = 0.0;
 jsGaussian["Standard Deviation"] = 1.0;
 jsGaussian["Seed"]  = _k->_seed++;
 _gaussianGenerator  = new Variable();
 _gaussianGenerator->setDistribution(jsGaussian);

 auto jsUniform = nlohmann::json();
 jsUniform["Type"] = "Uniform";
 jsUniform["Minimum"] = 0.0;
 jsUniform["Maximum"] = 1.0;
 jsUniform["Seed"] = _k->_seed++;

 _uniformGenerator = new Variable();
 _uniformGenerator->setDistribution(jsUniform);
}

DEA::~DEA()
{
  delete _uniformGenerator;
}

void DEA::initialize()
{
 // Checking for accepted problem types
  std::string pName = _k->_js["Problem"]["Type"];
  bool acceptableProblem = false;
  if (pName == "Optimization")  acceptableProblem = true;
  if (pName == "Bayesian Inference")  acceptableProblem = true;
  if (acceptableProblem == false) koraliError("DEA cannot solve problems of type: '%s'.\n", pName.c_str());

 // Allocating Memory
 _samplePopulation.resize(_k->N*_sampleCount);
 _sampleCandidates.resize(_k->N*_sampleCount);

 _previousMean.resize(_k->N);
 _sampleMeans.resize(_k->N);
 _bestEverCoordinates.resize(_k->N);
 _currentBestSample.resize(_k->N);
 _maxDistances.resize(_k->N);

 _isInitializedSample.resize(_sampleCount);
 _previousFitnessVector.resize(_sampleCount);
 _fitnessVector.resize(_sampleCount);

 _infeasibleSampleCount = 0;
 _bestSampleIndex             = 0;
 _previousBestValue     = -std::numeric_limits<double>::max();
 _currentBestValue      = -std::numeric_limits<double>::max();
 _previousBestEverValue = -std::numeric_limits<double>::max();
 _bestEverValue         = -std::numeric_limits<double>::max();

 for(size_t d = 0; d < _k->N; ++d)
  if(_k->_variables[d]->_upperBound < _k->_variables[d]->_lowerBound)
    koraliError("Lower Bound (%.4f) of variable \'%s\'  exceeds Upper Bound (%.4f).\n", _k->_variables[d]->_lowerBound, _k->_variables[d]->_name.c_str(), _k->_variables[d]->_upperBound);

 initSamples();

 for(size_t d = 0; d < _k->N; ++d) { _previousMean[d] = 0.0; _sampleMeans[d] = 0.0; }

 for(size_t i = 0; i < _sampleCount; ++i) for(size_t d = 0; d < _k->N; ++d) 
   _sampleMeans[d] += _samplePopulation[i*_k->N+d]/((double)_sampleCount);

}

void DEA::runGeneration()
{
 prepareGeneration();
 evaluateSamples();
 updateSolver();
}


void DEA::initSamples()
{
  for(size_t i = 0; i < _sampleCount; ++i) for(size_t d = 0; d < _k->N; ++d)
  {
    double width = _k->_variables[d]->_upperBound - _k->_variables[d]->_lowerBound;
    _samplePopulation[i*_k->N+d] = _k->_variables[d]->_lowerBound + width * _uniformGenerator->getRandomNumber();
  }
}


void DEA::prepareGeneration()
{
 size_t initial_infeasible = _infeasibleSampleCount;
 for (size_t i = 0; i < _sampleCount; ++i)
 {
  mutateSingle(i);
  while(isFeasible(i) == false)
  {
  _infeasibleSampleCount++;

  if (_fixInfeasible)
      fixInfeasible(i);
  else
      mutateSingle(i);

  if ( _maxResamplingsEnabled )
  if ( (_infeasibleSampleCount - initial_infeasible) > _maxResamplings )
  koraliWarning(KORALI_MINIMAL, "Exiting resampling loop (param %zu) because max resamplings (%zu) reached.\n", i, _maxResamplings);
  }
 }

 _finishedSampleCount = 0;
 for (size_t i = 0; i < _sampleCount; i++) _isInitializedSample[i] = false;

 _previousFitnessVector.assign(std::begin(_fitnessVector), std::end(_fitnessVector));
}


void DEA::mutateSingle(size_t sampleIdx)
{
    size_t a, b;
    do{ a = _uniformGenerator->getRandomNumber()*_sampleCount; } while(a == sampleIdx);
    do{ b = _uniformGenerator->getRandomNumber()*_sampleCount; } while(b == sampleIdx || b == a);

    if (_mutationRule == "Self Adaptive")
    {
        // Brest [2006]
        double tau1 = 0.1;
        double tau2 = 0.1;
        double Fl = 0.1;
        double Fu = 0.9;

        double rd2 = _uniformGenerator->getRandomNumber();
        double rd3 = _uniformGenerator->getRandomNumber();
        
        if(rd2 < tau1) 
        {
           double rd1 = _uniformGenerator->getRandomNumber();
            _mutationRate = Fl+rd1*Fu;
        } 
        if(rd3 < tau2)
        {
           double rd4 = _uniformGenerator->getRandomNumber();
           _crossoverRate = rd4;
        }
    }

    double* parent;
    if(_parentSelectionRule == "Random")
    {
        size_t c;
        do{ c = _uniformGenerator->getRandomNumber()*_sampleCount; } while(c == sampleIdx || c == a || c == b);
        parent = &_samplePopulation[c*_k->N];
    }
    else /* _parentSelectionRule == "Best" */
    {
        parent = &_samplePopulation[_bestSampleIndex*_k->N];
    }

    size_t rn = _uniformGenerator->getRandomNumber()*_k->N;
    for(size_t d = 0; d < _k->N; ++d)
    {
      if( (_uniformGenerator->getRandomNumber() < _crossoverRate) || (d == rn) )
          _sampleCandidates[sampleIdx*_k->N+d] = parent[d]+_mutationRate*(_samplePopulation[a*_k->N+d]-_samplePopulation[b*_k->N+d]);
      else
          _sampleCandidates[sampleIdx*_k->N+d] = _samplePopulation[sampleIdx*_k->N+d];
    }
}


bool DEA::isFeasible(size_t sampleIdx) const
{
  for(size_t d = 0; d < _k->N; ++d) 
    if ( (_sampleCandidates[sampleIdx*_k->N+d] < _k->_variables[d]->_lowerBound) || (_sampleCandidates[sampleIdx*_k->N+d] > _k->_variables[d]->_upperBound)) return false;
  return true;
}


void DEA::fixInfeasible(size_t sampleIdx)
{
  for(size_t d = 0; d < _k->N; ++d) 
  {
    if ( _sampleCandidates[sampleIdx*_k->N+d] < _k->_variables[d]->_lowerBound )
    { double len = _samplePopulation[sampleIdx*_k->N+d] - _k->_variables[d]->_lowerBound;
      _sampleCandidates[sampleIdx*_k->N+d] = _samplePopulation[sampleIdx*_k->N+d] - len * _uniformGenerator->getRandomNumber(); }
    if ( _sampleCandidates[sampleIdx*_k->N+d] > _k->_variables[d]->_upperBound )
    { double len = _k->_variables[d]->_upperBound - _samplePopulation[sampleIdx*_k->N+d];
      _sampleCandidates[sampleIdx*_k->N+d] = _samplePopulation[sampleIdx*_k->N+d] + len * _uniformGenerator->getRandomNumber(); }
  }
}


void DEA::evaluateSamples()
{
  while (_finishedSampleCount < _sampleCount)
  {
    for (size_t i = 0; i < _sampleCount; i++) if (_isInitializedSample[i] == false)
    {
       std::vector<double> _logTransformedSample(_k->N);

     for(size_t d = 0; d<_k->N; ++d)
       if (_k->_variables[d]->_isLogSpace == true)
           _logTransformedSample[d] = std::exp(_sampleCandidates[i*_k->N+d]);
       else
           _logTransformedSample[d] = _sampleCandidates[i*_k->N+d];

      _isInitializedSample[i] = true;
      _k->_conduit->evaluateSample(_logTransformedSample.data(), i);
    }
    _k->_conduit->checkProgress();
  }

}


void DEA::processSample(size_t sampleId, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&_sampleCandidates[sampleId*_k->N]);
 fitness += logPrior;
 if(std::isfinite(fitness) == false) 
 {
   fitness = -1.0 * std::numeric_limits<double>::max();
   koraliWarning(KORALI_NORMAL,"Sample %zu returned non finite fitness (fitness set to %e)!\n", sampleId, fitness);
 }
 _fitnessVector[sampleId] = fitness;
 _finishedSampleCount++;
}


void DEA::updateSolver()
{
    _bestSampleIndex = std::distance( std::begin(_fitnessVector), std::max_element(std::begin(_fitnessVector), std::end(_fitnessVector)) );
    _previousBestEverValue  = _bestEverValue;
    _previousBestValue = _currentBestValue;
    _currentBestValue  = _fitnessVector[_bestSampleIndex];

    for(size_t d = 0; d < _k->N; ++d) _currentBestSample[d] = _sampleCandidates[_bestSampleIndex*_k->N+d];
    
    _previousMean.assign(std::begin(_sampleMeans), std::end(_sampleMeans));
    std::fill(std::begin(_sampleMeans), std::end(_sampleMeans), 0.0);

    if(_currentBestValue > _bestEverValue) _bestEverCoordinates.assign(std::begin(_currentBestSample), std::end(_currentBestSample));

    switch (str2int(_acceptRule.c_str()))
    {
        case str2int("Best") : // only update best sample
            if(_currentBestValue > _bestEverValue)
            {
              for(size_t d = 0; d < _k->N; ++d) _samplePopulation[_bestSampleIndex*_k->N+d] = _sampleCandidates[_bestSampleIndex*_k->N+d];
              _bestEverValue = _currentBestValue;
            }
            break;

        case str2int("Greedy") : // accept all mutations better than parent
            for(size_t i = 0; i < _sampleCount; ++i) if(_fitnessVector[i] > _previousFitnessVector[i])
                for(size_t d = 0; d < _k->N; ++d) _samplePopulation[i*_k->N+d] = _sampleCandidates[i*_k->N+d];
            if(_currentBestValue > _bestEverValue) _bestEverValue = _currentBestValue;
            break;

        case str2int("Improved") : // update all samples better than _bestEverValue
            for(size_t i = 0; i < _sampleCount; ++i) if(_fitnessVector[i] > _bestEverValue)
                for(size_t d = 0; d < _k->N; ++d) _samplePopulation[i*_k->N+d] = _sampleCandidates[i*_k->N+d];
            if(_currentBestValue > _bestEverValue) _bestEverValue = _currentBestValue;
            break;

        case str2int("Iterative") : // iteratibely update _bestEverValue and accept samples
            for(size_t i = 0; i < _sampleCount; ++i) if(_fitnessVector[i] > _bestEverValue)
             { for(size_t d = 0; d < _k->N; ++d) _samplePopulation[i*_k->N+d] = _sampleCandidates[i*_k->N+d]; _bestEverValue = _fitnessVector[i]; }
            break;

        default :
            koraliError("Accept Rule (%s) not recognized.\n",  _acceptRule.c_str());
    }

    for(size_t i = 0; i < _sampleCount; ++i) for(size_t d = 0; d < _k->N; ++d) 
        _sampleMeans[d] += _samplePopulation[i*_k->N+d]/((double)_sampleCount);

    for(size_t d = 0; d < _k->N; ++d) 
    {
        double max = -std::numeric_limits<double>::max();
        double min = -max;
        for(size_t i = 0; i < _sampleCount; ++i)
        {
            if (_samplePopulation[i*_k->N+d] > max) max = _samplePopulation[i*_k->N+d];
            if (_samplePopulation[i*_k->N+d] < min) min = _samplePopulation[i*_k->N+d];
        }
        _maxDistances[d] = max-min;
    }

}


bool DEA::checkTermination()
{

 bool isFinished = false;
 if ( _minFitnessEnabled && (_k->_currentGeneration > 1) && (_bestEverValue >= _minFitness) )
 {
  isFinished = true;
  koraliLog(KORALI_MINIMAL, "Fitness Value (%+6.3e) > (%+6.3e).\n",  _bestEverValue, _minFitness);
 }
 
 if ( _maxFitnessEnabled && (_k->_currentGeneration > 1) && (_bestEverValue >= _maxFitness) )
 {
  isFinished = true;
  koraliLog(KORALI_MINIMAL, "Fitness Value (%+6.3e) > (%+6.3e).\n",  _bestEverValue, _maxFitness);
 }

 double range = fabs(_currentBestValue - _previousBestValue);
 if ( _minFitnessDiffThresholdEnabled && (_k->_currentGeneration > 1) && (range < _minFitnessDiffThreshold) )
 {
  isFinished = true;
  koraliLog(KORALI_MINIMAL, "Fitness Diff Threshold (%+6.3e) < (%+6.3e).\n",  range, _minFitnessDiffThreshold);
 }
 
 if ( _minStepSizeEnabled && (_k->_currentGeneration > 1) )
 {
   size_t cTemp = 0;
   for(size_t d = 0; d < _k->N; ++d) cTemp += (fabs(_sampleMeans[d] - _previousMean[d]) < _minStepSize) ? 1 : 0;
   if (cTemp == _k->N) 
   {
    isFinished = true;
    koraliLog(KORALI_MINIMAL, "Mean changes < %+6.3e for all variables.\n", _minStepSize);
   }
 }
 
 if(_maxGenerationsEnabled)
  if(_k->_currentGeneration > _maxGenerations)
  {
   _maxGenerationsTriggered = true;
   koraliLog( KORALI_MINIMAL, "Max Generations Reached.\n", _maxGenerations);
   isFinished = true;
  }

 if(_maxModelEvaluationsEnabled)
 if(_k->_functionEvaluationCount > _maxModelEvaluations)
 {
  _maxModelEvaluationsTriggered = true;
  koraliLog( KORALI_MINIMAL, "Max Model Evaluations Reached.\n", _maxModelEvaluations);
  isFinished = true;
 }

 return isFinished;
}


/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/


void DEA::printGeneration()
{
 koraliLog(KORALI_NORMAL, "Differential Evolution Generation %zu\n", _k->_currentGeneration);
 koraliLog(KORALI_NORMAL, "Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
 koraliLog(KORALI_DETAILED, "Variable = (MeanX, BestX):\n");
 for (size_t d = 0; d < _k->N; d++) koraliLogData(KORALI_DETAILED, "         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _sampleMeans[d], _bestEverCoordinates[d]);
 koraliLog(KORALI_DETAILED, "Max Width:\n");
 for (size_t d = 0; d < _k->N; d++) koraliLogData(KORALI_DETAILED, "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _maxDistances[d]);
 koraliLog(KORALI_DETAILED, "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
 koraliLog(KORALI_NORMAL, "--------------------------------------------------------------------\n");
}

void DEA::finalize()
{
 koraliLog(KORALI_MINIMAL, "Differential Evolution Finished\n");
 koraliLog(KORALI_MINIMAL, "Optimum found: %e\n", _bestEverValue);
 koraliLog(KORALI_MINIMAL, "Optimum found at:\n");
 for (size_t d = 0; d < _k->N; ++d) koraliLogData(KORALI_MINIMAL, "         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverCoordinates[d]);
 koraliLog(KORALI_MINIMAL, "Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
 koraliLog(KORALI_MINIMAL, "--------------------------------------------------------------------\n");
}
