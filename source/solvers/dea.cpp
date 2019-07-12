#include "korali.h"
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
 jsGaussian["Sigma"] = 1.0;
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
 // Allocating Memory
 _samplePopulation.resize(_k->N*_sampleCount);
 _sampleCandidates.resize(_k->N*_sampleCount);

 _oldMean.resize(_k->N);
 _mean.resize(_k->N);
 _bestEverSample.resize(_k->N);
 _currentBestSample.resize(_k->N);
 _maxWidth.resize(_k->N);

 _isInitializedSample.resize(_sampleCount);
 _prevfitnessVector.resize(_sampleCount);
 _fitnessVector.resize(_sampleCount);

 if (_objective == "Maximize")      _evaluationSign = 1.0;
 else if(_objective == "Minimize")  _evaluationSign = -1.0;
 else { fprintf(stderr,"[Korali] Error: Objective must be either be initialized to \'Maximize\' or \'Minimize\' (is %s).\n", _objective.c_str()); exit(-1); }         

 _infeasibleSampleCount = 0;
 _bestIndex             = 0;
 _previousBestValue     = -std::numeric_limits<double>::max();
 _currentBestValue      = -std::numeric_limits<double>::max();
 _previousBestEverValue = -std::numeric_limits<double>::max();
 _bestEverValue         = -std::numeric_limits<double>::max();

 for(size_t d = 0; d < _k->N; ++d) if(_variableSettings[d].upperBound < _variableSettings[d].lowerBound)
    { fprintf(stderr,"[Korali] Error: Lower Bound (%.4f) of variable \'%s\'  exceeds Upper Bound (%.4f).\n",          
            _variableSettings[d].lowerBound,
            _k->_variables[d]->_name.c_str(),
            _variableSettings[d].upperBound);
      exit(-1);
    }

 initSamples();

 for(size_t d = 0; d < _k->N; ++d) { _oldMean[d] = 0.0; _mean[d] = 0.0; }

 for(size_t i = 0; i < _sampleCount; ++i) for(size_t d = 0; d < _k->N; ++d) 
   _mean[d] += _samplePopulation[i*_k->N+d]/((double)_sampleCount);

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
    double width = _variableSettings[d].upperBound - _variableSettings[d].lowerBound;
    _samplePopulation[i*_k->N+d] = _variableSettings[d].lowerBound + width * _uniformGenerator->getRandomNumber();
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

	if ( _termCondMaxInfeasibleResamplingsEnabled )
	if ( (_infeasibleSampleCount - initial_infeasible) > _termCondMaxInfeasibleResamplings )
	{
		if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: exiting resampling loop (param %zu) , max resamplings (%zu) reached.\n", i, _termCondMaxInfeasibleResamplings);
		exit(-1);
	}
	}
 }

 _finishedSampleCount = 0;
 for (size_t i = 0; i < _sampleCount; i++) _isInitializedSample[i] = false;

 _prevfitnessVector.assign(std::begin(_fitnessVector), std::end(_fitnessVector));
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
        parent = &_samplePopulation[_bestIndex*_k->N];
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
    if ( (_sampleCandidates[sampleIdx*_k->N+d] < _variableSettings[d].lowerBound) || (_sampleCandidates[sampleIdx*_k->N+d] > _variableSettings[d].upperBound)) return false;
  return true;
}


void DEA::fixInfeasible(size_t sampleIdx)
{
  for(size_t d = 0; d < _k->N; ++d) 
  {
    if ( _sampleCandidates[sampleIdx*_k->N+d] < _variableSettings[d].lowerBound ) 
    { double len = _samplePopulation[sampleIdx*_k->N+d] - _variableSettings[d].lowerBound; 
      _sampleCandidates[sampleIdx*_k->N+d] = _samplePopulation[sampleIdx*_k->N+d] - len * _uniformGenerator->getRandomNumber(); }
    if ( _sampleCandidates[sampleIdx*_k->N+d] > _variableSettings[d].upperBound )
    { double len = _variableSettings[d].upperBound - _samplePopulation[sampleIdx*_k->N+d]; 
      _sampleCandidates[sampleIdx*_k->N+d] = _samplePopulation[sampleIdx*_k->N+d] + len * _uniformGenerator->getRandomNumber(); }
  }
}


void DEA::evaluateSamples()
{
  while (_finishedSampleCount < _sampleCount)
  {
    for (size_t i = 0; i < _sampleCount; i++) if (_isInitializedSample[i] == false)
    {
      _isInitializedSample[i] = true;
      _k->_conduit->evaluateSample(&_sampleCandidates[i*_k->N], i);
    }
    _k->_conduit->checkProgress();
  }

}


void DEA::processSample(size_t sampleIdx, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&_sampleCandidates[sampleIdx*_k->N]);
 _fitnessVector[sampleIdx] = _evaluationSign * (logPrior+fitness);
 _finishedSampleCount++;
}


void DEA::updateSolver()
{
    _bestIndex = std::distance( std::begin(_fitnessVector), std::max_element(std::begin(_fitnessVector), std::end(_fitnessVector)) );
    _previousBestEverValue  = _bestEverValue;
    _previousBestValue = _currentBestValue;
    _currentBestValue  = _fitnessVector[_bestIndex];

    for(size_t d = 0; d < _k->N; ++d) _currentBestSample[d] = _sampleCandidates[_bestIndex*_k->N+d];
    
    _oldMean.assign(std::begin(_mean), std::end(_mean));
    std::fill(std::begin(_mean), std::end(_mean), 0.0); 

    if(_currentBestValue > _bestEverValue) _bestEverSample.assign(std::begin(_currentBestSample), std::end(_currentBestSample));

    switch (str2int(_acceptRule.c_str()))
    {
        case str2int("Best") : // only update best sample
            if(_currentBestValue > _bestEverValue)
            {
              for(size_t d = 0; d < _k->N; ++d) _samplePopulation[_bestIndex*_k->N+d] = _sampleCandidates[_bestIndex*_k->N+d];
              _bestEverValue = _currentBestValue;
            }
            break;

        case str2int("Greedy") : // accept all mutations better than parent
            for(size_t i = 0; i < _sampleCount; ++i) if(_fitnessVector[i] > _prevfitnessVector[i])
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
            if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Error: Accept Rule (%s) not recognized.\n",  _acceptRule.c_str());
            exit(-1);
    }

    for(size_t i = 0; i < _sampleCount; ++i) for(size_t d = 0; d < _k->N; ++d) 
        _mean[d] += _samplePopulation[i*_k->N+d]/((double)_sampleCount);

    for(size_t d = 0; d < _k->N; ++d) 
    {
        double max = -std::numeric_limits<double>::max();
        double min = -max;
        for(size_t i = 0; i < _sampleCount; ++i)
        {
            if (_samplePopulation[i*_k->N+d] > max) max = _samplePopulation[i*_k->N+d];
            if (_samplePopulation[i*_k->N+d] < min) min = _samplePopulation[i*_k->N+d];
        }
        _maxWidth[d] = max-min;
    }

}


bool DEA::checkTermination()
{

 bool isFinished = false;
 if ( _termCondMinFitnessEnabled && (_k->currentGeneration > 1) && (_bestEverValue >= _termCondMinFitness) )
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Fitness Value (%+6.3e) > (%+6.3e).\n",  _bestEverValue, _termCondMinFitness);
 }
 
 if ( _termCondMaxFitnessEnabled && (_k->currentGeneration > 1) && (_bestEverValue >= _termCondMaxFitness) )
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Fitness Value (%+6.3e) > (%+6.3e).\n",  _bestEverValue, _termCondMaxFitness);
 }

 double range = fabs(_currentBestValue - _previousBestValue);
 if ( _termCondMinFitnessDiffThresholdEnabled && (_k->currentGeneration > 1) && (range < _termCondMinFitnessDiffThreshold) )
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Fitness Diff Threshold (%+6.3e) < (%+6.3e).\n",  range, _termCondMinFitnessDiffThreshold);
 }
 
 if ( _termCondMinStepSizeEnabled && (_k->currentGeneration > 1) )
 {
   size_t cTemp = 0;
   for(size_t d = 0; d < _k->N; ++d) cTemp += (fabs(_mean[d] - _oldMean[d]) < _termCondMinStepSize) ? 1 : 0;
   if (cTemp == _k->N) 
   {
    isFinished = true;
    if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Mean changes < %+6.3e for all variables.\n", _termCondMinStepSize);
   }
 }
 
 if( _termCondMaxGenerationsEnabled && (_k->currentGeneration >= _termCondMaxGenerations) )
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Maximum number of Generations reached (%lu).\n", _termCondMaxGenerations);
 }

 return isFinished;
}


/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/


void DEA::printGeneration()
{

 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Differential Evolution Generation %zu\n", _k->currentGeneration);
 
 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
  // TODO: sth like width of samples: printf("[Korali] Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", minEW, maxEW);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _mean[d], _bestEverSample[d]);
  printf("[Korali] Max Width:\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _maxWidth[d]);
  printf("[Korali] Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
 }

 if (_k->_verbosity >= KORALI_NORMAL)
   printf("--------------------------------------------------------------------\n");
}

void DEA::finalize()
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] Differential Evolution Finished\n");
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), _bestEverValue);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t d = 0; d < _k->N; ++d) printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverSample[d]);
    printf("[Korali] Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
    printf("--------------------------------------------------------------------\n");
 }
}
