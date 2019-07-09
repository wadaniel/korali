#include "korali.h"
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort
using namespace Korali::Solver;

constexpr size_t str2int(const char* str, int h = 0) { return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h]; }

DE::DE()
{
 // Initializing Generators
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"]  = "Gaussian";
 jsGaussian["Mean"]  = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = new Variable();
 _gaussianGenerator->setDistribution(jsGaussian);

 auto jsUniform = nlohmann::json();
 jsUniform["Type"] = "Uniform";
 jsUniform["Minimum"] = 0.0;
 jsUniform["Maximum"] = 1.0;
 jsUniform["Seed"] = _k->_seed++;

 _uniformGenerator = new Variable();
 _uniformGenerator->setDistribution(jsUniform);
}

void DE::initialize()
{
 // Allocating Memory
 _samplePopulation.resize(_k->N*_sampleCount);
 _sampleCandidates.resize(_k->N*_sampleCount);
 _transformedSamples.resize(_k->N*_sampleCount);

 _rgxOldMean.resize(_k->N);
 _rgxMean.resize(_k->N);
 _rgxBestEver.resize(_k->N);
 _curBestEver.resize(_k->N);
 _maxWidth.resize(_k->N);

 _isInitializedSample.resize(_sampleCount);
 _prevfitnessVector.resize(_sampleCount);
 _fitnessVector.resize(_sampleCount);

 _k->_isFinished = false;

 if (_objective == "Maximize")      _evaluationSign = 1.0;
 else if(_objective == "Minimize")  _evaluationSign = -1.0;
 else { fprintf(stderr,"[Korali] Error: Objective must be either be initialized to \'Maximize\' or \'Minimize\' (is %s).\n", _objective.c_str()); exit(-1); }         

 _infeasibleSampleCount   = 0;
 _bestIndex               = 0;
 _previousFunctionValue   = -std::numeric_limits<double>::max();
 _currentFunctionValue    = -std::numeric_limits<double>::max();
 _bestEver                = -std::numeric_limits<double>::max();

 for(size_t d = 0; d < _k->N; ++d) if(_variableSettings[d].upperBound < _variableSettings[d].lowerBound)
    { fprintf(stderr,"[Korali] Error: Lower Bound (%.4f) of variable \'%s\'  exceeds Upper Bound (%.4f).\n",          
            _variableSettings[d].lowerBound,
            _k->_variables[d]->_name.c_str(),
            _variableSettings[d].upperBound);
      exit(-1);
    }

 initSamples();

 for(size_t d = 0; d < _k->N; ++d) { _rgxOldMean[d] = 0.0; _rgxMean[d] = 0.0; }

 for(size_t i = 0; i < _sampleCount; ++i) for(size_t d = 0; d < _k->N; ++d) 
   _rgxMean[d] += _samplePopulation[i*_k->N+d]/((double)_sampleCount);

}

void DE::runGeneration()
{
 prepareGeneration();
 evaluateSamples();
 updateSolver();
}


void DE::initSamples()
{
  for(size_t i = 0; i < _sampleCount; ++i) for(size_t d = 0; d < _k->N; ++d)
  {
    double width = _variableSettings[d].upperBound - _variableSettings[d].lowerBound;
    _samplePopulation[i*_k->N+d] = _variableSettings[d].lowerBound + width * _uniformGenerator->getRandomNumber();
  }
}


void DE::prepareGeneration()
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


void DE::mutateSingle(size_t sampleIdx)
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


bool DE::isFeasible(size_t sampleIdx) const
{
  for(size_t d = 0; d < _k->N; ++d) 
    if ( (_sampleCandidates[sampleIdx*_k->N+d] < _variableSettings[d].lowerBound) || (_sampleCandidates[sampleIdx*_k->N+d] > _variableSettings[d].upperBound)) return false;
  return true;
}


void DE::fixInfeasible(size_t sampleIdx)
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


void DE::evaluateSamples()
{
  
  for (size_t i = 0; i < _sampleCount; i++) for(size_t d = 0; d < _k->N; ++d)
    //if(_k->_variables[d]->_isLogSpace == true)
    //    _transformedSamples[i*_k->N+d] = std::exp(_sampleCandidates[i*_k->N+d]);
    //else 
        _transformedSamples[i*_k->N+d] = _sampleCandidates[i*_k->N+d];


  while (_finishedSampleCount < _sampleCount)
  {
    for (size_t i = 0; i < _sampleCount; i++) if (_isInitializedSample[i] == false)
    {
      _isInitializedSample[i] = true;
      _k->_conduit->evaluateSample(&_transformedSamples[0], i);
    }
    _k->_conduit->checkProgress();
  }

}


void DE::processSample(size_t sampleIdx, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&_transformedSamples[sampleIdx*_k->N]);
 _fitnessVector[sampleIdx] = _evaluationSign * (logPrior+fitness);
 _finishedSampleCount++;
}


void DE::updateSolver()
{
    _bestIndex = std::distance( std::begin(_fitnessVector), std::max_element(std::begin(_fitnessVector), std::end(_fitnessVector)) );
    _previousFunctionValue = _currentFunctionValue;
    _currentFunctionValue  = _fitnessVector[_bestIndex];

    for(size_t d = 0; d < _k->N; ++d) _curBestEver[d] = _sampleCandidates[_bestIndex*_k->N+d];
    
    _rgxOldMean.assign(std::begin(_rgxMean), std::end(_rgxMean));
    std::fill(std::begin(_rgxMean), std::end(_rgxMean), 0.0); 

    if(_currentFunctionValue > _bestEver) _rgxBestEver.assign(std::begin(_curBestEver), std::end(_curBestEver));

    switch (str2int(_acceptRule.c_str()))
    {
        case str2int("Best") : // only update best sample
            if(_currentFunctionValue > _bestEver)
            {
              for(size_t d = 0; d < _k->N; ++d) _samplePopulation[_bestIndex*_k->N+d] = _sampleCandidates[_bestIndex*_k->N+d];
              _bestEver = _currentFunctionValue;
            }
            break;

        case str2int("Greedy") : // accept all mutations better than parent
            for(size_t i = 0; i < _sampleCount; ++i) if(_fitnessVector[i] > _prevfitnessVector[i])
                for(size_t d = 0; d < _k->N; ++d) _samplePopulation[i*_k->N+d] = _sampleCandidates[i*_k->N+d];
            if(_currentFunctionValue > _bestEver) _bestEver = _currentFunctionValue;
            break;

        case str2int("Improved") : // update all samples better than _bestEver
            for(size_t i = 0; i < _sampleCount; ++i) if(_fitnessVector[i] > _bestEver)
                for(size_t d = 0; d < _k->N; ++d) _samplePopulation[i*_k->N+d] = _sampleCandidates[i*_k->N+d];
            if(_currentFunctionValue > _bestEver) _bestEver = _currentFunctionValue;
            break;

        case str2int("Iterative") : // iteratibely update _bestEver and accept samples
            for(size_t i = 0; i < _sampleCount; ++i) if(_fitnessVector[i] > _bestEver)
             { for(size_t d = 0; d < _k->N; ++d) _samplePopulation[i*_k->N+d] = _sampleCandidates[i*_k->N+d]; _bestEver = _fitnessVector[i]; }
            break;

        default :
            if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Error: Accept Rule (%s) not recognized.\n",  _acceptRule.c_str());
            exit(-1);
    }

    for(size_t i = 0; i < _sampleCount; ++i) for(size_t d = 0; d < _k->N; ++d) 
        _rgxMean[d] += _samplePopulation[i*_k->N+d]/((double)_sampleCount);

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


void DE::checkTermination()
{

 if ( _termCondMinFitnessEnabled && (_k->currentGeneration > 1) && (_bestEver >= _termCondMinFitness) )
 {
  _k->_isFinished = true;
  printf("[Korali] Fitness Value (%+6.3e) > (%+6.3e).\n",  _bestEver, _termCondMinFitness);
 }
 
 if ( _termCondMaxFitnessEnabled && (_k->currentGeneration > 1) && (_bestEver >= _termCondMaxFitness) )
 {
  _k->_isFinished = true;
  printf("[Korali] Fitness Value (%+6.3e) > (%+6.3e).\n",  _bestEver, _termCondMaxFitness);
 }

 double range = fabs(_currentFunctionValue - _previousFunctionValue);
 if ( _termCondMinFitnessDiffThresholdEnabled && (_k->currentGeneration > 1) && (range < _termCondMinFitnessDiffThreshold) )
 {
  _k->_isFinished = true;
  printf("[Korali] Fitness Diff Threshold (%+6.3e) < (%+6.3e).\n",  range, _termCondMinFitnessDiffThreshold);
 }
 
 if ( _termCondMinStepSizeEnabled && (_k->currentGeneration > 1) )
 {
   size_t cTemp = 0;
   for(size_t d = 0; d < _k->N; ++d) cTemp += (fabs(_rgxMean[d] - _rgxOldMean[d]) < _termCondMinStepSize) ? 1 : 0;
   if (cTemp == _k->N) 
   {
    _k->_isFinished = true;
    printf("[Korali] Mean changes < %+6.3e for all variables.\n", _termCondMinStepSize);
   }
 }
 
 if( _termCondMaxGenerationsEnabled && (_k->currentGeneration >= _termCondMaxGenerations) )
 {
  _k->_isFinished = true;
  printf("[Korali] Maximum number of Generations reached (%lu).\n", _termCondMaxGenerations);
 }

}


/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/


void DE::printGeneration()
{

 if (_k->currentGeneration % _terminalOutputFrequency != 0) return;
 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Differential Evolution Generation %zu\n", _k->currentGeneration);
 
 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentFunctionValue, _bestEver);
  // TODO: sth like width of samples: printf("[Korali] Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", minEW, maxEW);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _rgxMean[d], _rgxBestEver[d]);
  printf("[Korali] Max Width:\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _maxWidth[d]);
  printf("[Korali] Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
 }

 if (_k->_verbosity >= KORALI_NORMAL)
   printf("--------------------------------------------------------------------\n");
}

void DE::finalize()
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] Differential Evolution Finished\n");
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), _bestEver);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t d = 0; d < _k->N; ++d) printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _rgxBestEver[d]);
    printf("[Korali] Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
    printf("--------------------------------------------------------------------\n");
 }
}
