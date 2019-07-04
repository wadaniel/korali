#include "korali.h"
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort
using namespace Korali::Solver;

constexpr size_t str2int(const char* str, int h = 0) { return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h]; }

void DE::initialize()
{
 // Allocating Memory
 samplePopulation = (double*) calloc (sizeof(double), _k->N*_s);
 candidates       = (double*) calloc (sizeof(double), _k->N*_s);

 rgxoldmean     = (double*) calloc (sizeof(double), _k->N);
 rgxmean        = (double*) calloc (sizeof(double), _k->N);
 rgxbestever    = (double*) calloc (sizeof(double), _k->N);
 curBestVector  = (double*) calloc (sizeof(double), _k->N);
 maxWidth       = (double*) calloc (sizeof(double), _k->N);

 histFuncValues = (double*) calloc (sizeof(double), _termCondMaxGenerations+1);

 initializedSample = (bool*) calloc (sizeof(bool), _s);
 oldFitnessVector  = (double*) calloc (sizeof(double), _s);
 fitnessVector     = (double*) calloc (sizeof(double), _s);

 // Init Generation
 _isFinished = false;

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

 countevals      = 0;
 countinfeasible = 0;
 bestIndex       = 0;
 prevFunctionValue    = -std::numeric_limits<double>::max();
 currentFunctionValue = -std::numeric_limits<double>::max();
 bestEver             = -std::numeric_limits<double>::max();

 /* set rgxmean */
 for (size_t i = 0; i < _k->N; ++i) if (_initialMeanDefined[i])
 {
	 if(_initialMeans[i] < _lowerBounds[i] || _initialMeans[i] > _upperBounds[i])
		fprintf(stderr,"[Korali] Warning: Initial Value (%.4f) for \'%s\' is out of bounds (%.4f-%.4f).\n",
						_initialMeans[i],
						_k->_variables[i]->_name.c_str(),
						_lowerBounds[i],
						_upperBounds[i]);
	 rgxmean[i] = rgxoldmean[i] = _initialMeans[i];
 }

 initSamples();
}

void DE::runGeneration()
{
 prepareGeneration();
 evaluateSamples();
 updateSolver(fitnessVector);
}


void DE::initSamples()
{
  for(size_t i = 0; i < _s; ++i) for(size_t d = 0; d < _k->N; ++d)
  {
    if (_initialMeanDefined[d] && _initialStdDevDefined[d]) 
    {
        samplePopulation[i*_k->N+d] = _initialMeans[d] + _initialStdDevs[d] * _gaussianGenerator->getRandomNumber();
        if(samplePopulation[i*_k->N+d] < _lowerBounds[d]) samplePopulation[i*_k->N+d] = _lowerBounds[d];
        if(samplePopulation[i*_k->N+d] > _upperBounds[d]) samplePopulation[i*_k->N+d] = _upperBounds[d];
    }
    else 
        samplePopulation[i*_k->N+d] = _lowerBounds[d] + (_upperBounds[d]-_lowerBounds[d]) * _uniformGenerator->getRandomNumber();
  }
}


void DE::prepareGeneration()
{
 size_t initial_infeasible = countinfeasible;
 for (size_t i = 0; i < _s; ++i)
 {
    mutateSingle(i);
    while(isFeasible(i) == false)
    {
        countinfeasible++;
        
        if (_fixinfeasible) fixInfeasible(i);
        else mutateSingle(i);
        
        if ( (countinfeasible - initial_infeasible) > _maxResamplings )
        {
          if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: exiting resampling loop (param %zu) , max resamplings (%zu) reached.\n", i, _maxResamplings);
          exit(-1);
        }
    }
 }

 finishedSamples = 0;
 for (size_t i = 0; i < _s; i++) initializedSample[i] = false;

 double *tmp = fitnessVector;
 fitnessVector = oldFitnessVector;
 oldFitnessVector = fitnessVector;
}


void DE::mutateSingle(size_t sampleIdx)
{
    size_t a, b;
    do{ a = _uniformGenerator->getRandomNumber()*_s; } while(a == sampleIdx);
    do{ b = _uniformGenerator->getRandomNumber()*_s; } while(b == sampleIdx || b == a);

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
    if(_parent == "Random")
    {
        size_t c;
        do{ c = _uniformGenerator->getRandomNumber()*_s; } while(c == sampleIdx || c == a || c == b);
        parent = &samplePopulation[c*_k->N];
    }
    else
    {
     /* _parent == "Best" */
        parent = &samplePopulation[bestIndex*_k->N];
    }

    size_t rn = _uniformGenerator->getRandomNumber()*_k->N;
    for(size_t d = 0; d < _k->N; ++d)
    {
      if( (_uniformGenerator->getRandomNumber() < _crossoverRate) || (d == rn) )
          candidates[sampleIdx*_k->N+d] = parent[d]+_mutationRate*(samplePopulation[a*_k->N+d]-samplePopulation[b*_k->N+d]);
      else
          candidates[sampleIdx*_k->N+d] = samplePopulation[sampleIdx*_k->N+d];
    }
}


bool DE::isFeasible(size_t sampleIdx) const
{
  for(size_t d = 0; d < _k->N; ++d) if ( (candidates[sampleIdx*_k->N+d] < _lowerBounds[d]) || (candidates[sampleIdx*_k->N+d] > _upperBounds[d])) return false;
  return true;
}


void DE::fixInfeasible(size_t sampleIdx)
{
  for(size_t d = 0; d < _k->N; ++d) 
  {
    if ( candidates[sampleIdx*_k->N+d] < _lowerBounds[d] ) candidates[sampleIdx*_k->N+d] *= _uniformGenerator->getRandomNumber(); 
    if ( candidates[sampleIdx*_k->N+d] > _upperBounds[d] ) 
    { double len = _upperBounds[d] - samplePopulation[sampleIdx*_k->N+d]; candidates[sampleIdx*_k->N+d] = samplePopulation[sampleIdx*_k->N+d] + len * _uniformGenerator->getRandomNumber(); }
  }
}


void DE::evaluateSamples()
{
  double* transformedSamples = new double[ _s * _k->N ];
  
  for (size_t i = 0; i < _s; i++) for(size_t d = 0; d < _k->N; ++d)
    if(_variableLogSpace[d] == true) 
        transformedSamples[i*_k->N+d] = std::exp(candidates[i*_k->N+d]);
    else 
        transformedSamples[i*_k->N+d] = candidates[i*_k->N+d];


  while (finishedSamples < _s)
  {
    for (size_t i = 0; i < _s; i++) if (initializedSample[i] == false)
    {
      initializedSample[i] = true; 
      _k->_conduit->evaluateSample(transformedSamples, i); countevals++;
    }
    _k->_conduit->checkProgress();
  }

  delete transformedSamples;
}


void DE::processSample(size_t sampleIdx, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&samplePopulation[sampleIdx*_k->N]);
 fitnessVector[sampleIdx] = _fitnessSign * (logPrior+fitness);
 finishedSamples++;
}


void DE::updateSolver(const double *fitnessVector)
{
    bestIndex            = maxIdx(fitnessVector, _s);
    prevFunctionValue    = currentFunctionValue;
    currentFunctionValue = fitnessVector[bestIndex];
    for(size_t d = 0; d < _k->N; ++d) curBestVector[d] = candidates[bestIndex*_k->N+d];
    
    if(currentFunctionValue > bestEver)
    {
        prevBest = bestEver;
        
        for(size_t d = 0; d < _k->N; ++d) 
        {
            rgxoldmean[d]  = rgxmean[d];
            rgxbestever[d] = curBestVector[d];
            rgxmean[d]     = 0.0;
        }
   
        switch (str2int(_acceptRule.c_str()))
        {
            case str2int("Best") : // only update best sample
                bestEver = currentFunctionValue;
                for(size_t d = 0; d < _k->N; ++d) samplePopulation[bestIndex*_k->N+d] = candidates[bestIndex*_k->N+d];
                break;

            case str2int("Greedy") : // accept if mutation better than parent
                for(size_t i = 0; i < _s; ++i) if(fitnessVector[i] > oldFitnessVector[i])
                    for(size_t d = 0; d < _k->N; ++d) samplePopulation[i*_k->N+d] = candidates[i*_k->N+d];
                bestEver = currentFunctionValue;
                break;

            case str2int("Iterative") : // iteratibely accept samples and update bestEver
                for(size_t i = 0; i < _s; ++i) if(fitnessVector[i] > bestEver)
                 { for(size_t d = 0; d < _k->N; ++d) samplePopulation[i*_k->N+d] = candidates[i*_k->N+d]; bestEver = fitnessVector[i]; }
                break;
 
            case str2int("Improved") : // update all samples better than bestEver
                for(size_t i = 0; i < _s; ++i) if(fitnessVector[i] > bestEver)
                    for(size_t d = 0; d < _k->N; ++d) samplePopulation[i*_k->N+d] = candidates[i*_k->N+d];
                bestEver = currentFunctionValue;
                break;

            default :
                if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Error: Accept Rule not recognize (%s).\n",  _acceptRule.c_str());
                exit(-1);
        }

        for(size_t i = 0; i < _s; ++i) for(size_t d = 0; d < _k->N; ++d) 
            rgxmean[d] += samplePopulation[i*_k->N+d]/((double)_s);
    }
    
    for(size_t d = 0; d < _k->N; ++d) 
    {
        double max = -std::numeric_limits<double>::max();
        double min = -max;
        for(size_t i = 0; i < _s; ++i)
        {
            if (samplePopulation[i*_k->N+d] > max) max = samplePopulation[i*_k->N+d];
            if (samplePopulation[i*_k->N+d] < min) min = samplePopulation[i*_k->N+d];
        }
        maxWidth[d] = max-min;
    }

}


bool DE::checkTermination()
{

 if ( _isTermCondMinFitness && (_k->currentGeneration > 1) && (bestEver >= _termCondMinFitness) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Fitness Value (%+6.3e) > (%+6.3e).",  bestEver, _termCondMinFitness);
 }
 
 if ( _isTermCondMaxFitness && (_k->currentGeneration > 1) && (bestEver >= _termCondMaxFitness) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Fitness Value (%+6.3e) > (%+6.3e).",  bestEver, _termCondMaxFitness);
 }

 double range = fabs(currentFunctionValue - prevFunctionValue);
 if ( _isTermCondFitnessDiffThreshold && (_k->currentGeneration > 1) && (range < _termCondFitnessDiffThreshold) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Fitness Diff Threshold (%+6.3e) < (%+6.3e).",  range, _termCondFitnessDiffThreshold);
 }
 
 if ( _isTermCondMinDeltaX && (_k->currentGeneration > 1) )
 {
   size_t cTemp = 0;
   for(size_t d = 0; d < _k->N; ++d) cTemp += (fabs(rgxmean[d] - rgxoldmean[d]) < _termCondMinDeltaX) ? 1 : 0;
   if (cTemp == _k->N) 
   {
    _isFinished = true;
    sprintf(_terminationReason, "Mean changes < %+6.3e for all variables.", _termCondMinDeltaX);
   }
 }

 if( _isTermCondMaxFitnessEvaluations && (countevals >= _termCondMaxFitnessEvaluations) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Conducted %lu function evaluations >= (%lu).", countevals, _termCondMaxFitnessEvaluations);
 }

 if( _isTermCondMaxGenerations && (_k->currentGeneration >= _termCondMaxGenerations) )
 {
  _isFinished = true;
  sprintf(_terminationReason, "Maximum number of Generations reached (%lu).", _termCondMaxGenerations);
 }

 return _isFinished;
}


/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/


size_t DE::maxIdx(const double *rgd, size_t len) const
{
 size_t res = 0;
 for(size_t i = 1; i < len; i++) if(rgd[i] > rgd[res]) res = i;
 return res;
}

void DE::printGeneration()
{

 if (_k->currentGeneration % terminalOutputFrequency != 0) return;
 
 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Current Function Value: Max = %+6.3e - Best = %+6.3e\n", currentFunctionValue, bestEver);
  // TODO: sth like width of samples: printf("[Korali] Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", minEW, maxEW);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), rgxmean[d], rgxbestever[d]);
  printf("[Korali] Max Width:\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), maxWidth[d]);
  printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
  printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
 }

 if (_k->_verbosity >= KORALI_NORMAL)
   printf("--------------------------------------------------------------------\n");
}

void DE::finalize()
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] Differential Evolution Finished\n");
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), bestEver);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t d = 0; d < _k->N; ++d) printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), rgxbestever[d]);
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
    printf("[Korali] Stopping Criterium: %s\n", _terminationReason);
    printf("--------------------------------------------------------------------\n");
 }
}
