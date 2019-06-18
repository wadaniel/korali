#include "korali.h"
#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort
using namespace Korali::Solver;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

constexpr size_t str2int(const char* str, int h = 0) { return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h]; }

DE::DE(nlohmann::json& js, std::string name)
{
 _name = name;
 
 setConfiguration(js);

 // Allocating Memory
 samplePopulation = (double*) calloc (sizeof(double), _k->N*_s);
 candidates       = (double*) calloc (sizeof(double), _k->N*_s);

 rgxoldmean     = (double*) calloc (sizeof(double), _k->N);
 rgxmean        = (double*) calloc (sizeof(double), _k->N);
 rgxbestever    = (double*) calloc (sizeof(double), _k->N);
 curBestVector  = (double*) calloc (sizeof(double), _k->N);

 histFuncValues = (double*) calloc (sizeof(double), _termCondMaxGenerations+1);

 initializedSample = (bool*) calloc (sizeof(bool), _s);
 oldFitnessVector  = (double*) calloc (sizeof(double), _s);
 fitnessVector     = (double*) calloc (sizeof(double), _s);

 // Init Generation
 currentGeneration = 0;

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

 // If state is defined:
 if (isDefined(js, {"State"}))
 {
  setState(js);
  js.erase("State");
 }

}

DE::~DE()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json DE::getConfiguration()
{
 auto js = nlohmann::json();

 js["DE"]["Sample Count"]    = _s;
 js["DE"]["Crossover Rate"]  = _crossoverRate;
 js["DE"]["Mutation Rate"]   = _mutationRate;
 js["DE"]["Objective"]       = _objective;
 js["DE"]["Max Resamplings"] = _maxResamplings;
 js["DE"]["Accept Rule"]     = _acceptRule;

 // Variable information
 for (size_t i = 0; i < _k->N; i++)
 {
  js["Variables"][i]["Name"]                = _varNames[i];
  js["Variables"][i]["DE"]["Lower Bound"]   = _lowerBounds[i];
  js["Variables"][i]["DE"]["Upper Bound"]   = _upperBounds[i];
  if (_initialMeanDefined[i]) js["Variables"][i]["DE"]["Initial Mean"] = _initialMeans[i];
  if (_initialStdDevDefined[i]) js["Variables"][i]["DE"]["Initial Standard Deviation"] = _initialStdDevs[i];
  js["Variables"][i]["DE"]["Log Space"]  = _variableLogSpace[i];
 }

 js["DE"]["Termination Criteria"]["Max Generations"]["Value"]         = _termCondMaxGenerations;
 js["DE"]["Termination Criteria"]["Max Generations"]["Active"]        = _isTermCondMaxGenerations;
 js["DE"]["Termination Criteria"]["Max Model Evaluations"]["Value"]   = _termCondMaxFitnessEvaluations;
 js["DE"]["Termination Criteria"]["Max Model Evaluations"]["Active"]  = _isTermCondMaxFitnessEvaluations;
 js["DE"]["Termination Criteria"]["Min Fitness"]["Value"]             = _termCondFitness;
 js["DE"]["Termination Criteria"]["Min Fitness"]["Active"]            = _isTermCondFitness;
 js["DE"]["Termination Criteria"]["Fitness Diff Threshold"]["Value"]  = _termCondFitnessDiffThreshold;
 js["DE"]["Termination Criteria"]["Fitness Diff Threshold"]["Active"] = _isTermCondFitnessDiffThreshold;
 js["DE"]["Termination Criteria"]["Min DeltaX"]["Value"]              = _termCondMinDeltaX;
 js["DE"]["Termination Criteria"]["Min DeltaX"]["Active"]             = _isTermCondMinDeltaX;

 // State Information
 js["DE"]["State"]["BestEverFunctionValue"]     = bestEver;
 js["DE"]["State"]["PreviousBestFunctionValue"] = prevBest;
 js["DE"]["State"]["EvaluationCount"]           = countevals;

 for (size_t i = 0; i < _s; i++) js["DE"]["State"]["FunctionValues"] += fitnessVector[i];
 for (size_t i = 0; i < _s; i++) js["DE"]["State"]["PreviousFunctionValues"] += oldFitnessVector[i];

 for (size_t i = 0; i < _k->N; i++) js["DE"]["State"]["CurrentMeanVector"]  += rgxmean[i];
 for (size_t i = 0; i < _k->N; i++) js["DE"]["State"]["PreviousMeanVector"] += rgxoldmean[i];
 for (size_t i = 0; i < _k->N; i++) js["DE"]["State"]["BestEverVector"]     += rgxbestever[i];
 for (size_t i = 0; i < _k->N; i++) js["DE"]["State"]["CurrentBestVector"]  += curBestVector[i];

 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->N; j++) js["DE"]["State"]["Samples"][i][j] = samplePopulation[i*_k->N + j];
 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->N; j++) js["DE"]["State"]["Candidates"][i][j] = candidates[i*_k->N + j];

 return js;
}



void DE::setConfiguration(nlohmann::json& js)
{
 _s                             = consume(js, { "DE", "Sample Count" }, KORALI_NUMBER); // 5x - 10x Dim
 _crossoverRate                 = consume(js, { "DE", "Crossover Rate" }, KORALI_NUMBER, std::to_string(0.9)); // Rainer Storn [1996]
 _mutationRate                  = consume(js, { "DE", "Mutation Rate" }, KORALI_NUMBER, std::to_string(0.5));  // Rainer Storn [1996]
 _objective                     = consume(js, { "DE", "Objective" }, KORALI_STRING, "Maximize");
 _maxResamplings                = consume(js, { "DE", "Max Resamplings" }, KORALI_NUMBER, std::to_string(1e6));
 _acceptRule                    = consume(js, { "DE", "Accept Rule" }, KORALI_STRING, "Iterative");

 if(_s < 4)  { fprintf( stderr, "[Korali] %s Error: Sample Count must be larger 3 (is %zu)\n", _name.c_str(), _s); exit(-1); }
 if(_crossoverRate <= 0.0 || _crossoverRate > 1.0 )  { fprintf( stderr, "[Korali] %s Error: Invalid Crossover Rate, must be in (0,1] (is %f)\n", _name.c_str(), _crossoverRate); exit(-1); }
 if(_mutationRate < 0.0 || _mutationRate > 2.0 )  { fprintf( stderr, "[Korali] %s Error: Invalid Mutation Rate, must be in [0,2] (is %f)\n", _name.c_str(), _mutationRate); exit(-1); }
 if( (_acceptRule != "Best") && (_acceptRule != "Greedy") && (_acceptRule != "Iterative") && (_acceptRule != "Paired") )  
 { fprintf( stderr, "[Korali] %s Error: Invalid Accept Rule, must be 'Best', 'Greedy', 'Iterative' or 'Paired' (is %s)\n", _name.c_str(), _acceptRule.c_str()); exit(-1); }
 
 _fitnessSign   = 0;
 if(_objective == "Maximize") _fitnessSign = 1;
 if(_objective == "Minimize") _fitnessSign = -1;
 if(_fitnessSign == 0)  { fprintf( stderr, "[Korali] %s Error: Invalid setting for Objective: %s\n", _name.c_str(), _objective.c_str()); exit(-1); }

 // Setting variable information
 _initialMeans         = (double*) calloc(sizeof(double), _k->N);
 _initialStdDevs       = (double*) calloc(sizeof(double), _k->N);
 _initialMeanDefined   = (bool*) calloc(sizeof(bool), _k->N);
 _initialStdDevDefined = (bool*) calloc(sizeof(bool), _k->N);
 
 _lowerBounds = (double*) calloc(sizeof(double), _k->N);
 _upperBounds = (double*) calloc(sizeof(double), _k->N);

 _variableLogSpace = (bool*) calloc(sizeof(bool), _k->N);

 for (size_t d = 0; d < _k->N; ++d)
 {
  _varNames.push_back(consume(js["Variables"][d], { "Name" }, KORALI_STRING, "X"+std::to_string(d)));
  
  _lowerBounds[d] = consume(js["Variables"][d], { "DE", "Lower Bound" }, KORALI_NUMBER);
  _upperBounds[d] = consume(js["Variables"][d], { "DE", "Upper Bound" }, KORALI_NUMBER);

  _initialMeanDefined[d]   = isDefined(js["Variables"][d], { "DE", "Initial Mean" });
  _initialStdDevDefined[d] = isDefined(js["Variables"][d], { "DE", "Initial Standard Deviation" });
  if( _initialStdDevDefined[d] != _initialMeanDefined[d]) 
    { fprintf( stderr, "[Korali] %s Error: Invalid setting for Initial Mean and Initial Stanard Deviation for variable %s (both must be defined or none)\n", _name.c_str(), _k->_variables[d]->_name.c_str()); exit(-1); }

  if (_initialMeanDefined[d])   _initialMeans[d]   = consume(js["Variables"][d], { "DE", "Initial Mean" }, KORALI_NUMBER);
  if (_initialStdDevDefined[d]) _initialStdDevs[d] = consume(js["Variables"][d], { "DE", "Initial Standard Deviation" }, KORALI_NUMBER);

  _variableLogSpace[d] = consume(js["Variables"][d], { "DE", "Log Space" }, KORALI_BOOLEAN, "false");
 }

 // Setting termination criteria
 _isTermCondMaxGenerations        = consume(js, { "DE", "Termination Criteria", "Max Generations", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMaxGenerations          = consume(js, { "DE", "Termination Criteria", "Max Generations", "Value" }, KORALI_NUMBER, std::to_string(1000));
 _isTermCondMaxFitnessEvaluations = consume(js, { "DE", "Termination Criteria", "Max Model Evaluations", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMaxFitnessEvaluations   = consume(js, { "DE", "Termination Criteria", "Max Model Evaluations", "Value" }, KORALI_NUMBER, std::to_string(std::numeric_limits<size_t>::max()));
 _isTermCondFitness               = consume(js, { "DE", "Termination Criteria", "Fitness", "Active" }, KORALI_BOOLEAN, "true");
 _termCondFitness                 = consume(js, { "DE", "Termination Criteria", "Fitness", "Value" }, KORALI_NUMBER, std::to_string(std::numeric_limits<double>::max()));
 _isTermCondFitnessDiffThreshold  = consume(js, { "DE", "Termination Criteria", "Fitness Diff Threshold", "Active" }, KORALI_BOOLEAN, "true");
 _termCondFitnessDiffThreshold    = consume(js, { "DE", "Termination Criteria", "Fitness Diff Threshold", "Value" }, KORALI_NUMBER, std::to_string(0.0));
 _isTermCondMinDeltaX             = consume(js, { "DE", "Termination Criteria", "Min DeltaX", "Active" }, KORALI_BOOLEAN, "true");
 _termCondMinDeltaX               = consume(js, { "DE", "Termination Criteria", "Min DeltaX", "Value" }, KORALI_NUMBER, std::to_string(0.0));

}


void DE::setState(nlohmann::json& js)
{
 currentGeneration    = js["DE"]["State"]["Current Generation"];
 bestEver             = js["DE"]["State"]["BestEverFunctionValue"];
 prevBest             = js["DE"]["State"]["PreviousBestFunctionValue"];
 countevals           = js["DE"]["State"]["EvaluationCount"];

 for (size_t i = 0; i < _k->N; i++) rgxmean[i]       = js["DE"]["State"]["CurrentMeanVector"][i];
 for (size_t i = 0; i < _k->N; i++) rgxoldmean[i]    = js["DE"]["State"]["PreviousMeanVector"][i];
 for (size_t i = 0; i < _k->N; i++) rgxbestever[i]   = js["DE"]["State"]["BestEverVector"][i];
 for (size_t i = 0; i < _s; i++) fitnessVector[i]    = js["DE"]["State"]["FunctionValues"][i];
 for (size_t i = 0; i < _s; i++) oldFitnessVector[i] = js["DE"]["State"]["OldFunctionValues"][i];
 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->N; j++) samplePopulation[i*_k->N + j] = js["DE"]["State"]["Samples"][i][j];
 for (size_t i = 0; i < _s; i++) for (size_t j = 0; j < _k->N; j++) candidates[i*_k->N + j] = js["DE"]["State"]["Candidates"][i][j];
}


/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/


void DE::run()
{
 if (_k->_verbosity >= KORALI_MINIMAL) {
   printf("[Korali] Starting %s (Objective: %s).\n", _name.c_str(), _objective.c_str());
   printf("--------------------------------------------------------------------\n");
 }
 
 startTime = std::chrono::system_clock::now();
 initSamples();
 _k->saveState(currentGeneration);

 while(!checkTermination())
 {
   t0 = std::chrono::system_clock::now();
   prepareGeneration();
   evaluateSamples(); 
   updateSolver(fitnessVector);
   currentGeneration++;

   t1 = std::chrono::system_clock::now();

   printGeneration();
   _k->saveState(currentGeneration);
 }

 endTime = std::chrono::system_clock::now();

 printFinal();

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
        mutateSingle(i);
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
    size_t a, b, c;
    do{ a = _uniformGenerator->getRandomNumber()*_s; } while(a == sampleIdx);
    do{ b = _uniformGenerator->getRandomNumber()*_s; } while(b == sampleIdx || b == a);
    do{ c = _uniformGenerator->getRandomNumber()*_s; } while(c == sampleIdx || c == a || c == b);

    size_t e = _uniformGenerator->getRandomNumber()*_k->N;
    for(size_t d = 0; d < _k->N; ++d)
    {
      if(_uniformGenerator->getRandomNumber() < _crossoverRate || d == _k->N-1)
          candidates[sampleIdx*_k->N+d] = samplePopulation[c*_k->N+d]+_mutationRate*(samplePopulation[a*_k->N+d]-samplePopulation[b*_k->N+d]);
      else
          candidates[sampleIdx*_k->N+d] = samplePopulation[sampleIdx*_k->N+d];
    }
}


bool DE::isFeasible(size_t sampleIdx) const
{
  for(size_t d = 0; d < _k->N; ++d) if ( (candidates[sampleIdx*_k->N+d] < _lowerBounds[d]) || (candidates[sampleIdx*_k->N+d] > _upperBounds[d])) return false;
  return true;
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
    size_t minIndex      = maxIdx(fitnessVector, _s);
    prevFunctionValue    = currentFunctionValue;
    currentFunctionValue = fitnessVector[minIndex];
    for(size_t d = 0; d < _k->N; ++d) curBestVector[d] = candidates[minIndex*_k->N+d];
    
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
                for(size_t d = 0; d < _k->N; ++d) samplePopulation[minIndex*_k->N+d] = candidates[minIndex*_k->N+d];
                break;

            case str2int("Greedy") : // update all samples better than bestEver
                for(size_t i = 0; i < _s; ++i) if(fitnessVector[i] > bestEver)
                    for(size_t d = 0; d < _k->N; ++d) samplePopulation[i*_k->N+d] = candidates[i*_k->N+d];
                bestEver = currentFunctionValue;
                break;

            case str2int("Iterative") : // iteratibely accept samples and update bestEver
                for(size_t i = 0; i < _s; ++i) if(fitnessVector[i] > bestEver)
                 { for(size_t d = 0; d < _k->N; ++d) samplePopulation[i*_k->N+d] = candidates[i*_k->N+d]; bestEver = fitnessVector[i]; }
                break;
 
            case str2int("Paired") : // accept if mutation better than parent
                for(size_t i = 0; i < _s; ++i) if(fitnessVector[i] > oldFitnessVector[i])
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
    
}


bool DE::checkTermination()
{

 bool terminate = false;

 if ( _isTermCondFitness && (currentGeneration > 1) && (bestEver >= _termCondFitness) )
 {
  terminate = true;
  sprintf(_terminationReason, "Fitness Value (%+6.3e) > (%+6.3e).",  bestEver, _termCondFitness);
 }

 double range = fabs(currentFunctionValue - prevFunctionValue);
 if ( _isTermCondFitnessDiffThreshold && (currentGeneration > 1) && (range <= _termCondFitnessDiffThreshold) )
 {
  terminate = true;
  sprintf(_terminationReason, "Fitness Diff Threshold (%+6.3e) < (%+6.3e).",  range, _termCondFitnessDiffThreshold);
 }
 
 if ( _isTermCondMinDeltaX && (currentGeneration > 1) )
 {
   size_t cTemp = 0;
   for(size_t d = 0; d < _k->N; ++d) cTemp += (fabs(rgxmean[d] - rgxoldmean[d]) < _termCondMinDeltaX) ? 1 : 0;
   if (cTemp == _k->N) 
   {
    terminate = true;
    sprintf(_terminationReason, "Mean changes < %+6.3e for all variables.", _termCondMinDeltaX);
   }
 }

 if( _isTermCondMaxFitnessEvaluations && (countevals >= _termCondMaxFitnessEvaluations) )
 {
  terminate = true;
  sprintf(_terminationReason, "Conducted %lu function evaluations >= (%lu).", countevals, _termCondMaxFitnessEvaluations);
 }

 if( _isTermCondMaxGenerations && (currentGeneration >= _termCondMaxGenerations) )
 {
  terminate = true;
  sprintf(_terminationReason, "Maximum number of Generations reached (%lu).", _termCondMaxGenerations);
 }

 return terminate;
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


void DE::printGeneration() const
{
 if (currentGeneration % _k->_outputFrequency != 0) return;

 if (_k->_verbosity >= KORALI_MINIMAL)
   printf("[Korali] Generation %ld - Duration: %fs (Total Elapsed Time: %fs)\n", currentGeneration, std::chrono::duration<double>(t1-t0).count(), std::chrono::duration<double>(t1-startTime).count());

 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Current Function Value: Max = %+6.3e - Best = %+6.3e\n", currentFunctionValue, bestEver);
  // TODO: sth like width of samples: printf("[Korali] Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", minEW, maxEW);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), rgxmean[d], rgxbestever[d]);
  printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
  printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
 }

 if (_k->_verbosity >= KORALI_NORMAL)
   printf("--------------------------------------------------------------------\n");

}


void DE::printFinal() const
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] %s Finished\n", _name.c_str());
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), bestEver);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t d = 0; d < _k->N; ++d) printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), rgxbestever[d]);
    printf("[Korali] Number of Function Evaluations: %zu\n", countevals);
    printf("[Korali] Number of Infeasible Samples: %zu\n", countinfeasible);
    printf("[Korali] Stopping Criterium: %s\n", _terminationReason);
    printf("[Korali] Total Elapsed Time: %fs\n", std::chrono::duration<double>(endTime-startTime).count());
    printf("--------------------------------------------------------------------\n");
 }
}
