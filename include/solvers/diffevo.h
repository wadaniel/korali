#ifndef _KORALI_SOLVERS_DE_H_
#define _KORALI_SOLVERS_DE_H_

#include "solvers/base.h"
#include <chrono>
#include <map>

namespace Korali { namespace Solver {

/******************************************************************************
Module Name: Differential Evolution
Type: Solver, Optimizer
Alias: DE
Description:
******************************************************************************/


class DE : public Base
{
 public:

 size_t resultOutputFrequency;
 size_t terminalOutputFrequency;

 // These are DE-Specific, but could be used for other methods in the future
 double* _lowerBounds;
 double* _upperBounds;
 double* _initialMeans;
 double* _initialStdDevs;
 bool* _initialMeanDefined;
 bool* _initialStdDevDefined;
 bool* _variableLogSpace;

 // Runtime Methods (to be inherited from base class in the future)
 void initSamples();
 void prepareGeneration();
 bool checkTermination() override;
 void updateDistribution(const double *fitnessVector);

 void initialize() override;
 void finalize() override;

 void runGeneration() override;
 void processSample(size_t sampleId, double fitness) override;

 private:

 // Korali Runtime Variables
 int _fitnessSign; /* maximizing vs optimizing (+- 1) */
 std::string _parent; /* Random or Best */
 std::string _mutationRule; /* Default or Self  Adaptive */
 std::string _acceptRule; /* Best, Greedy, Iterative or Improved */
 bool _fixinfeasible; /* fix infeasible sample (no resampling) */ 
 std::string _objective; /* Maximize or Minimize */ 
 double* oldFitnessVector; /* objective function values previous generation [_s] */
 double* fitnessVector; /* objective function values [_s] */
 double* samplePopulation; /* sample coordinates [_s x _k->N] */
 double* candidates; /* candidates to evaluate */
 bool* initializedSample; /* flag to distribute work */
 Variable* _gaussianGenerator;
 Variable* _uniformGenerator;

 size_t _s; /* number of samples per generation */
 size_t finishedSamples; /* counter of evaluated samples to terminate evaluation */

 // Stop conditions
 size_t _maxResamplings; // Max number of resamplings if sample is infeasible
 size_t _termCondMaxGenerations; // Max number of generations
 size_t _termCondMaxFitnessEvaluations;   // Defines maximum number of fitness evaluations
 double _termCondMinFitness; // Defines the minimum fitness allowed, otherwise it stops
 double _termCondMaxFitness; // Defines the maximum fitness allowed, otherwise it stops
 double _termCondFitnessDiffThreshold; // Defines minimum function value differences before stopping
 double _termCondMinDeltaX; // Defines minimum delta of input parameters among generations before it stops.
 bool _isTermCondMaxGenerations, _isTermCondMaxFitnessEvaluations, 
      _isTermCondMinFitness, _isTermCondMaxFitness,
      _isTermCondMinDeltaX, _isTermCondFitnessDiffThreshold; // flgs to activate termination criteria
 
 // Private DE-Specific Variables
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */
 size_t bestIndex; /* index of best sample */
 double bestEver; /* best ever fitness */
 double prevBest; /* best ever fitness from previous generation */
 double *rgxmean; /* mean "parent" */
 double *rgxoldmean; /* mean "parent" previous generation */
 double *rgxbestever; /* bestever vector */
 double *curBestVector; /* current best vector */
 double *histFuncValues; /* holding historical best function values */
 double* maxWidth; /* max distance between samples per dimension */

 size_t countevals; /* Number of function evaluations */
 size_t countinfeasible; /* Number of samples outside of domain given by bounds */

 // Private DE-Specific Methods
 void mutateSingle(size_t sampleIdx); /* sample individual */
 bool isFeasible(size_t sampleIdx) const; /* check if sample inside lower & upper bounds */
 void fixInfeasible(size_t sampleIdx); /* force sample inside lower & upper bounds */
 void updateSolver(const double *fitnessVector); /* update states of DE */
 void evaluateSamples(); /* evaluate all samples until done */

 // Private DE-ES-Specific Variables 
 double _crossoverRate;
 double _mutationRate;
 
 // Helper Methods
 size_t maxIdx(const double *rgd, size_t len) const;
 
 void setConfiguration() override;
 void getConfiguration() override;
 void printGeneration() override;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVERS_DE_H_
