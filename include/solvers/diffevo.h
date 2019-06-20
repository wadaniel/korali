#ifndef _KORALI_SOLVERS_DE_H_
#define _KORALI_SOLVERS_DE_H_

#include "solvers/base.h"
#include <chrono>
#include <map>

namespace Korali { namespace Solver {

class DE : public Base
{
 public:

 // Constructor / Destructor
 DE(nlohmann::json& js, std::string name);
 ~DE();

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
 void run() override;
 void processSample(size_t sampleId, double fitness) override;

 // Serialization Methods
 void getConfiguration(nlohmann::json& js) override;
 void setConfiguration(nlohmann::json& js) override;
 void setState(nlohmann::json& js) override;
 void saveState() const override;

 private:

 // Korali Runtime Variables
 int _fitnessSign; /* maximizing vs optimizing (+- 1) */
 std::string _acceptRule; /* Best, Greedy or Iterative */
 std::string _objective; /* Maximize or Minimize */ 
 double* oldFitnessVector; /* objective function values previous generation [_s] */
 double* fitnessVector; /* objective function values [_s] */
 double* samplePopulation; /* sample coordinates [_s x _k->N] */
 double* candidates; /* candidates to evaluate */
 bool* initializedSample; /* flag to distribute work */
 char _terminationReason[500]; /* buffer for exit reason */
 Variable* _gaussianGenerator;
 Variable* _uniformGenerator;

 size_t _s; /* number of samples per generation */
 size_t currentGeneration; /* generation count */
 size_t finishedSamples; /* counter of evaluated samples to terminate evaluation */

 // Stop conditions
 size_t _maxResamplings; // Max number of resamplings if sample is infeasible
 size_t _termCondMaxGenerations; // Max number of generations
 size_t _termCondMaxFitnessEvaluations;   // Defines maximum number of fitness evaluations
 double _termCondFitness; // Defines the maximum fitness allowed, otherwise it stops
 double _termCondFitnessDiffThreshold; // Defines minimum function value differences before stopping
 double _termCondMinDeltaX; // Defines minimum delta of input parameters among generations before it stops.
 bool _isTermCondMaxGenerations, _isTermCondMaxFitnessEvaluations, _isTermCondFitness,
      _isTermCondMinDeltaX, _isTermCondFitnessDiffThreshold; // flgs to activate termination criteria
 
 // Private DE-Specific Variables
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */
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
 void updateSolver(const double *fitnessVector); /* update states of DE */
 void evaluateSamples(); /* evaluate all samples until done */

 // Private DE-ES-Specific Variables 
 double _crossoverRate;
 double _mutationRate;
 
 // Helper Methods
 size_t maxIdx(const double *rgd, size_t len) const;
 
 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVERS_DE_H_
