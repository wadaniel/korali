#ifndef _KORALI_SOLVERS_DE_H_
#define _KORALI_SOLVERS_DE_H_

#include "solvers/base.h"
#include "variables/gaussian.h"
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

 // Runtime Methods (to be inherited from base class in the future)
 void prepareGeneration();
 bool checkTermination();
 void updateDistribution(const double *fitnessVector);
 void run() override;
 void processSample(size_t sampleId, double fitness) override;

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
 void setState(nlohmann::json& js) override;

 private:

 // Korali Runtime Variables
 int _fitnessSign; /* maximizing vs optimizing (+- 1) */
 std::string _objective; /* Maximize or Minimize */ 
 double* _fitnessVector; /* objective function values [_s] */
 double* _samplePopulation; /* sample coordinates [_s x _k->N] */
 size_t _currentGeneration; /* generation count */
 bool* _initializedSample; /* flag to distribute work */
 char _terminationReason[500]; /* buffer for exit reason */

 size_t _finishedSamples; /* counter of evaluated samples to terminate evaluation */
 size_t _s; /* number of samples per generation */

 // Stop conditions
 size_t _termCondMaxGenerations; // Max number of generations.
 size_t _termCondMaxFitnessEvaluations;   // Defines maximum number of fitness evaluations
 double _termCondFitness; // Defines the maximum fitness allowed, otherwise it stops
 double _termCondFitnessDiffThreshold; // Defines minimum function value differences before stopping
 double _termCondMinDeltaX; // Defines minimum delta of input parameters among generations before it stops.
 double _termCondTolUpXFactor; // Defines the minimum fitness allowed, otherwise it stops //TODO: check again what it is in CMAES(DW)
 bool _isTermCondMaxGenerations, _isTermCondMaxFitnessEvaluations, _isTermCondFitness,
      _isTermCondFitnessDiffThreshold, _isTermCondMinDeltaX; // flgs to activate termination criteria

 // Private DE-Specific Variables
 //Variable::Gaussian* _gaussianGenerator;
 double bestEver; /* best ever fitness */
 double prevBest; /* best ever fitness from previous generation */
 double *rgxmean; /* mean "parent" */
 double *rgxbestever; /* bestever vector */
 double *curBestVector; /* current best vector */
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */

 double *histFuncValues; /* holding historical best function values */

 size_t countevals; /* Number of function evaluations */
 size_t countinfeasible; /* Number of samples outside of domain given by bounds */


 // Private DE-Specific Methods
 void sampleSingle(size_t sampleIdx); /* sample individual */
 void evaluateSamples(); /* evaluate all samples until done */
 bool isFeasible(size_t sampleIdx) const; /* check if sample inside lower & upper bounds */

 // Private DE-ES-Specific Variables 
 double _crossoverRate;
 double _mutationRate;
 
 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVERS_DE_H_
