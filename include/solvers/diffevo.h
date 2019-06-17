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
 bool* _initialMeanDefined;
 bool* _initialStdDevDefined;
 bool* _variableLogSpace;

 // Runtime Methods (to be inherited from base class in the future)
 void initSamples();
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
 double* samplePopulation; /* sample coordinates [_s x _k->N] */
 double* candidates; /* candidates to evaluate */
 size_t currentGeneration; /* generation count */
 bool* _initializedSample; /* flag to distribute work */
 char _terminationReason[500]; /* buffer for exit reason */
 Variable::Gaussian* _gaussianGenerator;
 Variable::Uniform* _uniformGenerator;

 size_t _finishedSamples; /* counter of evaluated samples to terminate evaluation */
 size_t _s; /* number of samples per generation */

 // Stop conditions
 size_t _termCondMaxGenerations; // Max number of generations.
 size_t _termCondMaxFitnessEvaluations;   // Defines maximum number of fitness evaluations
 double _termCondFitness; // Defines the maximum fitness allowed, otherwise it stops
 double _termCondFitnessDiffThreshold; // Defines minimum function value differences before stopping
 bool _isTermCondMaxGenerations, _isTermCondMaxFitnessEvaluations, _isTermCondFitness,
      _isTermCondFitnessDiffThreshold; // flgs to activate termination criteria

 // Private DE-Specific Variables
 //Variable::Gaussian* _gaussianGenerator;
 double bestEver; /* best ever fitness */
 double prevBest; /* best ever fitness from previous generation */
 double *rgxmean; /* mean "parent" */
 double *rgxold; /* mean "parent" previous generation */
 double *rgxbestever; /* bestever vector */
 double *curBestVector; /* current best vector */
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */

 double *histFuncValues; /* holding historical best function values */

 size_t countevals; /* Number of function evaluations */


 // Private DE-Specific Methods
 void mutateSingle(size_t sampleIdx); /* sample individual */
 void updateSolver(const double *fitnessVector); /* update states of DE */
 void evaluateSamples(); /* evaluate all samples until done */

 // Private DE-ES-Specific Variables 
 double _crossoverRate;
 double _mutationRate;
 
 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVERS_DE_H_
