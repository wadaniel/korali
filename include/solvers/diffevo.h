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
his is an implementation of the *Differential Evolution Algorithm* algorithm,
as published in [Storn1997](https://link.springer.com/article/10.1023/A:1008202821328.

DE optimizes a problem by updating a population of candidate solutions through 
mutation and recombination. The update rules are simple and the objective 
function must not be differentiable. Our implementation includes various adaption 
and updating strategies [Brest2006](https://ieeexplore.ieee.org/document/4016057).

**Requirements:**

+ The *Sample Count* needs to be defined..
+ The *Lower Bound* needs to be defined for each variable.
+ The *Upper Bound* needs to be defined for every variable.
******************************************************************************/


class DE : public Base
{
 public:

/******************************************************************************
Setting Name: Objective
Type: Solver Setting
Format: String
Mandatory: No
Default Value: Maximize
Default Enabled:
Description:
Specifies whether the problem evaluation is to be minimized or maximized.
******************************************************************************/
std::string _objective;

/******************************************************************************
Setting Name: Sample Count
Type: Solver Setting
Format: Integer
Mandatory: Yes
Default Value:
Default Enabled:
Description:
Specifies the number of samples to evaluate per generation (preferably 5-10x 
number of variables).
******************************************************************************/
size_t _sampleCount;

/******************************************************************************
Setting Name: Crossover Rate
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.9
Default Enabled:
Description:
Controls the rate at which dimensions from samples are mixed (must be in [0,1]).
******************************************************************************/
double _crossoverRate;

/******************************************************************************
Setting Name: Mutation Rate
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.5
Default Enabled:
Description:
Controls the scaling of the vector differentials (must be in [0,2], preferably < 1).
******************************************************************************/
double _mutationRate;

/******************************************************************************
Setting Name: Result Output Frequency
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 1
Default Enabled:
Description:
Specifies the output frequency of intermediate result files.
******************************************************************************/
size_t _resultOutputFrequency;

/******************************************************************************
Setting Name: Terminal Output Frequency
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 1
Default Enabled:
Description:
Specifies the output frequency onto the terminal screen.
******************************************************************************/
size_t _terminalOutputFrequency;

/******************************************************************************
Setting Name: Mutation Rule
Type: Solver Setting
Format: String
Mandatory: No
Default Value: Default
Default Enabled:
Description:
Controls the Mutation Rate: "Default" (rate is fixed) or "Self Adaptive" 
(udpating rule in [Brest2006]).
******************************************************************************/
std::string _mutationRule;

/******************************************************************************
Setting Name: Parent Selection Rule
Type: Solver Setting
Format: String
Mandatory: No
Default Value: Random
Default Enabled:
Description:
Controls the selection of the parent vecor: "Random" or "Best" (best sample 
from previous generation).
******************************************************************************/
std::string _parentSelectionRule;

/******************************************************************************
Setting Name: Accept Rule
Type: Solver Setting
Format: String
Mandatory: No
Default Value: Greedy
Default Enabled:
Description:
Sets the sample accept rule after mutation and evaluation: "Best", "Greedy", 
"Iterative" or "Improved".
******************************************************************************/
std::string _acceptRule;

/******************************************************************************
Setting Name: Fix Infeasible
Type: Solver Setting
Format: Boolean
Mandatory: No
Default Value: True
Default Enabled:
Description:
If set true, Korali samples a random sample between Parent and the voiolated 
boundary. If set false, infeasible samples are mutated again until feasible.
******************************************************************************/
bool _fixInfeasible;

/******************************************************************************
Setting Name: Max Resamplings
Type: Termination Criterion
Format: Integer
Mandatory: No
Default Value: 1e9
Default Enabled: True
Description:
Max number of mutations per sample per generation if infeasible (only relevant 
if Fix Infeasible is set False).
******************************************************************************/
size_t _termCondMaxInfeasibleResamplings;
bool _termCondMaxInfeasibleResamplingsEnabled;
 
/******************************************************************************
Setting Name: Min Fitness
Type: Termination Criterion
Format: Real
Mandatory: No
Default Value: -Inf
Default Enabled: false
Description:
Specifies the target fitness to stop minimization.
******************************************************************************/
double _termCondMinFitness; 
bool   _termCondMinFitnessEnabled;

/******************************************************************************
Setting Name: Max Fitness
Type: Termination Criterion
Format: Real
Mandatory: No
Default Value: +Inf
Default Enabled: false
Description:
Specifies the target fitness to stop maximization.
******************************************************************************/
double _termCondMaxFitness;
bool   _termCondMaxFitnessEnabled;

/******************************************************************************
Setting Name: Min Fitness Diff Threshold
Type: Termination Criterion
Format: Real
Mandatory: No
Default Value: 0.0
Default Enabled: false
Description:
Specifies the minimum fitness differential between two consecutive generations 
before stopping execution.
******************************************************************************/
double _termCondMinFitnessDiffThreshold;
bool   _termCondMinFitnessDiffThresholdEnabled;

/******************************************************************************
Setting Name: Min Step Size
Type: Termination Criterion
Format: Real
Mandatory: No
Default Value: 1e-12
Default Enabled: false
Description:
Specifies the minimal step size of the sample mean from one gneration to another.
******************************************************************************/
double _termCondMinStepSize; 
bool   _termCondMinStepSizeEnabled;
 
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
 double* oldFitnessVector; /* objective function values previous generation [_s] */
 double* fitnessVector; /* objective function values [_s] */
 double* samplePopulation; /* sample coordinates [_s x _k->N] */
 double* candidates; /* candidates to evaluate */
 bool* initializedSample; /* flag to distribute work */
 Variable* _gaussianGenerator;
 Variable* _uniformGenerator;

 size_t currentGeneration; /* generation count */
 size_t finishedSamples; /* counter of evaluated samples to terminate evaluation */


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
 
 // Helper Methods
 size_t maxIdx(const double *rgd, size_t len) const;
 
 void setConfiguration() override;
 void getConfiguration() override;
 void printGeneration() override;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVERS_DE_H_
