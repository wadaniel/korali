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
Default Value: "Maximize"
Default Enabled:
Description:
Specifies whether the problem evaluation is to be minimized or maximized.
******************************************************************************/
std::string _objective;

/******************************************************************************
Setting Name: Sample Count
Type: Solver Setting
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
Default Value: 0.9
Default Enabled:
Description:
Controls the rate at which dimensions from samples are mixed (must be in [0,1]).
******************************************************************************/
double _crossoverRate;

/******************************************************************************
Setting Name: Mutation Rate
Type: Solver Setting
Default Value: 0.5
Default Enabled:
Description:
Controls the scaling of the vector differentials (must be in [0,2], preferably < 1).
******************************************************************************/
double _mutationRate;

/******************************************************************************
Setting Name: Result Output Frequency
Type: Solver Setting
Default Value: 1
Default Enabled:
Description:
Specifies the output frequency of intermediate result files.
******************************************************************************/
size_t _resultOutputFrequency;

/******************************************************************************
Setting Name: Terminal Output Frequency
Type: Solver Setting
Default Value: 1
Default Enabled:
Description:
Specifies the output frequency onto the terminal screen.
******************************************************************************/
size_t _terminalOutputFrequency;

/******************************************************************************
Setting Name: Mutation Rule
Type: Solver Setting
Default Value: "Default"
Default Enabled:
Description:
Controls the Mutation Rate: "Default" (rate is fixed) or "Self Adaptive" 
(udpating rule in [Brest2006]).
******************************************************************************/
std::string _mutationRule;

/******************************************************************************
Setting Name: Parent Selection Rule
Type: Solver Setting
Default Value: "Random"
Default Enabled:
Description:
Controls the selection of the parent vecor: "Random" or "Best" (best sample 
from previous generation).
******************************************************************************/
std::string _parentSelectionRule;

/******************************************************************************
Setting Name: Accept Rule
Type: Solver Setting
Default Value: "Greedy"
Default Enabled:
Description:
Sets the sample accept rule after mutation and evaluation: "Best", "Greedy", 
"Iterative" or "Improved".
******************************************************************************/
std::string _acceptRule;

/******************************************************************************
Setting Name: Fix Infeasible
Type: Solver Setting
Default Value: true
Default Enabled:
Description:
If set true, Korali samples a random sample between Parent and the voiolated 
boundary. If set false, infeasible samples are mutated again until feasible.
******************************************************************************/
bool _fixInfeasible;

/******************************************************************************
Setting Name: Max Generations
Type: Termination Criterion
Default Value: 1e3
Default Enabled: true
Description:
Maximal number of generations to run.
******************************************************************************/
size_t _termCondMaxGenerations;
bool   _termCondMaxGenerationsEnabled;

/******************************************************************************
Setting Name: Max Resamplings
Type: Termination Criterion
Default Value: 1e9
Default Enabled: true
Description:
Max number of mutations per sample per generation if infeasible (only relevant 
if Fix Infeasible is set False).
******************************************************************************/
size_t _termCondMaxInfeasibleResamplings;
bool _termCondMaxInfeasibleResamplingsEnabled;
 
/******************************************************************************
Setting Name: Min Fitness
Type: Termination Criterion
Default Value: -INFINITY
Default Enabled: false
Description:
Specifies the target fitness to stop minimization.
******************************************************************************/
double _termCondMinFitness; 
bool   _termCondMinFitnessEnabled;

/******************************************************************************
Setting Name: Max Fitness
Type: Termination Criterion
Default Value: +INFINITY
Default Enabled: false
Description:
Specifies the target fitness to stop maximization.
******************************************************************************/
double _termCondMaxFitness;
bool   _termCondMaxFitnessEnabled;

/******************************************************************************
Setting Name: Min Fitness Diff Threshold
Type: Termination Criterion
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
Default Value: 1e-12
Default Enabled: false
Description:
Specifies the minimal step size of the sample mean from one gneration to another.
******************************************************************************/
double _termCondMinStepSize; 
bool   _termCondMinStepSizeEnabled;
 
/******************************************************************************
Setting Name: Evaluation Sign
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
The sign for the fitness evaluation to determine whether this is a maximization
or minimization operation.
******************************************************************************/
int _evaluationSign;

/******************************************************************************
Setting Name: Fitness Vector
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Objection Function Values
******************************************************************************/
std::vector<double> _fitnessVector;

/******************************************************************************
Setting Name: Previous Fitness Vector
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Objection Function Values from previous evaluations
******************************************************************************/
std::vector<double> _prevfitnessVector;

/******************************************************************************
Setting Name: Sample Population
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Sample coordinate information
******************************************************************************/
std::vector<double> _samplePopulation;

/******************************************************************************
Setting Name: Sample Candidates
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Sample Candidates to evaluate
******************************************************************************/
std::vector<double> _sampleCandidates;

/******************************************************************************
Setting Name: Tansformed Samples
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Sample Candidates in log space
******************************************************************************/
std::vector<double> _transformedSamples;

/******************************************************************************
Setting Name: Is Initialized Sample
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Determines which samples are initialized.
******************************************************************************/
std::vector<bool> _isInitializedSample;

/******************************************************************************
Setting Name: Finished Sample Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Counter of evaluated samples to terminate evaluation.
******************************************************************************/
size_t _finishedSampleCount;

/******************************************************************************
Setting Name: Current Function Value
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best fitness from current generation.
******************************************************************************/
double _currentFunctionValue;

/******************************************************************************
Setting Name: Previous Function Value
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best fitness from previous generation.
******************************************************************************/
double _previousFunctionValue;

/******************************************************************************
Setting Name: Best Sample Index
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best Sample Index.
******************************************************************************/
size_t _bestIndex;

/******************************************************************************
Setting Name: Best Sample Ever
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best Sample Ever.
******************************************************************************/
double _bestEver;

/******************************************************************************
Setting Name: Previous Best Sample Ever
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Previous Best Sample Ever.
******************************************************************************/
double _previousBestEver;

/******************************************************************************
Setting Name: RGX Mean
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Mean Parent
******************************************************************************/
std::vector<double> _rgxMean;

/******************************************************************************
Setting Name: RGX Old Mean
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Old Mean Parent
******************************************************************************/
std::vector<double> _rgxOldMean;

/******************************************************************************
Setting Name: RGX Best Ever
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best RGX
******************************************************************************/
std::vector<double> _rgxBestEver;

/******************************************************************************
Setting Name: Current Best Ever
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Current Best Ever
******************************************************************************/
std::vector<double> _curBestEver;

/******************************************************************************
Setting Name: Max Width
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Max distance between samples per dimension.
******************************************************************************/
std::vector<double> _maxWidth;

/******************************************************************************
Setting Name: Max Width
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Max distance between samples per dimension.
******************************************************************************/
size_t _evaluationCount;

/******************************************************************************
Setting Name: Function Evaluation Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Keeps count of the number of function evaluations so far
******************************************************************************/
size_t _functionEvaluationCount;

/******************************************************************************
Setting Name: Infeasible Sample Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Keeps count of the number of function evaluations so far
******************************************************************************/
size_t _infeasibleSampleCount;

/******************************************************************************
* Variable Settings
******************************************************************************/

struct variableSetting
{

/******************************************************************************
Setting Name: Lower Bound
Type: Variable Setting
Default Value:
Default Enabled:
Description:
Specifies the lower bound for the variable's value. Korali will not generate samples
for which this variable falls below the specified minimum. By default, Korali sets this
value to -Infinity.
******************************************************************************/
double lowerBound;

/******************************************************************************
Setting Name: Upper Bound
Type: Variable Setting
Default Value:
Default Enabled:
Description:
Specifies the upper bound for the variable's value. Korali will not generate samples
for which this variable falls below the specified maximum. By default, Korali sets this
value to +Infinity.
******************************************************************************/
double upperBound;

};

std::vector<variableSetting> _variableSettings;
/******************************************************************************/


 // DE Methods

 Variable* _gaussianGenerator;
 Variable* _uniformGenerator;

 DE();

 void mutateSingle(size_t sampleIdx); /* sample individual */
 bool isFeasible(size_t sampleIdx) const; /* check if sample inside lower & upper bounds */
 void fixInfeasible(size_t sampleIdx); /* force sample inside lower & upper bounds */
 void updateSolver(); /* update states of DE */
 void evaluateSamples(); /* evaluate all samples until done */

 void initSamples();
 void prepareGeneration();
 bool checkTermination() override;

 void initialize() override;
 void finalize() override;

 void runGeneration() override;
 void processSample(size_t sampleId, double fitness) override;

 void setConfiguration() override;
 void getConfiguration() override;
 void printGeneration() override;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVERS_DE_H_
