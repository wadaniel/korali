#ifndef _KORALI_SOLVERS_CCMAES_HPP_
#define _KORALI_SOLVERS_CCMAES_HPP_

#include "solvers/base.hpp"
#include <chrono>
#include <vector>
#include <map>

namespace Korali { namespace Solver {

/******************************************************************************
Module Name: Constrained Covariance Matrix Adaptation Evolution Strategy
Type: Solver, Optimizer
Alias: CCMAES
Description:
### Base CCMAES

This solver also implements the *Constrained Covariance Matrix Adaptation Evolution Strategy*, as published in [Arampatzis2019](https://dl.acm.org/citation.cfm?doid=3324989.3325725).

In an evolution strategy, new candidate solutions are sampled according to a multivariate normal distribution in $\mathbb {R} ^{n}$. Recombination amounts to selecting a new mean value for the distribution. Mutation amounts to adding a random vector, a perturbation with zero mean. Pairwise dependencies between the variables in the distribution are represented by a covariance matrix. The covariance matrix adaptation (CMA) is a method to update the covariance matrix of this distribution.
CCMAES is an extension of [CCMAES](/usage/solvers/optimizers/cmaes/) for constrained optimization problems. It uses the principle of *viability boundaries* to find an initial mean vector for the proposal distribution that does not violate constraints, and secondly it uses a  *constraint handling technique* to efficiently adapt the proposal distribution to the constraints.


** Usage **

```python
  k["Solver"] = "CCMAES"
```

**Base Requirements:**

+ The *Sample Count* $\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.

** Constraint Requirements:**

+ At least one *Constraint function* defined.
+ The *Viability Sample Count* $via-\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.
******************************************************************************
Plotting:
Plotting for CCMAES should include clear markings of rejected samples, or
showing the sample space that does not fit the constraints?
******************************************************************************/

class CCMAES : public Base
{
 public:

/******************************************************************************
Setting Name: Sample Count
Type: Solver Setting
Default Value: 32
Default Enabled:
Description:
Specifies the number of samples to evaluate per generation (preferably 
4+3*log(N) number of variables).
******************************************************************************/
size_t _sampleCount;

/******************************************************************************
Setting Name: Mu Value
Type: Solver Setting
Default Value: _sampleCount*0.5
Default Enabled:
Description:
Number of best samples used to update the covariance matrix and the mean.
******************************************************************************/
size_t _muValue;

/******************************************************************************
Setting Name: Mu Type
Type: Solver Setting
Default Value: "Logarithmic"
Default Enabled:
Description:
Weights given to the Mu best values to update the covariance matrix and the mean.
******************************************************************************/
std::string _muType;

/******************************************************************************
Setting Name: Initial Sigma Cumulation Factor
Type: Solver Setting
Default Value: -1.0
Default Enabled:
Description:
Controls the learning rate of the conjugate evolution path.
******************************************************************************/
double _initialSigmaCumulationFactor;

/******************************************************************************
Setting Name: Initial Damp Factor
Type: Solver Setting
Default Value: -1.0
Default Enabled:
Description:
Controls the updates of the covariance matrix scaling factor.
******************************************************************************/
double _initialDampFactor;

/******************************************************************************
Setting Name: Is Sigma Bounded
Type: Solver Setting
Default Value: false
Default Enabled:
Description:
Sets an upper bound for the covariance matrix scaling factor. The upper bound 
is given by the average of the initial standard deviation of the variables
******************************************************************************/
bool _isSigmaBounded;

/******************************************************************************
Setting Name: Initial Cumulative Covariance
Type: Solver Setting
Default Value: -1.0
Default Enabled:
Description:
Controls the learning rate of the evolution path for the covariance update
(must be in (0,1]).
******************************************************************************/
double _initialCumulativeCovariance;

/******************************************************************************
Setting Name: Is Diagonal
Type: Solver Setting
Default Value: false
Default Enabled:
Description:
Covariance matrix updates will be optimized for diagonal matrices.
******************************************************************************/
bool _isDiagonal;

/******************************************************************************
Setting Name: Viability Sample Count
Type: Solver Setting
Default Value: 2
Default Enabled:
Description:
Specifies the number of samples per generation during the viability 
regime, i.e. during the search for a mean vector not violating the constraints.
******************************************************************************/
size_t _viabilitySampleCount;

/******************************************************************************
Setting Name: Viability Mu
Type: Solver Setting
Default Value: 1
Default Enabled:
Description:
Number of best samples used to update the covariance matrix and the mean 
during the viability regime.
******************************************************************************/
size_t _viabilityMu;

/******************************************************************************
Setting Name: Max Covariance Matrix Corrections
Type: Solver Setting
Default Value: 1e6
Default Enabled:
Description:
Max number of covairance matrix adaptions per generation during the constraint 
handling loop.
******************************************************************************/
size_t _maxCovarianceMatrixCorrections;
 
/******************************************************************************
Setting Name: Target Success Rate
Type: Solver Setting
Default Value: 0.1818
Default Enabled:
Description:
Controls the updates of the covariance matrix scaling factor during the
viability regime.
******************************************************************************/
double _targetSuccessRate;

/******************************************************************************
Setting Name: Covariance Matrix Adaption Strength
Type: Solver Setting
Default Value: 0.1
Default Enabled:
Description:
Controls the covariane matrix adaption strength if samples violate constraints.
******************************************************************************/
double _covarianceMatrixAdaptionStrength;

/******************************************************************************
Setting Name: Normal Vector Learning Rate
Type: Solver Setting
Default Value: -1.0
Default Enabled:
Description:
Learning rate of constraint normal vectors (must be in (0, 1]).
******************************************************************************/
double _normalVectorLearningRate;

/******************************************************************************
Setting Name: Global Success Learning Rate
Type: Solver Setting
Default Value: 0.2
Default Enabled:
Description:
Learning rate of success probability of objective function improvements. 
Required for covariance matrix scaling factor update during viability regime.
******************************************************************************/
double _globalSuccessLearningRate;

/******************************************************************************
Setting Name: Max Infeasible Resampling
Type: Termination Criterion
Default Value: 1e9
Default Enabled: true
Description:
Number of resamplings per candidate per generation if sample is outside of 
Lower and Upper Bound. 
******************************************************************************/
size_t _termCondMaxInfeasibleResamplings;
bool   _termCondMaxInfeasibleResamplingsEnabled;

/******************************************************************************
Setting Name: Min Fitness
Type: Termination Criterion
Default Value: -std::numeric_limits<float>::max()
Default Enabled: false
Description:
Specifies the target fitness to stop minimization.
******************************************************************************/
double _termCondMinFitness;
bool   _termCondMinFitnessEnabled;

/******************************************************************************
Setting Name: Max Fitness
Type: Termination Criterion
Default Value: std::numeric_limits<float>::max()
Default Enabled: false
Description:
Specifies the target fitness to stop maximization.
******************************************************************************/
double _termCondMaxFitness;
bool   _termCondMaxFitnessEnabled;

/******************************************************************************
Setting Name: Min Fitness Diff Threshold
Type: Termination Criterion
Default Value: 1e-9
Default Enabled: true
Description:
Specifies the minimum fitness differential between two consecutive generations 
before stopping execution.
******************************************************************************/
double _termCondMinFitnessDiffThreshold;
bool   _termCondMinFitnessDiffThresholdEnabled;

/******************************************************************************
Setting Name: Min Standard Deviation
Type: Termination Criterion
Default Value: 1e-12
Default Enabled: false
Description:
Specifies the minimal standard deviation per dimension of the proposal.
******************************************************************************/
double _termCondMinStandardDeviation;
bool   _termCondMinStandardDeviationEnabled;

/******************************************************************************
Setting Name: Max Standard Deviation
Type: Termination Criterion
Default Value: 1e18
Default Enabled: false
Description:
Specifies the maximal standard deviation per dimension of the proposal.
******************************************************************************/
double _termCondMaxStandardDeviation;
bool   _termCondMaxStandardDeviationEnabled;

/******************************************************************************
Setting Name: Max Condition Covariance Matrix
Type: Termination Criterion
Default Value: 1e18
Default Enabled: false
Description:
Specifies the maximum condition of the covariance matrix
******************************************************************************/
double _termCondMaxCovMatrixCondition;
bool   _termCondMaxCovMatrixConditionEnabled;

/******************************************************************************
Setting Name: Min Standard Deviation Step Factor
Type: Termination Criterion
Default Value: 1e-18
Default Enabled: false
Description:
Specifies a scaling factor under which the standard deviation does not change
in the direction of the eigenvectors.
******************************************************************************/
double _termCondMinStandardDeviationStepFactor;
bool   _termCondMinStandardDeviationStepFactorEnabled;

/******************************************************************************
Setting Name: Is Viability Regime
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
True if mean is outside feasible domain.
******************************************************************************/
bool _isViabilityRegime;

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
Setting Name: Sample Population
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Sample coordinate information
******************************************************************************/
std::vector<double> _samplePopulation;

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
Setting Name: Current Sample Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of samples active ( _s or _via_s ).
******************************************************************************/
size_t _currentSampleCount;

/******************************************************************************
Setting Name: Current Sample Mu
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of samples active ( _mu or _mu_s ).
******************************************************************************/
size_t _currentSampleMu;

/******************************************************************************
Setting Name: Mu Weights
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Weights for each of the Mu samples
******************************************************************************/
std::vector<double> _muWeights;

/******************************************************************************
Setting Name: Effective Mu
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Variance effective selection mass
******************************************************************************/
double _effectiveMu;

/******************************************************************************
Setting Name: Sigma Cumulation Factor
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
increment for sigma, default calculated from muEffective and dimension 
(must be in (0,1) ).
******************************************************************************/
double _sigmaCumulationFactor;

/******************************************************************************
Setting Name: Damp Factor
Type: Internal Attribute
Default Value: -1.0
Default Enabled:
Description:
Dampening parameter controls step size adaption (must be in (0,1) ).
******************************************************************************/
double _dampFactor;

/******************************************************************************
Setting Name: Cumulative Covariance
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Default calculated from dimension.
******************************************************************************/
double _cumulativeCovariance;

/******************************************************************************
Setting Name: Chi Square Number
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
expectation of $||N(0,I)||^2$
******************************************************************************/
double _chiN;

/******************************************************************************
Setting Name: Covariance Eigenvalue Evaluation Frequency
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Establishes how frequently the eigenvalues are updated.
******************************************************************************/
size_t _covarianceEigenEvalFreq;

/******************************************************************************
Setting Name: Sigma
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Determines the step size.
******************************************************************************/
double _sigma;

/******************************************************************************
Setting Name: Trace
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Variable to init sigma (or set upper bound).
******************************************************************************/
double _trace;

/******************************************************************************
Setting Name: Best Ever Value
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best ever value found.
******************************************************************************/
double _bestEverValue;

/******************************************************************************
Setting Name: Previous Best Ever Value
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best ever value as of previous generation.
******************************************************************************/
double _previousBestEverValue;

/******************************************************************************
Setting Name: Mean Proposal
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Mean of proposal distribution.
******************************************************************************/
std::vector<double> _mean;

/******************************************************************************
Setting Name: Best Ever Sample
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Coordinates of best ever sample.
******************************************************************************/
std::vector<double> _bestEverSample;

/******************************************************************************
Setting Name: Current Best Sample
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Coorindates of current best sample.
******************************************************************************/
std::vector<double> _currentBestSample;

/******************************************************************************
Setting Name: Sorting Index
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Sorted indeces of samples according to their fitness value.
******************************************************************************/
std::vector<size_t> _sortingIndex;

/******************************************************************************
Setting Name: Current Best Value
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best value of current generation.
******************************************************************************/
double _currentBestValue;

/******************************************************************************
Setting Name: Previous Best Value
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best fitness previous generation.
******************************************************************************/
double _previousBestValue;

/******************************************************************************
Setting Name: Covariance Matrix
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
(Unscaled) covariance Matrix of proposal distribution.
******************************************************************************/
std::vector<double> _C;

/******************************************************************************
Setting Name: Auxiliar Covariance Matrix
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Temporary Storage for Covariance Matrix
******************************************************************************/
std::vector<double> _Ctmp;

/******************************************************************************
Setting Name: Covariance Eigenvector Matrix
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Matrix with eigenvectors in columns.
******************************************************************************/
std::vector<double> _B;

/******************************************************************************
Setting Name: Auxiliar Covariance Eigenvector Matrix
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Temporary Storage for Matrix with eigenvectors in columns.
******************************************************************************/
std::vector<double> _Btmp;

/******************************************************************************
Setting Name: Axis Lengths
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Axis lengths (sqrt(Evals))
******************************************************************************/
std::vector<double> _axisLengths;

/******************************************************************************
Setting Name: Temporary Axis Lengths
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Temporary storage for Axis lengths
******************************************************************************/
std::vector<double> _tmpAxisLengths;

/******************************************************************************
Setting Name: Random Number Storage
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Temporary storage for Random Number Generation
******************************************************************************/
std::vector<double> _Z;

/******************************************************************************
Setting Name: BDZ Matrix
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Temporary storage to hold B*D*randn()
******************************************************************************/
std::vector<double> _BDZ;

/******************************************************************************
Setting Name: Auxiliar Storage for BDZ
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Auxiliar B*D*z
******************************************************************************/
std::vector<double> _BDZtmp;

/******************************************************************************
Setting Name: Mean Update
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Vector pointing from old mean to mean
******************************************************************************/
std::vector<double> _meanUpdate;

/******************************************************************************
Setting Name: Evolution Path
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Evolution path for Covariance Matrix update.
******************************************************************************/
std::vector<double> _evolutionPath;

/******************************************************************************
Setting Name: Conjugate Evolution Path
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Conjugate evolution path for Covariance Matrix update for sigma update.
******************************************************************************/
std::vector<double> _conjugateEvolutionPath;

/******************************************************************************
Setting Name: Conjugate Evolution Path L2 Norm
Type: Internal Attribute
Default Value:
Default Enabled:
Description:

******************************************************************************/
double _conjugateEvolutionPathL2Norm;

/******************************************************************************
Setting Name: Previous Mean
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Mean "parent" on previous generation.
******************************************************************************/
std::vector<double> _previousMean;

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
Setting Name: Maximum Diagonal Covariance Matrix Element
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Maximum diagonal element of the Covariance Matrix.
******************************************************************************/
double _maxDiagCElement;

/******************************************************************************
Setting Name: Minimum Diagonal Covariance Matrix Element
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Minimum diagonal element of the Covariance Matrix.
******************************************************************************/
double _minDiagCElement;

/******************************************************************************
Setting Name: Maximum Covariance Eigenvalue
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Maximum Covariance Matrix Eigenvalue
******************************************************************************/
double _maxCovarianceEigenvalue;

/******************************************************************************
Setting Name: Minimum Covariance Eigenvalue
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Minimum Covariance Matrix Eigenvalue
******************************************************************************/
double _minCovarianceEigenvalue;

/******************************************************************************
Setting Name: Is Eigensystem Updated
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Flag determining if the covariance eigensystem is up to date.
******************************************************************************/
bool _isEigenSystemUpdate;

// Private CCCMAES-Specific Variables

/******************************************************************************
Setting Name: Viability Indicator
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Evaluation of each constraint for each sample.
******************************************************************************/
std::vector<std::vector<bool>> _viabilityIndicator;

/******************************************************************************
Setting Name: Constraints Defined
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
True if the number of constraints is higher than zero
******************************************************************************/
bool _constraintsDefined;

/******************************************************************************
Setting Name: Covariance Matrix Adaption Factor
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
This is the $\beta$ factor that indicates how fast the covariance matrix
is adapted.
******************************************************************************/
double _beta;

/******************************************************************************
Setting Name: Best Valid Sample
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Best sample without constraint violations (otherwise -1).
******************************************************************************/
int _bestValidSample;

/******************************************************************************
Setting Name: Global Success Rate
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Estimated Global Success Rate, required for calibration of covariance matrix
scaling factor updates.
******************************************************************************/
double _globalSuccessRate;

/******************************************************************************
Setting Name: Viability Function Value
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Viability Function Value
******************************************************************************/
double _viabilityValue;

/******************************************************************************
Setting Name: Resampled Parameter Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of resampled parameters due constraint violation.
******************************************************************************/
size_t _resampleCount;

/******************************************************************************
Setting Name: Covariance Matrix Adaptation Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of Covariance Matrix Adaptations
******************************************************************************/
size_t _adaptationCount;

/******************************************************************************
Setting Name: Constraint Evaluation Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of constraint evaluations.
******************************************************************************/
size_t _constraintEvaluationCount;

/******************************************************************************
Setting Name: Viability Boundaries
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Viability Boundaries
******************************************************************************/
std::vector<double> _viabilityBoundaries;

/******************************************************************************
Setting Name: Viability Improvement
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Sample evaluations larger than fviability.
******************************************************************************/
std::vector<bool> _viabilityImprovement;

/******************************************************************************
Setting Name: Maximum Violation Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Maximal amount of constraint violations.
******************************************************************************/
size_t _maxViolationCount;

/******************************************************************************
Setting Name: Sample Constraint Violation Counts
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Maximal amount of constraint violations.
******************************************************************************/
std::vector<size_t> _sampleViolationCounts;

/******************************************************************************
Setting Name: Constraint Evaluations
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Constraint Evaluations $$e$$
******************************************************************************/
std::vector<std::vector<double>> _constraintEvaluations;

/******************************************************************************
Setting Name: Normal Constraint Approximation
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Normal approximation of constraints
******************************************************************************/
std::vector<std::vector<double>> _constraintNormal;

/******************************************************************************
Setting Name: Best Constraint Evaluations
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Constraint evaluations for best ever.
******************************************************************************/
std::vector<double> _bestEverConstraintEvaluation;

/******************************************************************************
Setting Name: Has Discrete Variables
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Flag indicating if at least one of the variables is discrete.
******************************************************************************/
bool _hasDiscreteVariables;

/******************************************************************************
Setting Name: Discrete Mutations
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Vector storing discrete mutations, required for covariance matrix update.
******************************************************************************/
std::vector<double> _discreteMutations;

/******************************************************************************
Setting Name: Number Discrete Mutations
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of discrete mutations in current generation.
******************************************************************************/
size_t _numDiscreteMutations;

/******************************************************************************
Setting Name: Granularity
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Flag determining if the covariance eigensystem is up to date.
******************************************************************************/
std::vector<double> _granularity;

/******************************************************************************
Setting Name: Number Masking Matrix Entries
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of nonzero entries on diagonal in Masking Matrix
******************************************************************************/
size_t _numMaskingMatrixEntries;

/******************************************************************************
Setting Name: Masking Matrix 
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Diagonal Matrix signifying where an integer mutation may be conducted.
******************************************************************************/
std::vector<double> _maskingMatrix;

/******************************************************************************
Setting Name: Masking Matrix Sigma
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Diagonal Matrix signifying where an integer mutation may be conducted.
******************************************************************************/
std::vector<double> _maskingMatrixSigma;

/******************************************************************************
Setting Name: Chi Square Number Discrete Mutations
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
expectation of $||N(0,I^S)||^2$ for discrete mutations
******************************************************************************/
double _chiS;


/******************************************************************************
* Variable Settings
******************************************************************************/

struct variableSetting
{

/******************************************************************************
Setting Name: Lower Bound
Type: Variable Setting
Default Value: NAN
Default Enabled:
Description:
Specifies the lower bound for the variable's value. Korali will not generate 
samples for which this variable falls below the specified minimum. By default, 
Korali sets this value to NAN.
******************************************************************************/
double lowerBound;

/******************************************************************************
Setting Name: Upper Bound
Type: Variable Setting
Default Value: NAN
Default Enabled:
Description:
Specifies the upper bound for the variable's value. Korali will not generate 
samples for which this variable falls below the specified maximum. By default, 
Korali sets this value to NAN.
******************************************************************************/
double upperBound;

/******************************************************************************
Setting Name: Initial Mean
Type: Variable Setting
Default Value: 0.5*(_variableSettings[i].upperBound + _variableSettings[i].lowerBound)
Default Enabled:
Description:
Defines the initial mean for the proposal distribution. This value must be 
defined between the variable's Mininum and Maximum settings. By default, Korali
sets this value in the center of the domain.
******************************************************************************/
double initialMean;

/******************************************************************************
Setting Name: Initial Standard Deviation
Type: Variable Setting
Default Value: 0.3*(_variableSettings[i].upperBound - _variableSettings[i].lowerBound)
Default Enabled:
Description:
Defines the initial standard deviation of the proposal distribution for a 
variable. By default, Korali sets this value to 30% of the domain width.
******************************************************************************/
double initialStdDev;

/******************************************************************************
Setting Name: Minimum Standard Deviation Changes
Type: Variable Setting
Default Value: 0.0
Default Enabled:
Description:
Defines a lower bound for the standard deviation updates of the proposal
distribution for a variable. Korali increases the scaling factor $\sigma$ if 
this value is undershot. By default, Korali sets this value to 0.0 (inactive).
******************************************************************************/
double minStdDevChange;

/******************************************************************************
Setting Name: Discrete
Type: Variable Setting
Default Value: false
Default Enabled:
Description:
Specifies if the variable can only take discrete values within the range of the 
lower and upper bound.
******************************************************************************/
bool discrete;

/******************************************************************************
Setting Name: Granularity
Type: Variable Setting
Default Value: 0.0
Default Enabled:
Description:
Specifies the granulatiry of a discrete variable, a granularity of 1.0
means that the variable can only take values in (.., -1.0, 0.0, +1.0, +2.0, ..)
where the levels are set symmetric around the initial mean (here 0.0).
******************************************************************************/
double granularity;

};

std::vector<variableSetting> _variableSettings;
/******************************************************************************/

 std::shared_ptr<Variable> _gaussianGenerator;
 std::shared_ptr<Variable> _uniformGenerator;

 CCMAES();
 ~CCMAES();

 void sampleSingle(size_t sampleIdx); /* sample individual */
 void evaluateSamples(); /* evaluate all samples until done */
 void adaptC(int hsig); /* CCMAES covariance matrix adaption */
 void updateSigma(); /* update Sigma */
 void updateEigensystem(std::vector<double>& M);
 void numericalErrorTreatment();
 void eigen(size_t N, std::vector<double>& C, std::vector<double>& diag, std::vector<double>& Q) const;
 void sort_index(const std::vector<double>& vec, std::vector<size_t>& _sortingIndex, size_t n) const;
 bool isFeasible(size_t sampleIdx) const; /* check if sample inside lower & upper bounds */

 // Runtime Methods (to be inherited from base class in the future)
 void prepareGeneration();
 bool checkTermination() override;
 void updateDistribution();
 void initialize() override;
 void runGeneration() override;
 void processSample(size_t sampleId, double fitness) override;

 // Private CCCMAES-Specific Methods
 void initMuWeights(size_t numsamples); /* init _muWeights and dependencies */
 void initCovariance(); /* init sigma, C and B */
 void checkMeanAndSetRegime(); /* check if mean inside valid domain, if yes, update internal vars */
 void updateConstraints();
 void updateViabilityBoundaries(); /* update & shrink viability boundaries */
 void handleConstraints(); /* covariance adaption for invalid samples */
 void reEvaluateConstraints(); /* re evaluate constraints, in handleConstraints,  count violations etc.. */

 void updateDiscreteMutationMatrix(); /* for integer optimization */
 
 void finalize() override;
 void setConfiguration() override;
 void getConfiguration() override;
 void printGeneration() override;
};

} // namespace Korali::Solver

} // namespace Korali

#endif // _KORALI_SOLVERS_CCMAES_HPP_
