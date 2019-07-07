#ifndef _KORALI_SOLVERS_CMAES_H_
#define _KORALI_SOLVERS_CMAES_H_

#include "solvers/base.h"
#include <chrono>
#include <vector>
#include <map>

namespace Korali { namespace Solver {

/******************************************************************************
Module Name: CMAES
Type: Solver, Optimizer
Alias: CMAES
Description:
### Base CMA-ES

This is the implementation of the *Covariance Matrix Adaptation Evolution Strategy*, as published in [Hansen2006](https://doi.org/10.1007/3-540-32494-1_4).

In an evolution strategy, new candidate solutions are sampled according to a multivariate normal distribution in $\mathbb {R} ^{n}$. Recombination amounts to selecting a new mean value for the distribution. Mutation amounts to adding a random vector, a perturbation with zero mean. Pairwise dependencies between the variables in the distribution are represented by a covariance matrix. The covariance matrix adaptation (CMA) is a method to update the covariance matrix of this distribution.

CMA-ES works iteratively, evaluating a number $\lambda$ of samples per generation, and improving the covariance matrix for the samples in the next generation.

**Base Requirements:**

+ The *Sample Count* $\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.

### Constrained CMA-ES

This solver also implements the *Constrained Covariance Matrix Adaptation Evolution Strategy*, as published in [Arampatzis2019](https://dl.acm.org/citation.cfm?doid=3324989.3325725).

CCMA-ES is an extension of [CMA-ES](/usage/solvers/optimizers/cmaes/) for constrained optimization problems. It uses the principle of *viability boundaries* to find an initial mean vector for the proposal distribution that does not violate constraints, and secondly it uses a  *constraint handling technique* to efficiently adapt the proposal distribution to the constraints.

** Constraint Requirements:**

+ At least one *Constraint function* defined.
+ The *Viability Sample Count* $via-\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.
******************************************************************************/

class CMAES : public Base
{
 public:

/******************************************************************************
Setting Name: Objective
Type: Solver Setting
Format: String
Mandatory: No
Default Value: "Maximize"
Default Enabled:
Description:
Specifies whether the problem evaluation is to be minimized or maximized.
******************************************************************************/
std::string _objective;

/******************************************************************************
Setting Name: Sample Count
Type: Solver Setting
Format: Integer
Mandatory: No
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
Format: Integer
Mandatory: No
Default Value: 16
Default Enabled:
Description:
Number of best samples used to update the covariance matrix and the mean.
******************************************************************************/
size_t _muValue;

/******************************************************************************
Setting Name: Mu Type
Type: Solver Setting
Format: String
Mandatory: No
Default Value: "Logarithmic"
Default Enabled:
Description:
Weights given to the Mu best values to update the covariance matrix and the mean.
******************************************************************************/
std::string _muType;

/******************************************************************************
Setting Name: Initial Sigma Cumulation Factor
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 1.0
Default Enabled:
Description:
Controls the learning rate of the conjugate evolution path.
******************************************************************************/
double _initialSigmaCumulationFactor;

/******************************************************************************
Setting Name: Initial Damp Factor
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 1.0
Default Enabled:
Description:
Controls the updates of the covariance matrix scaling factor.
******************************************************************************/
double _initialDampFactor;

/******************************************************************************
Setting Name: Is Sigma Bounded
Type: Solver Setting
Format: Boolean
Mandatory: No
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
Format: Real
Mandatory: No
Default Value: 1.0
Default Enabled:
Description:
Controls the learning rate of the evolution path for the covariance update
(must be in (0,1]).
******************************************************************************/
double _initialCumulativeCovariance;

//Unused
//double _initialCovMatrixLearningRate;

/******************************************************************************
Setting Name: Is Diagonal
Type: Solver Setting
Format: Boolean
Mandatory: No
Default Value: false
Default Enabled:
Description:
Covariance matrix updates will be optimized for diagonal matrices.
******************************************************************************/
bool _isDiag;

/******************************************************************************
Setting Name: Use Viability Regime
Type: Solver Setting
Format: Boolean
Mandatory: No
Default Value: false
Default Enabled:
Description:
Determines if a viability regime should be used.
******************************************************************************/
bool _isViabilityRegime;


/******************************************************************************
Setting Name: Viability Sample Count
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 8
Default Enabled:
Description:
Specifies the number of samples per generation during the viability 
regime, i.e. during the search for a mean vector not violating the constraints.
******************************************************************************/
size_t _viabilitySampleCount;

/******************************************************************************
Setting Name: Viability Mu
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 4
Default Enabled:
Description:
Number of best samples used to update the covariance matrix and the mean 
during the viability regime.
******************************************************************************/
size_t _viabilityMu;

/******************************************************************************
Setting Name: Max Covariance Matrix Corrections
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 1e6
Default Enabled:
Description:
Max number of covairance matrix adaptions per generation during the constraint 
handling loop.
******************************************************************************/
size_t _maxCovMatrixCorrections;
 
/******************************************************************************
Setting Name: Target Success Rate
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.1818
Default Enabled:
Description:
Controls the updates of the covariance matrix scaling factor during the
viability regime.
******************************************************************************/
double _targetSuccessRate;

/******************************************************************************
Setting Name: Covariance Matrix Adaption Strenth
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.1
Default Enabled:
Description:
Controls the covariane matrix adaption strength if samples violate constraints.
******************************************************************************/
double _covMatrixAdaptionStrength;

/******************************************************************************
Setting Name: Normal Vector Learning Rate
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.5
Default Enabled:
Description:
Learning rate of constraint normal vectors (must be in (0, 1]).
******************************************************************************/
double _normalVectorLearningRate;

/******************************************************************************
Setting Name: Global Success Learning Rate
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.2
Default Enabled:
Description:
Learning rate of success probability of objective function improvements. 
Required for covariance matrix scaling factor update during viability regime.
******************************************************************************/
double _globalSuccessLearningRate;
 
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
Setting Name: Max Infeasible Resampling
Type: Termination Criterion
Format: Integer
Mandatory: No
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
Format: Real
Mandatory: No
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
Format: Real
Mandatory: No
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
Format: Real
Mandatory: No
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
Format: Real
Mandatory: No
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
Format: Real
Mandatory: No
Default Value: 1e-18
Default Enabled: false
Description:
Specifies the maximal standard deviation per dimension of the proposal.
******************************************************************************/
double _termCondMaxStandardDeviation;
bool   _termCondMaxStandardDeviationEnabled;

/******************************************************************************
Setting Name: Max Condition Covariance Matrix
Type: Termination Criterion
Format: Real
Mandatory: No
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
Format: Real
Mandatory: No
Default Value: 1e18
Default Enabled: false
Description:
Specifies a scaling factor under which the standard deviation does not change
in the direction of the eigenvectors.
******************************************************************************/
double _termCondMinStandardDeviationStepFactor;
bool   _termCondMinStandardDeviationStepFactorEnabled;


/******************************************************************************
* Variable Settings
******************************************************************************/

struct variableSetting
{

/******************************************************************************
Setting Name: Lower Bound
Type: Variable Setting
Format: Real
Mandatory: No
Default Value: -INFINITY
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
Format: Real
Mandatory: No
Default Value: +INFINITY
Default Enabled:
Description:
Specifies the upper bound for the variable's value. Korali will not generate samples
for which this variable falls below the specified maximum. By default, Korali sets this
value to +Infinity.
******************************************************************************/
double upperBound;

/******************************************************************************
Setting Name: Initial Mean
Type: Variable Setting
Format: Real
Mandatory: No
Default Value: 0.0
Default Enabled:
Description:
Defines the initial mean for the proposal distribution. This value must be defined
between the variable's Mininum and Maximum settings By default, Korali sets this
value in the center of the domain.
******************************************************************************/
double initialMean;

/******************************************************************************
Setting Name: Initial Standard Deviation
Type: Variable Setting
Format: Real
Mandatory: No
Default Value: 1.0
Default Enabled:
Description:
Defines the initial standard deviation of the proposal distribution for a variable.
By default, Korali sets this value to 30% of the domain width.
******************************************************************************/
double initialStdDev;

/******************************************************************************
Setting Name: Minimum Stardard Deviation Changes
Type: Variable Setting
Format: Real
Mandatory: No
Default Value: 0.0
Default Enabled:
Description:
Defines a lower bound for the standard deviation updates of the proposal distribution for a variable.
Korali increases the scaling factor $\sigma$ if this value is undershot. By default,
Korali sets this value to 0.0 (inactive).
******************************************************************************/
double minStdDevChange;

};

std::vector<variableSetting> _variableSettings;
/******************************************************************************/

// Ctor & Dtor
CMAES();

// Runtime Methods (to be inherited from base class in the future)
void prepareGeneration();
bool checkTermination() override;
void updateDistribution();
void initialize() override;
void runGeneration() override;
void processSample(size_t sampleId, double fitness) override;

private:
 // Korali Runtime Variables
 int _fitnessSign; /* maximizing vs optimizing (+- 1) */
 std::vector<double> _fitnessVector; /* objective function values [_s] */
 std::vector<double> _samplePopulation; /* sample coordinates [_s x _k->N] */
 std::vector<bool> _initializedSample; /* flag to distribute work */
 std::vector<double> _transformedSamples;

 size_t _finishedSamples; /* counter of evaluated samples to terminate evaluation */
 size_t _current_s; /* number of samples active ( _s or _via_s ) */
 size_t _current_mu; /* number of samples active ( _mu or _mu_s ) */
 std::vector<double> _muWeights; /* weights for mu best samples */
 double _muEffective; /* variance effective selection mass */
 //double _muCovarianceIn; /* read from configuration, placeholder for reinit */
 //double _muCovariance; /* internal parameter to calibrate updates */

 double _sigmaCumulationFactor; /* increment for sigma, default calculated from muEffective and dimension */
 double _dampFactor; /* dampening parameter determines controls step size adaption */
 double _cumulativeCovariance; /* default calculated from dimension */
 double _covarianceMatrixLearningRate; /* parameter to calibrate cov updates */
 double _chiN; /* expectation of ||N(0,I)||^2 */
 size_t _covarianceEigenEvalFreq;

 // Stop conditions
 // Private CMAES-Specific Variables
 double sigma;  /* step size */
 double _trace; /* to init sigma (or set upper bound) */
 Variable* _gaussianGenerator;

 double bestEver; /* best ever fitness */
 double prevBest; /* best ever fitness from previous generation */
 std::vector<double> rgxmean; /* mean "parent" */
 std::vector<double> rgxbestever; /* bestever vector */
 std::vector<double> curBestVector; /* current best vector */
 std::vector<size_t> index; /* sorting index of current sample pop (index[0] idx of current best). */
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */

 std::vector<double> C; /* Covariance Matrix */
 std::vector<double> Ctmp; /* tmp Covariance Matrix for eigen decomp for safety check */
 std::vector<double> B; /* matrix with eigenvectors in columns */
 std::vector<double> Btmp; /* matrix for eigenvectors calculation for safety check*/
 std::vector<double> axisD; /* axis lengths (sqrt(Evals)) */
 std::vector<double> axisDtmp; /* for axis lengths calculation for saftey check */
 
 std::vector<double> Z; /* randn() */
 std::vector<double> BDZ; /* B*D*randn() */

 std::vector<double> rgpc; /* evolution path for cov update */
 std::vector<double> rgps; /* conjugate evolution path for sigma update */
 std::vector<double> rgxold; /* mean "parent" previous generation */
 std::vector<double> rgBDz; /* for B*D*z */
 std::vector<double> rgdTmp; /* temporary (random) vector used in different places */

 size_t countevals; /* Number of function evaluations */
 size_t countinfeasible; /* Number of samples outside of domain given by bounds */
 double maxdiagC; /* max diagonal element of C */
 double mindiagC; /* min diagonal element of C */
 double maxEW; /* max Eigenwert of C */
 double minEW; /* min Eigenwert of C */
 double psL2; /* L2 norm of rgps */

 bool flgEigensysIsUptodate;

 // Private CMA-ES-Specific Methods
 void sampleSingle(size_t sampleIdx); /* sample individual */
 void evaluateSamples(); /* evaluate all samples until done */
 void adaptC(int hsig); /* CMA-ES covariance matrix adaption */
 void updateEigensystem(std::vector<double>& M, int flgforce = 1);
 void eigen(size_t N, std::vector<double>& C, std::vector<double>& diag, std::vector<double>& Q) const;
 void sort_index(const std::vector<double>& vec, std::vector<size_t>& index, size_t n) const;
 bool isFeasible(size_t sampleIdx) const; /* check if sample inside lower & upper bounds */

 // Private CCMA-ES-Specific Variables
 bool _hasConstraints; /* True if num constraints greater 0 */
 double _beta; /* Factor of covariance matrix adaption size */

 int bestValidIdx; /* best sample with wo constraint violation (otherwise -1) */
 double _globalSucRate; /* estim. global success rate */
 double fviability; /* viability func value */
 size_t resampled; /* number of resampled parameters due constraint violation */
 size_t correctionsC; /* number of cov matrix adaptions */
 size_t countcevals; /* number of constraint evaluations */
 std::vector<double> sucRates; /* constraint success rates */
 double *viabilityBounds; /* viability boundaries */
 bool *viabilityImprovement; /* sample evaluations larger than fviability */ //TODO: not neeeded?
 size_t maxnumviolations; /* maximal amount of constraint violations */
 size_t *numviolations; /* number of constraint violations for each sample */
 bool **viabilityIndicator; /* constraint evaluation better than viability bound */
 double **constraintEvaluations; /* evaluation of each constraint for each sample  */
 double **v; /* normal approximation of constraints */
 double *besteverCeval; /* constraint evaluations for best ever */

 // Workspace for gsl
 gsl_vector* gsl_eval;
 gsl_matrix* gsl_evec;
 gsl_eigen_symmv_workspace* gsl_work;

 // Private CCMA-ES-Specific Methods
 void initMuWeights(size_t numsamples); /* init _muWeights and dependencies */
 void initCovariance(); /* init sigma, C and B */
 void initCovCorrectionParams(); /* init beta and cv */
 void checkMeanAndSetRegime(); /* check if mean inside valid domain, if yes, update internal vars */
 void updateConstraints();
 void updateViabilityBoundaries(); /* update & shrink viability boundaries */
 void handleConstraints(); /* covariance adaption for invalid samples */
 void reEvaluateConstraints(); /* re evaluate constraints, in handleConstraints,  count violations etc.. */

 void finalize() override;
 void setConfiguration() override;
 void getConfiguration() override;
 void printGeneration() override;
};

} // namespace Korali::Solver

} // namespace Korali

#endif // _KORALI_SOLVERS_CMAES_H_
