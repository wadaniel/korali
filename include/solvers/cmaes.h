#ifndef _KORALI_SOLVERS_CMAES_H_
#define _KORALI_SOLVERS_CMAES_H_

#include "solvers/base.h"
#include <chrono>
#include <vector>
#include <map>

namespace Korali { namespace Solver {

/******************************************************************************
Module Name: Covariance Matrix Adaptation Evolution Strategy
Type: Solver, Optimizer
Alias: CMAES
Description:
### Base CMAES

This is the implementation of the *Covariance Matrix Adaptation Evolution Strategy*, as published in [Hansen2006](https://doi.org/10.1007/3-540-32494-1_4).

In an evolution strategy, new candidate solutions are sampled according to a multivariate normal distribution in $\mathbb {R} ^{n}$. Recombination amounts to selecting a new mean value for the distribution. Mutation amounts to adding a random vector, a perturbation with zero mean. Pairwise dependencies between the variables in the distribution are represented by a covariance matrix. The covariance matrix adaptation (CMA) is a method to update the covariance matrix of this distribution.

CMAES works iteratively, evaluating a number $\lambda$ of samples per generation, and improving the covariance matrix for the samples in the next generation.


** Usage **

```python
  k["Solver"] = "CMAES"
```

**Base Requirements:**

+ The *Sample Count* $\lambda$ needs to be defined.
+ The *Initial Mean* needs to be defined for every variable.
+ The *Initial Standard Deviation* needs to be defined for every variable.

******************************************************************************
Plotting:
Here we explain the **CMAES** result plot in further detail and how it can be
used to validate your optimization.

The module korali.plotter (run with `python3 -m korali.plotter`) command visualizes some of the most meaningful states of CMAES
stored in the result files in the output directory (`_korali_result`).
To plot a running simulation use the command `python3 -m korali.plotter --live` for incremental plots.

In the figure below we see the evolution of the CMAES algorithm during 100
optimization steps, respectively 1000 function evaluations (here the sample size
is 10), of the negative 2-dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) function.

![figure](optimizing_rosenbrock.png)

**Quadrant 1**: The first quadrant (upper left) shows 4 graphs plus markers (crosses):

* $| F |$ (red): $F$ is the best function evaluation of the current generation.
      Note that the colour of $F$ changes if $F < 0$ (red) or $F \geq 0$ (blue).
      Also, the absolute value of F is plotted since the vertical axis is given
      in log-scale.

* $\kappa(\mathbf{C})$ (cyan): This line shows the condition of the covariance matrix
          of the proposal distribution at every generation. The condition is given
          by the ratio of the largest Eigenvalue to the smallest Eigenvalue. A
          large condition number may lead to numerical instabilities, this can be
          treated by normalizing the domain of the objective variables.

* $|| \mathbf{p}_{\sigma} ||$ (black):  The evolution path is a measure of the travel
      direction of the mean of the proposal distribution of CMAES. The
      Euclidean norm of the evolution path plays an important rule in the Sigma
      updating rule.

* $\sigma$ (yellow) : $\sigma$ is the scaling parameter of the covariance matrix.
      The scaling parameter is updated at every generation. If Sigma becomes
          very large or small it may have an adverse effect on the optimization.

* $| F - F_{best}|$ (crosses) : At every generation we calculate the absolute
      difference betwen the current best function evaluation ($F$) and the overall
      best found evaluation ($F_{best}$) of CMAES. The crosses appear only if the
      current generation does not impeove the overall result, i.e. $F < F_{best}$
      in current generation and $| F - F_{best} |$ is greater 0.

A good indicator of convergance of CMAES to the global maximum is given by a steady decrease of $| F - F_{best} |$.

**Quadrant 2**: Objective Variables: This plot shows the evolution of the objective variables corresponding to the
evaluation of $|F|$. A line per each dimension of the optimization problem is plotted.

**Quadrant 3**: Square Root of Eigenvalues $\mathbf{C}$: The square root of the Eigenvalues of $\mathbf{C}$ are the lengths of the axes of the (unscaled) covariance matrix. Optimally the lengths of the axes are of same magnitude.

**Quadrant 4**: $\sigma \sqrt{diag(\mathbf{C})}$: the square root of the diagonal elements of the (scaled) covariance matrix of
the proposal distribution approximate the standard deviation of the parameters. Ideally the standard deviations of all coordinates are of same magnitude.

**Example: Shekel function**

The following figure shows the results of an unsuccessful maximization of the
negative of the [Shekel](https://en.wikipedia.org/wiki/Shekel_function) function in 4
dimensions and with 10 local maxima.

[We know](http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2354.htm)
that the Shekel function has a global minimum at (4, 4, 4, 4),
respectivel maximum in the negative case.
In quadrant 2 we see that CMAES converged to a different result.

In general the global optimum is not known, following observations indicate
ill convergence. Restarting CMAES from different starting points as well as
tuning CMAES internal parameters might improve optimization:

* Increasing condition (quadrant 1) of the covariance matrix of the proposal
  distribution, respectively diverging axes lenghts and standard deviations
  (quadrants 3 & 4).

* None decreasing values for $| F - F_{best} |$. Arguably CMAES found a better
  function evaluation on a different hill but the algorithm is trapped (the
  objective variables stabilized sampling does not overcome the saddle points).

![figure](optimizing_shekel.png)
******************************************************************************/

class CMAES : public Base
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
bool _isDiag;

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
size_t _maxCovMatrixCorrections;
 
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
double _covMatrixAdaptionStrength;

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
Setting Name: Chi Number
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
Sorting _sortingIndex of current sample pop (_sortingIndex[0] idx of current best).
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
std::vector<double> _axisD;

/******************************************************************************
Setting Name: Temporary Axis Lengths
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Temporary storage for Axis lengths
******************************************************************************/
std::vector<double> _axisDtmp;

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
std::vector<double> _y;

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


/******************************************************************************
* Variable Settings
******************************************************************************/

struct variableSetting
{

/******************************************************************************
Setting Name: Lower Bound
Type: Variable Setting
Default Value: -std::numeric_limits<float>::max()
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
Default Value: +std::numeric_limits<float>::max()
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
Default Value: (_variableSettings[i].upperBound + _variableSettings[i].lowerBound)*0.5
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
Default Value: (_variableSettings[i].upperBound - _variableSettings[i].lowerBound)*0.25
Default Enabled:
Description:
Defines the initial standard deviation of the proposal distribution for a variable.
By default, Korali sets this value to 30% of the domain width.
******************************************************************************/
double initialStdDev;

/******************************************************************************
Setting Name: Minimum Standard Deviation Changes
Type: Variable Setting
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

 // Workspace for GSL
 Variable* _gaussianGenerator;

 CMAES();
 ~CMAES();

 void sampleSingle(size_t sampleIdx); /* sample individual */
 void evaluateSamples(); /* evaluate all samples until done */
 void adaptC(int hsig); /* CMAES covariance matrix adaption */
 void updateEigensystem(std::vector<double>& M, int flgforce = 1);
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

 void initMuWeights(size_t numsamples); /* init _muWeights and dependencies */
 void initCovariance(); /* init sigma, C and B */
 void checkMeanAndSetRegime(); /* check if mean inside valid domain, if yes, update internal vars */

 void finalize() override;
 void setConfiguration() override;
 void getConfiguration() override;
 void printGeneration() override;
};

} // namespace Korali::Solver

} // namespace Korali

#endif // _KORALI_SOLVERS_CMAES_H_
