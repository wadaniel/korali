#ifndef _KORALI_SOLVERS_CMAES_H_
#define _KORALI_SOLVERS_CMAES_H_

#include "solvers/base.h"
#include <chrono>
#include <map>

namespace Korali { namespace Solver {

/******************************************************************************
Module Name: CMAES
Type: Solver, Optimizer
Alias: CMAES
Description:
******************************************************************************/

class CMAES : public Base
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
std::string _objective; /* Maximize or Minimize */ 

/******************************************************************************
Setting Name: Sample Count
Type: Solver Setting
Format: Integer
Mandatory: Yes
Default Value:
Default Enabled:
Description:
Specifies the number of samples to evaluate per generation (preferably 
4+3*log(N) number of variables).
******************************************************************************/
size_t _s;

/******************************************************************************
Setting Name: Mu Value
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 0.5*(Sample Count)
Default Enabled:
Description:
Number of best samples used to update the covariance matrix and the mean.
******************************************************************************/
size_t _mu;

/******************************************************************************
Setting Name: Mu Type
Type: Solver Setting
Format: String
Mandatory: No
Default Value: Logarithmic
Default Enabled:
Description:
Weights given to the Mu best values to update the covariance matrix and the mean.
******************************************************************************/
std::string _muType;

/******************************************************************************
Setting Name: Sigma Cumulation Factor
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: (calibrated internally)
Default Enabled:
Description:
Controls the learning rate of the conjugate evolution path.
******************************************************************************/
double _sigmaCumulationFactorIn;

/******************************************************************************
Setting Name: Damp Factor
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: (calibrated internally)
Default Enabled:
Description:
Controls the updates of the covariance matrix scaling factor.
******************************************************************************/
double _dampFactorIn;

/******************************************************************************
Setting Name: Is Sigma Bounded
Type: Solver Setting
Format: Boolean
Mandatory: No
Default Value: False
Default Enabled:
Description:
Sets an upper bound for the covariance matrix scaling factor. The upper bound 
is given by the average of the initial standard deviation of the variables
******************************************************************************/
bool _isSigmaBounded;

/******************************************************************************
Setting Name: Cumulative Covariance
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: (calibrated internally)
Default Enabled:
Description:
Controls the learning rate of the evolution path for the covariance update
(must be in (0,1]).
******************************************************************************/
double _cumulativeCovarianceIn;

/******************************************************************************
Setting Name: Covariance Matrix Learning Rate
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: (calibrated internally)
Default Enabled:
Description:
Controls the learning rate of the covariance matrix (must be in (0,1]).
******************************************************************************/
double _covMatrixLearningRateIn;

/******************************************************************************
Setting Name: Is Diagonal
Type: Solver Setting
Format: Boolean
Mandatory: No
Default Value: False
Default Enabled:
Description:
Covariance matrix updates will be optimized for diagonal matrices.
******************************************************************************/
bool _isdiag;

/******************************************************************************
Setting Name: Viability Sample Count
Type: Solver Setting
Format: Integer
Mandatory: Yes
Default Value:
Default Enabled:
Description:
Specifies the number of samples per generation during the viability 
regime, i.e. during the search for a mean vector not violating the constraints.
******************************************************************************/
size_t _via_s;

/******************************************************************************
Setting Name: Viability Mu
Type: Solver Setting
Format: Integer
Mandatory: Yes
Default Value:
Default Enabled:
Description:
Number of best samples used to update the covariance matrix and the mean 
during the viability regime.
******************************************************************************/
size_t _via_mu;

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
size_t _maxCorrections;
 
/******************************************************************************
Setting Name: Max Resampling
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 1e9
Default Enabled:
Description:
Number of resamplings per candidate per generation if sample is outside of 
Lower and Upper Bound
******************************************************************************/
size_t _maxResamplings;

/******************************************************************************
Setting Name: Target Success Rate
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 2/11
Default Enabled:
Description:
Controls the updates of the covariance matrix scaling factor during the
viability regime.
******************************************************************************/
double _targetSucRate;

/******************************************************************************
Setting Name: Covariance Matrix Adaption Size
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.1
Default Enabled:
Description:
Controls the covariane matrix adaption strength if samples violate constraints.
******************************************************************************/
double _adaptionSize;

/******************************************************************************
Setting Name: Normal Vector Learning Rate
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: (internally calibrated)
Default Enabled:
Description:
Learning rate of constraint normal vectors (must be in (0, 1]).
******************************************************************************/
double _cv;

/******************************************************************************
Setting Name: Global Success Learning Rate
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: (internally calibrated)
Default Enabled:
Description:
Learning rate of success probability of objective function improvements. 
Required for covariance matrix scaling factor update during viability regime.
******************************************************************************/
double _cp;
 
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
size_t resultOutputFrequency;

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
size_t terminalOutputFrequency;

// These are CMA-ES Specific, but could be used for other methods in the future
double* _lowerBounds;
double* _upperBounds;
double* _initialMeans;
double* _initialStdDevs;
double* _minStdDevChanges;
bool* _variableLogSpace; /* Apply log transform of variable before evaluation */

// Runtime Methods (to be inherited from base class in the future)
void prepareGeneration();
bool checkTermination() override;
void updateDistribution(const double *fitnessVector);
void initialize() override;
void run() override;
void processSample(size_t sampleId, double fitness) override;

private:

 // Korali Runtime Variables
 int _fitnessSign; /* maximizing vs optimizing (+- 1) */
 double* _fitnessVector; /* objective function values [_s] */
 double* _samplePopulation; /* sample coordinates [_s x _k->N] */
 size_t _currentGeneration; /* generation count */
 bool* _initializedSample; /* flag to distribute work */
 char _terminationReason[500]; /* buffer for exit reason */
 double*_transformedSamples;

 size_t _finishedSamples; /* counter of evaluated samples to terminate evaluation */
 size_t _current_s; /* number of samples active ( _s or _via_s ) */
 size_t _current_mu; /* number of samples active ( _mu or _mu_s ) */
 double* _muWeights; /* weights for mu best samples */
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
 size_t _termCondMaxGenerations; // Max number of generations.
 size_t _termCondMaxFitnessEvaluations;   // Defines maximum number of fitness evaluations
 double _termCondMinFitness; // Defines the minimum fitness allowed, otherwise it stops
 double _termCondMaxFitness; // Defines the maximum fitness allowed, otherwise it stops
 double _termCondFitnessDiffThreshold; // Defines minimum function value differences before stopping
 double _termCondMinDeltaX; // Defines minimum delta of input parameters among generations before it stops.
 double _termCondTolUpXFactor; // Defines the minimum fitness allowed, otherwise it stops
 double _termCondCovCond; // Defines the maximal condition number of the covariance matrix
 double _termCondMinStepFac; // Factor to calculate min step length
 bool _isTermCondMaxGenerations, _isTermCondMaxFitnessEvaluations, _isTermCondMinFitness, 
      _isTermCondMaxFitness, _isTermCondFitnessDiffThreshold, _isTermCondMinDeltaX, 
      _isTermCondTolUpXFactor, _isTermCondCovCond, _isTermCondMinStepFac; // flgs to activate termination criteria

 // Private CMAES-Specific Variables
 double sigma;  /* step size */
 double _trace; /* to init sigma (or set upper bound) */
 Variable* _gaussianGenerator;

 double bestEver; /* best ever fitness */
 double prevBest; /* best ever fitness from previous generation */
 double *rgxmean; /* mean "parent" */
 double *rgxbestever; /* bestever vector */
 double *curBestVector; /* current best vector */
 size_t *index; /* sorting index of current sample pop (index[0] idx of current best). */
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */

 double **C; /* Covariance Matrix */
 double **Ctmp; /* tmp Covariance Matrix for eigen decomp for safety check */
 double **B; /* matrix with eigenvectors in columns */
 double **Btmp; /* matrix for eigenvectors calculation for safety check*/
 double *axisD; /* axis lengths (sqrt(Evals)) */
 double *axisDtmp; /* for axis lengths calculation for saftey check */
 
 double **Z; /* randn() */
 double **BDZ; /* B*D*randn() */

 double *rgpc; /* evolution path for cov update */
 double *rgps; /* conjugate evolution path for sigma update */
 double *rgxold; /* mean "parent" previous generation */
 double *rgBDz; /* for B*D*z */
 double *rgdTmp; /* temporary (random) vector used in different places */
 double *histFuncValues; /* holding historical best function values */

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
 void updateEigensystem(double **M, int flgforce = 1);
 void eigen(size_t N, double **C, double *diag, double **Q) const;
 size_t maxIdx(const double *rgd, size_t len) const;
 size_t minIdx(const double *rgd, size_t len) const;
 void sort_index(const double *rgFunVal, size_t *index, size_t n) const;
 bool isFeasible(size_t sampleIdx) const; /* check if sample inside lower & upper bounds */
 double doubleRangeMax(const double *rgd, size_t len) const;
 double doubleRangeMin(const double *rgd, size_t len) const;

 // Private CCMA-ES-Specific Variables
 bool _hasConstraints; /* True if num constraints greater 0 */
 double _beta; /* Factor of covariance matrix adaption size */

 bool _isViabilityRegime; /* true if mean violates constraints */
 int bestValidIdx; /* best sample with wo constraint violation (otherwise -1) */
 double _globalSucRate; /* estim. global success rate */
 double fviability; /* viability func value */
 size_t resampled; /* number of resampled parameters due constraint violation */
 size_t correctionsC; /* number of cov matrix adaptions */
 size_t countcevals; /* number of constraint evaluations */
 double *sucRates; /* constraint success rates */
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

 void setConfiguration() override;
 void getConfiguration() override;

 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} // namespace Korali::Solver

} // namespace Korali

#endif // _KORALI_SOLVERS_CMAES_H_
