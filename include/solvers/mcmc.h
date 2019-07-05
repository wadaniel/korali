#ifndef _KORALI_SOLVER_MCMC_H_
#define _KORALI_SOLVER_MCMC_H_

#include "solvers/base.h"
#include "variable.h"
#include <gsl/gsl_rng.h>

namespace Korali { namespace Solver {

/******************************************************************************
Module Name: Markov Chain Monte Carlo
Type: Solver, Sampler
Alias: MCMC
Description:
This is an implementation of the *Delayed Rejection Adaptive Metropolis* algorithm,
as published in [Haario2006](https://link.springer.com/article/10.1007%2Fs11222-006-9438-0).
This solver can also be configured to run the standard *Metropolis Hastings* method.

**Requirements:**

+ The *Chain Length*, i.e. the number of generated samples.
+ The *Initial Mean* of the proposal distribution for each variable.
+ The *Standard Deviation* of the proposal distribution for each variable.
******************************************************************************/

class MCMC : public Base
{
 public:

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

/******************************************************************************
Setting Name: Chain Length
Type: Solver Setting
Format: Integer
Mandatory: Yes
Default Value:
Default Enabled:
Description:
Specifies the output frequency of intermediate result files.
******************************************************************************/
size_t maxChainLength;

/******************************************************************************
Setting Name: Burn In
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 0
Default Enabled:
Description:
Specifies the number of preliminary MCMC steps before samples are being drawn.
This may reduce effects from improper initialization.
******************************************************************************/
size_t burnIn;

/******************************************************************************
Setting Name: Rejection Levels
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 1
Default Enabled:
Description:
Controls the number of accept-reject stages per MCMC step.
******************************************************************************/
size_t rejectionLevels;

/******************************************************************************
Setting Name: Use Adaptive Sampling
Type: Solver Setting
Format: Boolean
Mandatory: No
Default Value: false
Default Enabled:
Description:
Specifies if covariance matrix of the proposal distribution is calculated from
the samples.

A "false" value corresponds to the standard Metropolis Hastings.
******************************************************************************/
bool useAdaptiveSampling;

/******************************************************************************
Setting Name: Non Adaption Period
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 0
Default Enabled:
Description:
Number of steps (after Burn In steps) during which the initial standard deviation
is used (only relevant for Adaptive Sampling). If "0" is specified, Korali will
set this value to $5%$ of the Chain Length.
******************************************************************************/
size_t nonAdaptionPeriod;

/******************************************************************************
Setting Name: Chain Covariance Scaling
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.0
Default Enabled:
Description:
Scaling factor of the chain covariance (only relevant for Adaptive Sampling).
If "0.0" is specified, Korali will precalibrate this factor from the number of
variables: $\frac{2.4^2}{N}$ [Gelman1995].
******************************************************************************/
double chainCovarianceScaling;

/******************************************************************************
Setting Name: Chain Covariance Increment
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.001
Default Enabled:
Description:
Small constant to avoid singularity of the chain covariance.
******************************************************************************/
double chainCovarianceIncrement;

/******************************************************************************
Setting Name: Initial Mean
Type: Variable Setting
Format: Real
Mandatory: Yes
Default Value:
Default Enabled:
Description:
Specifies the Initial Mean of the proposal distribution.
******************************************************************************/
std::vector<double> variableInitialMeans;

/******************************************************************************
Setting Name: Standard Deviation
Type: Variable Setting
Format: Real
Mandatory: Yes
Default Value:
Default Enabled:
Description:
Specifies the Standard Deviation for each variable. The proposal distribution
is defined through a covariance matrix with the variance of the variables in
its diagonal.
******************************************************************************/
std::vector<double> variableStandardDeviations;

/******************************************************************************
Setting Name: Log Space
Type: Variable Setting
Format: Boolean
Mandatory: No
Default Value: false
Default Enabled:
Description:
Indicates whether the variable is expressed in Logarithmic Space.
******************************************************************************/
std::vector<bool> variableLogSpaces;

/******************************************************************************
Setting Name: Max Function Evaluations
Type: Termination Criterion
Format: Integer
Mandatory: No
Default Value: 100000
Default Enabled: true
Description:
Specifies the maximum number of function evaluations (only relevant if
Rejection Levels are large).
******************************************************************************/
size_t maxFunctionEvaluations;
bool maxFunctionEvaluationsEnabled;

 // MCMC Configuration
 //size_t _maxresamplings; /* Max number resamplings inside generation loop */
 char _terminationReason[500];

 Korali::Variable* _gaussianGenerator; /* Gaussian random number generator */
 Korali::Variable* _uniformGenerator; /* Uniform random number generator */

 // MCMC Runtime Variables
 double* z; /* Placeholder random numbers */
 double* clPoint; /* Leader parameter values */
 double clLogLikelihood; /* Leader fitness value */
 double* ccPoints; /*  Candidates parameter values */
 double* transformedSamples; /* Candidate parameters log transformed */
 double* ccLogPriors; /* Candidates prior value */
 double* ccLogLikelihoods; /* Candidates fitness value */
 double* alpha; /* alphas for recursive calculation of delayed rejection schemes */
 double acceptanceRateProposals; /* Ratio proposed to accepted Samples */
 size_t rejections; /* Rejections in current generation*/
 size_t naccept; /* Number of accepted samples */
 size_t countgens; /* Number of proposed samples */
 size_t databaseEntries; /* Accepted Samples */
 double* databasePoints; /* Variable values of samples in DB */
 double* databaseFitness; /* Fitness of samples in DB */
 size_t countevals; /* Number of function evaluations */
 double* chainMean; /* Mean of mcmc chain */
 double* tmpC; /* Placeholder chain cov calculation */
 double* chainCov; /* Variance of mcmc chain */
 size_t chainLength;

 // MCMC Status variables
 double* _covarianceMatrix; /* Covariance of Proposal Distribution */

 // Korali Methods
 void initialize() override;
 void finalize() override;

 void runGeneration() override;
 void processSample(size_t c, double fitness) override;

  // Internal MCMC Methods
 void updateDatabase(double* point, double fitness);
 void generateCandidate(size_t sampleIdx);
 void sampleCandidate(size_t sampleIdx);
 void acceptReject(size_t trial); /* Accept or reject sample with multiple trials */
 double recursiveAlpha(double& D, const double llk0, const double* logliks, size_t N) const; /* calculate acceptance ratio alpha_N */
 void evaluateSample();
 void updateState();
 bool setCandidatePriorAndCheck(size_t sampleIdx);
 bool checkTermination() override;

 void setConfiguration() override;
 void getConfiguration() override;
 void printGeneration() override;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_MCMC_H_
