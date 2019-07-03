#ifndef _KORALI_SOLVER_MCMC_H_
#define _KORALI_SOLVER_MCMC_H_

#include "solvers/base.h"
#include "variable.h"
#include <gsl/gsl_rng.h>

namespace Korali { namespace Solver {

/******************************************************************************
Solver Name: Markov Chain Monte Carlo
Type: Sampler
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
* Solver Settings
******************************************************************************/

/******************************************************************************
Name: Result Output Frequency
Type: Integer
Mandatory: No
Default Value: 1
Description:
Specifies the output frequency of intermediate result files.
******************************************************************************/
size_t resultOutputFrequency;

/******************************************************************************
Name: Terminal Output Frequency
Type: Integer
Mandatory: No
Default Value: 1
Description:
Specifies the output frequency onto the terminal screen.
******************************************************************************/
size_t terminalOutputFrequency;

/******************************************************************************
Name: Chain Length
Type: Integer
Mandatory: Yes
Default Value: ---
Description:
Specifies the output frequency of intermediate result files.
******************************************************************************/
size_t chainLength;

/******************************************************************************
Name: Burn In
Type: Integer
Mandatory: No
Default Value: 0
Description:
Specifies the number of preliminary MCMC steps before samples are being drawn.
This may reduce effects from improper initialization.
******************************************************************************/
size_t burnIn;

/******************************************************************************
* Solver Setting
*******************************************************************************
Name: Rejection Levels
Type: Integer
Mandatory: No
Default Value: 1
Description:
Controls the number of accept-reject stages per MCMC step.
******************************************************************************/
size_t rejectionLevels;

/******************************************************************************
Name: Use Adaptive Sampling
Type: Boolean
Mandatory: No
Default Value: False
Description:
Specifies if covariance matrix of the proposal distribution is calculated from
the samples.

A "False" value corresponds to the standard Metropolis Hastings.
******************************************************************************/
bool useAdaptiveSampling;

/******************************************************************************
Name: Non Adaption Period
Type: Integer
Mandatory: No
Default Value: 0
Description:
Number of steps (after Burn In steps) during which the initial standard deviation
is used (only relevant for Adaptive Sampling). If "0" is specified, Korali will
set this value to $5%$ of the Chain Length.
******************************************************************************/
size_t nonAdaptionPeriod;

/******************************************************************************
Name: Chain Covariance Scaling
Type: Real
Mandatory: No
Default Value: 0.0
Description:
Scaling factor of the chain covariance (only relevant for Adaptive Sampling).
If "0.0" is specified, Korali will precalibrate this factor from the number of
variables: $\frac{2.4^2}{N}$ [Gelman1995].
******************************************************************************/
double chainCovarianceScaling;

/******************************************************************************
Name: Chain Covariance Increment
Type: Real
Mandatory: No
Default Value: 0.001
Description:
Small constant to avoid singularity of the chain covariance.
******************************************************************************/
double chainCovarianceIncrement;

/******************************************************************************
* Variable Settings
******************************************************************************/

/******************************************************************************
Name: Initial Mean
Type: Real
Mandatory: Yes
Default Value: ---
Description:
Specifies the Initial Mean of the proposal distribution.
******************************************************************************/
std::vector<double> variableInitialMeans;

/******************************************************************************
Name: Standard Deviation
Type: Real
Mandatory: Yes
Default Value: ---
Description:
Specifies the Standard Deviation for each variable. The proposal distribution
is defined through a covariance matrix with the variance of the variables in
its diagonal.
******************************************************************************/
std::vector<double> variableStandardDeviations;

/******************************************************************************
Name: Log Space
Type: Boolean
Mandatory: No
Default Value: False
Description:
Indicates whether the variable is expressed in Logarithmic Space.
******************************************************************************/
std::vector<bool> variableLogSpaces;

/******************************************************************************
* Termination Criteria
******************************************************************************/

/******************************************************************************
Name: Max Function Evaluations
Type: Integer
Mandatory: No
Default Value: 100000
Default State: Enabled
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

 // MCMC Status variables
 double* _covarianceMatrix; /* Covariance of Proposal Distribution */

 // Korali Methods
 void initialize() override;
 void run() override;
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

 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_MCMC_H_
