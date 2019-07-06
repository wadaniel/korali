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
Setting Name: Covariance Matrix
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Hold the covariance matrix of proposal distribution.
******************************************************************************/
std::vector<double> covarianceMatrix;

/******************************************************************************
Setting Name: Chain Leader Parameters
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Current (theta) parameters of the chain leader sample.
******************************************************************************/
std::vector<double> chainLeaderParameters;

/******************************************************************************
Setting Name: Chain Leader LogLikelihood
Type: Internal Attribute
Format: Real
Mandatory:
Default Value:
Default Enabled:
Description:
The logLikelihood of the current chain leader.
******************************************************************************/
double chainLeaderLogLikelihood;

/******************************************************************************
Setting Name: Chain Candidate Parameters
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Current (theta) parameters of the chain leader sample.
******************************************************************************/
std::vector<double> chainCandidatesParameters;

/******************************************************************************
Setting Name: Chain Candidates LogLikelihoods
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
The logLikelihoods of the chain candidates.
******************************************************************************/
std::vector<double> chainCandidatesLogLikelihoods;

/******************************************************************************
Setting Name: Chain Candidates LogPriors
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
The logPriors of the chain candidates.
******************************************************************************/
std::vector<double> chainCandidatesLogPriors;

/******************************************************************************
Setting Name: Log Transformed Samples
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Candidate parameters log transformed.
******************************************************************************/
std::vector<double> logTransformedSamples;

/******************************************************************************
Setting Name: Rejection Alphas
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Alphas for recursive calculation of delayed rejection schemes
******************************************************************************/
std::vector<double> rejectionAlphas;

/******************************************************************************
Setting Name: Acceptance Rate
Type: Internal Attribute
Format: Real
Mandatory:
Default Value:
Default Enabled:
Description:
Ratio proposed to accepted samples.
******************************************************************************/
double acceptanceRate;

/******************************************************************************
Setting Name: Rejection Count
Type: Internal Attribute
Format: Integer
Mandatory:
Default Value:
Default Enabled:
Description:
Number of rejections in the current generation.
******************************************************************************/
size_t rejectionCount;

/******************************************************************************
Setting Name: Acceptance Count
Type: Internal Attribute
Format: Integer
Mandatory:
Default Value:
Default Enabled:
Description:
Number of accepted samples.
******************************************************************************/
size_t acceptanceCount;

/******************************************************************************
Setting Name: Proposed Sample Count
Type: Internal Attribute
Format: Integer
Mandatory:
Default Value:
Default Enabled:
Description:
Number of proposed samples.
******************************************************************************/
size_t proposedSampleCount;

/******************************************************************************
Setting Name: Database Entry Count
Type: Internal Attribute
Format: Integer
Mandatory:
Default Value:
Default Enabled:
Description:
Number of accepted samples stored in the database.
******************************************************************************/
size_t databaseEntryCount;

/******************************************************************************
Setting Name: Sample Parameters Database
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Variable values of samples stored in the database.
******************************************************************************/
std::vector<double> sampleParametersDatabase;

/******************************************************************************
Setting Name: Sample Fitness Database
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Fitness of the samples stored in the database.
******************************************************************************/
std::vector<double> sampleFitnessDatabase;

/******************************************************************************
Setting Name: Chain Mean
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Mean of Markov Chain Monte Carlo Chain
******************************************************************************/
std::vector<double> chainMean;

/******************************************************************************
Setting Name: Chain Covariance Placeholder
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Placeholder chain covariance calculation
******************************************************************************/
std::vector<double> chainCovariancePlaceholder;

/******************************************************************************
Setting Name: Chain Covariance
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Chain Covariance
******************************************************************************/
std::vector<double> chainCovariance;

/******************************************************************************
Setting Name: Chain Length
Type: Internal Attribute
Format: Integer
Mandatory:
Default Value:
Default Enabled:
Description:
Current Chain Length
******************************************************************************/
size_t chainLength;

 // MCMC Configuration
 Korali::Variable* _gaussianGenerator; /* Gaussian random number generator */
 Korali::Variable* _uniformGenerator; /* Uniform random number generator */

 // Korali Methods
 void initialize() override;
 void finalize() override;

 void runGeneration() override;
 void processSample(size_t c, double fitness) override;

  // Internal MCMC Methods
 void updateDatabase(std::vector<double>&, double fitness);
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
