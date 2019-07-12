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

** Usage **

```python
  k["Solver"] = "MCMC"
```
	
**Requirements:**

+ The *Chain Length*, i.e. the number of generated samples.
+ The *Initial Mean* of the proposal distribution for each variable.
+ The *Standard Deviation* of the proposal distribution for each variable.
******************************************************************************
Plotting:
Here we explain technical details of the **MCMC** result plot.

The `python3 -m korali.plotter` command plots the distribution of the samples at every
generation. The samples are read from the json-files stored in the output
directory (`/_korali_result/`).

A plot of the samples obtained after the final gerneration of MCMC
function is given below. Here, the target function is the exponential of the
negative of the 2-dimensional [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function)
function.

![figure](sampling_rosenbrock.png)

**Diagonal Plots**. The diagonal plots show the marginal probability densities of the samples for
each variable. Note that the indices of the vertical axes correspond to the
upper and lower triangle plot and not to the diagonal plots.

**Upper Triangle**. In the plots in the upper triangle we see the actual realization of the samples.
The axes represent the dimensions, respectiely the objective variables,
of the problem and we show a two-dimensional plot for every variable pair.

**Lower Triangle**. The lower triangle shows the probability density of the samples for each variable pair.
The density is approximated through a smoothening operator applied to the number
******************************************************************************/

class MCMC : public Base
{
 public:

/******************************************************************************
Setting Name: Max Chain Length
Type: Solver Setting
Default Value:
Default Enabled:
Description:
Number of samples to be generated.
******************************************************************************/
size_t _maxChainLength;

/******************************************************************************
Setting Name: Burn In
Type: Solver Setting
Default Value: 0
Default Enabled:
Description:
Specifies the number of preliminary MCMC steps before samples are being drawn.
This may reduce effects from improper initialization.
******************************************************************************/
size_t _burnIn;

/******************************************************************************
Setting Name: Rejection Levels
Type: Solver Setting
Default Value: 1
Default Enabled:
Description:
Controls the number of accept-reject stages per MCMC step.
******************************************************************************/
size_t _rejectionLevels;

/******************************************************************************
Setting Name: Use Adaptive Sampling
Type: Solver Setting
Default Value: false
Default Enabled:
Description:
Specifies if covariance matrix of the proposal distribution is calculated from
the samples.
******************************************************************************/
bool _useAdaptiveSampling;

/******************************************************************************
Setting Name: Non Adaption Period
Type: Solver Setting
Default Value: 0
Default Enabled:
Description:
Number of steps (after Burn In steps) during which the initial standard deviation
is used (only relevant for Adaptive Sampling). If "0" is specified, Korali will
set this value to $5%$ of the Chain Length.
******************************************************************************/
size_t _nonAdaptionPeriod;

/******************************************************************************
Setting Name: Chain Covariance Increment
Type: Solver Setting
Default Value: 0.001
Default Enabled:
Description:
Small constant to avoid singularity of the chain covariance.
******************************************************************************/
double _chainCovarianceIncrement;

/******************************************************************************
Setting Name: Chain Covariance Scaling
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Small constant to avoid singularity of the chain covariance.
******************************************************************************/
double _chainCovarianceScaling;

/******************************************************************************
Setting Name: Covariance Matrix
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Hold the covariance matrix of proposal distribution.
******************************************************************************/
std::vector<double> _covarianceMatrix;

/******************************************************************************
Setting Name: Chain Leader Parameters
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Parameters of the current sample.
******************************************************************************/
std::vector<double> _chainLeaderParameters;

/******************************************************************************
Setting Name: Chain Leader LogLikelihood
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
The logLikelihood of the current sample.
******************************************************************************/
double _chainLeaderLogLikelihood;

/******************************************************************************
Setting Name: Chain Candidate Parameters
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Current (theta) parameters of the chain leader sample.
******************************************************************************/
std::vector<double> _chainCandidatesParameters;

/******************************************************************************
Setting Name: Chain Candidates LogLikelihoods
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
The logLikelihoods of the chain candidates.
******************************************************************************/
std::vector<double> _chainCandidatesLogLikelihoods;

/******************************************************************************
Setting Name: Chain Candidates LogPriors
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
The logPriors of the chain candidates.
******************************************************************************/
std::vector<double> _chainCandidatesLogPriors;

/******************************************************************************
Setting Name: Rejection Alphas
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Alphas for recursive calculation of delayed rejection schemes
******************************************************************************/
std::vector<double> _rejectionAlphas;

/******************************************************************************
Setting Name: Acceptance Rate
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Ratio proposed to accepted samples.
******************************************************************************/
double _acceptanceRate;

/******************************************************************************
Setting Name: Rejection Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of rejections in the current generation.
******************************************************************************/
size_t _rejectionCount;

/******************************************************************************
Setting Name: Acceptance Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of accepted samples.
******************************************************************************/
size_t _acceptanceCount;

/******************************************************************************
Setting Name: Proposed Sample Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of proposed samples.
******************************************************************************/
size_t _proposedSampleCount;

/******************************************************************************
Setting Name: Database Entry Count
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Number of accepted samples stored in the database.
******************************************************************************/
size_t _databaseEntryCount;

/******************************************************************************
Setting Name: Sample Parameters Database
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Variable values of samples stored in the database.
******************************************************************************/
std::vector<double> _sampleParametersDatabase;

/******************************************************************************
Setting Name: Sample Fitness Database
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Fitness of the samples stored in the database.
******************************************************************************/
std::vector<double> _sampleFitnessDatabase;

/******************************************************************************
Setting Name: Chain Mean
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Mean of Markov Chain Monte Carlo Chain
******************************************************************************/
std::vector<double> _chainMean;

/******************************************************************************
Setting Name: Chain Covariance Placeholder
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Placeholder chain covariance calculation
******************************************************************************/
std::vector<double> _chainCovariancePlaceholder;

/******************************************************************************
Setting Name: Chain Covariance
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Chain Covariance
******************************************************************************/
std::vector<double> _chainCovariance;

/******************************************************************************
Setting Name: Chain Length
Type: Internal Attribute
Default Value:
Default Enabled:
Description:
Current Chain Length
******************************************************************************/
size_t _chainLength;

/******************************************************************************
* Variable Settings
******************************************************************************/

struct variableSetting
{

/******************************************************************************
Setting Name: Initial Mean
Type: Variable Setting
Default Value:
Default Enabled:
Description:
Specifies the Initial Mean of the proposal distribution.
******************************************************************************/
double initialMean;

/******************************************************************************
Setting Name: Standard Deviation
Type: Variable Setting
Default Value:
Default Enabled:
Description:
Specifies the Standard Deviation for each variable. The proposal distribution
is defined through a covariance matrix with the variance of the variables in
its diagonal.
******************************************************************************/
double standardDeviation;

};

std::vector<variableSetting> _variableSettings;
/******************************************************************************/

// MCMC Configuration

 Korali::Variable* _gaussianGenerator; /* Gaussian random number generator */
 Korali::Variable* _uniformGenerator; /* Uniform random number generator */

 MCMC();
 ~MCMC();

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
