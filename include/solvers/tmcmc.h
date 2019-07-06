#ifndef _KORALI_SOLVER_TMCMC_H_
#define _KORALI_SOLVER_TMCMC_H_

#include "solvers/base.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

namespace Korali { namespace Solver {

/******************************************************************************
Module Name: Transitional Markov Chain Monte Carlo
Type: Solver, Sampler
Alias: TMCMC
Description:
This is the implementation of the *Transitional Markov Chain Monte Carlo*
algorithm, as published in [Ching2007](https://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399%282007%29133%3A7%28816%29).

TMCMC avoids sampling from difficult target probability densities
(e.g. posterior distributions in a Bayesian inference problem) but samples from
a series of intermediate PDFs that converge to the target PDF.
This technique is also known as Sampling Importance Resampling in the Bayesian community.

**Requirements:**

+ The *Population Size* used at every generation needs to be defined.
+ The *Problem* needs to be of type Bayesian.
******************************************************************************/

class TMCMC : public Base
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
Setting Name: Population Size
Type: Solver Setting
Format: Integer
Mandatory: Yes
Default Value:
Default Enabled:
Description:
Specifies the number of samples drawn from the posterior distribution at each generation.
******************************************************************************/
size_t populationSize;

/******************************************************************************
Setting Name: Burn In
Type: Solver Setting
Format: Integer
Mandatory: No
Default Value: 0
Default Enabled:
Description:
Specifies the number of additional MCMC steps per sample per generation.
Note that only the last sample per chain is considered for the recombination.
******************************************************************************/
size_t burnIn;

/******************************************************************************
Setting Name: Initial Coefficient of Variation
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 1.0
Default Enabled:
Description:
Initial value for the target coefficient of variation to search for the exponent
 $\rho_{i+1}$. By default, Korali will set this value to 1.00 as suggested in [Ching2007].
******************************************************************************/
double initialCoefficientOfVariation;

/******************************************************************************
Setting Name: Covariance Scaling
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.04
Default Enabled:
Description:
Covariance scaling factor $\beta^2$ of proposal distribution.
By default, Korali sets this value 0.04 as suggested in [Ching2007].
******************************************************************************/
double covarianceScaling;

/******************************************************************************
Setting Name: Use Local Covariance
Type: Solver Setting
Format: Boolean
Mandatory: No
Default Value: false
Default Enabled:
Description:
If $true, TMCMC calculates a local covariance matrix per sample from its neighbours.
******************************************************************************/
bool useLocalCovariance;

/******************************************************************************
Setting Name: Min Rho Update
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 0.00001
Default Enabled:
Description:
Minimum increment of the exponent $\rho_{i+1}$. This parameter prevents TMCMC from stalling.
******************************************************************************/
double minRhoUpdate;

/******************************************************************************
Setting Name: Max Rho Update
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 1.0
Default Enabled:
Description:
Maximum increment of the exponent $\rho{i+1}$. By default, Korali will set this value to 1.0 (inactive).
******************************************************************************/
double maxRhoUpdate;

/******************************************************************************
Setting Name: Chain Pending Fitness
Type: Internal Attribute
Format: Array of Booleans
Mandatory:
Default Value:
Default Enabled:
Description:
Indicates that the fitness result for the chain is pending
******************************************************************************/
std::vector<bool> chainPendingFitness;

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
Setting Name: Chain Leader Parameters
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Current (theta) parameters of the chain leader sample.
******************************************************************************/
std::vector<double> chainLeadersParameters;

/******************************************************************************
Setting Name: Chain Leaders LogLikelihoods
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
The logLikelihoods of the chain leaders.
******************************************************************************/
std::vector<double> chainLeadersLogLikelihoods;

/******************************************************************************
Setting Name: Finished Chains Count
Type: Internal Attribute
Format: Integer
Mandatory:
Default Value:
Default Enabled:
Description:
Number of finished chains.
******************************************************************************/
size_t finishedChainsCount;

/******************************************************************************
Setting Name: Current Chain Step
Type: Internal Attribute
Format: Array of Integers
Mandatory:
Default Value:
Default Enabled:
Description:
The current execution step for every chain.
******************************************************************************/
std::vector<size_t> currentChainStep;

/******************************************************************************
Setting Name: Chain Lengths
Type: Internal Attribute
Format: Array of Integers
Mandatory:
Default Value:
Default Enabled:
Description:
Length for each of the chains.
******************************************************************************/
std::vector<size_t> chainLengths;

/******************************************************************************
Setting Name: Coefficient of Variation
Type: Internal Attribute
Format: Real
Mandatory:
Default Value:
Default Enabled:
Description:
Current coefficient of variation
******************************************************************************/
double coefficientOfVariation;

/******************************************************************************
Setting Name: Chain Count
Type: Internal Attribute
Format: Integer
Mandatory:
Default Value:
Default Enabled:
Description:
Unique selections after resampling (forming new chain)
******************************************************************************/
size_t chainCount;

/******************************************************************************
Setting Name: Annealing Exponent
Type: Internal Attribute
Format: Real
Mandatory:
Default Value:
Default Enabled:
Description:
Indicates how the calculated distribution fits the real distribution
******************************************************************************/
double annealingExponent;

/******************************************************************************
Setting Name: Accepted Samples Count
Type: Internal Attribute
Format: Integer
Mandatory:
Default Value:
Default Enabled:
Description:
Accepted samples after proposal
******************************************************************************/
size_t acceptedSamplesCount;

/******************************************************************************
Setting Name: logEvidence
Type: Internal Attribute
Format: Real
Mandatory:
Default Value:
Default Enabled:
Description:
Calculated logEvidence of the model so far
******************************************************************************/
double logEvidence;

/******************************************************************************
Setting Name: Proposals Acceptance Rate
Type: Internal Attribute
Format: Real
Mandatory:
Default Value:
Default Enabled:
Description:
Acceptance rate calculated from samples
******************************************************************************/
double proposalsAcceptanceRate;

/******************************************************************************
Setting Name: Selection Acceptance Rate
Type: Internal Attribute
Format: Real
Mandatory:
Default Value:
Default Enabled:
Description:
Acceptance rate calculated from chain count
******************************************************************************/
double selectionAcceptanceRate;

/******************************************************************************
Setting Name: Covariance Matrix
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Sample covariance of leader fitness values
******************************************************************************/
std::vector<double> covarianceMatrix;

/******************************************************************************
Setting Name: Mean Theta
Type: Internal Attribute
Format: Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Mean of leader fitness values
******************************************************************************/
std::vector<double> meanTheta;

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
Setting Name: Local Covariance Matrices
Type: Internal Attribute
Format: Array of Array of Reals
Mandatory:
Default Value:
Default Enabled:
Description:
Local covariances of chain leaders
******************************************************************************/
std::vector<std::vector<double>> localCovarianceMatrices;

 // TMCMC Status variables
 gsl_rng  *range;

 TMCMC();

  // Korali Methods
 void initialize() override;
 void finalize() override;

 void runGeneration() override;
 void processSample(size_t c, double fitness) override;
 bool checkTermination() override;

  // Internal TMCMC Methods
 void initializeSamples();
 void resampleGeneration();
 void updateDatabase(double* point, double fitness);
 void generateCandidate(size_t candidate);
 void evaluateSample(size_t candidate);
 void computeChainCovariances(std::vector< std::vector<double> >& chain_cov, size_t newchains);
 void minSearch(double const *fj, size_t fn, double pj, double objTol, double& xmin, double& fmin);
 bool isFeasibleCandidate(size_t candidate);
 static double tmcmc_objlogp(double x, const double *fj, size_t fn, double pj, double zero);
 static double objLog(const gsl_vector *v, void *param);

 void setConfiguration() override;
 void getConfiguration() override;
 void printGeneration() override;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_TMCMC_H_
