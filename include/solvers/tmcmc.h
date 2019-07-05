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
Setting Name: Max Function Evaluations
Type: Termination Criterion
Format: Integer
Mandatory: No
Default Value: 20
Default Enabled: true
Description:
Specifies the maximum number of generations to run.
******************************************************************************/
size_t maxGenerations;
bool maxGenerationsEnabled;


/******************************************************************************
Internal Variable Name: Chain Pending Fitness
Format: Array of Booleans
Description:
Indicates that the fitness result for the chain is pending
******************************************************************************/
std::vector<bool> chainPendingFitness;

/******************************************************************************
Internal Variable Name: Chain Candidate Parameters
Format: Array of Reals
Description:
Current (theta) parameters of the chain leader sample.
******************************************************************************/
std::vector<double> chainCandidatesParameters;

/******************************************************************************
Internal Variable Name: Log Transformed Samples
Format: Array of Reals
Description:
Candidate parameters log transformed.
******************************************************************************/
std::vector<double> logTransformedSamples;

/******************************************************************************
Internal Variable Name: Chain Candidates LogLikelihoods
Format: Array of Reals
Description:
The logLikelihoods of the chain candidates.
******************************************************************************/
std::vector<double> chainCandidatesLogLikelihoods;

/******************************************************************************
Internal Variable Name: Chain Leader Parameters
Format: Array of Reals
Description:
Current (theta) parameters of the chain leader sample.
******************************************************************************/
std::vector<double> chainLeadersParameters;

/******************************************************************************
Internal Variable Name: Chain Leaders LogLikelihoods
Format: Array of Reals
Description:
The logLikelihoods of the chain leaders.
******************************************************************************/
std::vector<double> chainLeadersLogLikelihoods;

/******************************************************************************
Internal Variable Name: Finished Chains Count
Format: Integer
Description:
Number of finished chains.
******************************************************************************/
size_t finishedChainsCount;

/******************************************************************************
Internal Variable Name: Current Chain Step
Format: Array of Integers
Description:
The current execution step for every chain.
******************************************************************************/
std::vector<size_t> currentChainStep;

/******************************************************************************
Internal Variable Name: Chain Lengths
Format: Array of Integers
Description:
Length for each of the chains.
******************************************************************************/
std::vector<size_t> chainLengths;

/******************************************************************************
Internal Variable Name: Coefficient of Variation
Format: Real
Description:
Current coefficient of variation
******************************************************************************/
double coefficientOfVariation;


/******************************************************************************
Internal Variable Name: Chain Count
Format: Integer
Description:
Unique selections after resampling (forming new chain)
******************************************************************************/
size_t chainCount;

/******************************************************************************
Internal Variable Name: Annealing Exponent
Format: Real
Description:
Indicates how the calculated distribution fits the real distribution
******************************************************************************/
double annealingExponent;

/******************************************************************************
Internal Variable Name: Accepted Samples Count
Format: Integer
Description:
Accepted samples after proposal
******************************************************************************/
size_t acceptedSamplesCount;

/******************************************************************************
Internal Variable Name: logEvidence
Format: Real
Description:
Calculated logEvidence of the model so far
******************************************************************************/
double logEvidence;

/******************************************************************************
Internal Variable Name: Proposals Acceptance Rate
Format: Real
Description:
Acceptance rate calculated from samples
******************************************************************************/
double proposalsAcceptanceRate;

/******************************************************************************
Internal Variable Name: Selection Acceptance Rate
Format: Real
Description:
Acceptance rate calculated from chain count
******************************************************************************/
double selectionAcceptanceRate;

/******************************************************************************
Internal Variable Name: Covariance Matrix
Format: Array of Reals
Description:
Sample covariance of leader fitness values
******************************************************************************/
std::vector<double> covarianceMatrix;

/******************************************************************************
Internal Variable Name: Mean Theta
Format: Array of Reals
Description:
Mean of leader fitness values
******************************************************************************/
std::vector<double> meanTheta;

/******************************************************************************
Internal Variable Name: Database Entry Count
Format: Integer
Description:
Number of accepted samples stored in the database.
******************************************************************************/
size_t databaseEntryCount;

/******************************************************************************
Internal Variable Name: Sample Parameters Database
Format: Array of Reals
Description:
Variable values of samples stored in the database.
******************************************************************************/
std::vector<double> sampleParametersDatabase;

/******************************************************************************
Internal Variable Name: Sample Fitness Database
Format: Array of Reals
Description:
Fitness of the samples stored in the database.
******************************************************************************/
std::vector<double> sampleFitnessDatabase;

/******************************************************************************
Internal Variable Name: Local Covariance Matrices
Format: Array of Array of Reals
Description:
Local covariances of chain leaders
******************************************************************************/
//std::vector< std::vector<double> > localCovarianceMatrices;

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

 // TMCMC Status variables
 double* _covarianceMatrix; /* Sample covariance of leader fitness values */
 double* _meanTheta; /* Mean of leader fitness values */
 size_t  _databaseEntries; /* Num samples in DB (must equal population size) */
 double* _databasePoints; /* Variable values of samples in DB */
 double* _databaseFitness; /* Fitness of samples in DB */
 double **local_cov; /* Local covariances of leaders */
 bool* _variableLogSpace; /* Apply log transform of variable before evaluation */

 gsl_rng  *range;
 gsl_rng** chainGSLRange;

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
 void computeChainCovariances(double** chain_cov, size_t newchains);
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
