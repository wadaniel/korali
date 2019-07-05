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
Setting Name: Coefficient of Variation
Type: Solver Setting
Format: Real
Mandatory: No
Default Value: 1.0
Default Enabled:
Description:
Target coefficient of variation to search for the exponent $\rho_{i+1}$.
By default, Korali will set this value to 1.00 as suggested in [Ching2007].
******************************************************************************/
double coefficientOfVariation;

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

 // TMCMC Runtime Variables
 gsl_rng  *range;
 gsl_rng** chainGSLRange;
 bool*   chainPendingFitness; /* Indicates that the fitness result for the chain is pending */
 double* ccPoints; /* Chain candidate parameter values */
 double* transformedSamples; /* Candidate parameters log transformed */
 double* ccLogLikelihood; /* Chain candidate fitness value */
 double* clPoints; /* Chain leader parameter values */
 double* clLogLikelihood; /* Chain leader fitness */
 size_t  finishedChains; 
 size_t* chainCurrentStep;
 size_t* chainLength;

 // TMCMC Status variables
 size_t  _countevals; /* Number of function evaluations */
 size_t  _nChains; /* Unique selections after resampling (forming new chain) */
 double  _coefficientOfVariation; /* Actual coefficient of variation of weights */
 double  _annealingExponent; /* Annealing exponent */
 size_t  _uniqueEntries; /* Accepted samples after proposal */
 double  _logEvidence; /* Log of evidence of model */
 double  _acceptanceRateProposals; /* Acceptance rate calculated from _uniqueEntries */
 double  _acceptanceRateSelections; /* Acceptance rate calculated from _nChains */
 double* _covarianceMatrix; /* Sample covariance of leader fitness values */
 double* _meanTheta; /* Mean of leader fitness values */
 size_t  _databaseEntries; /* Num samples in DB (must equal population size) */
 double* _databasePoints; /* Variable values of samples in DB */
 double* _databaseFitness; /* Fitness of samples in DB */
 double **local_cov; /* Local covariances of leaders */
 bool* _variableLogSpace; /* Apply log transform of variable before evaluation */

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
 void computeChainCovariances(double** chain_cov, size_t newchains) const;
 void minSearch(double const *fj, size_t fn, double pj, double objTol, double& xmin, double& fmin) const;
 bool isFeasibleCandidate(size_t candidate) const;
 static double tmcmc_objlogp(double x, const double *fj, size_t fn, double pj, double zero);
 static double objLog(const gsl_vector *v, void *param);

 void setConfiguration() override;
 void getConfiguration() override;
 void printGeneration() override;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_TMCMC_H_
