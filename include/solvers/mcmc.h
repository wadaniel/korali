#ifndef _KORALI_SOLVER_MCMC_H_
#define _KORALI_SOLVER_MCMC_H_

#include "solvers/base.h"
#include "variable.h"
#include <gsl/gsl_rng.h>

namespace Korali { namespace Solver {

class MCMC : public Base
{
 public:

 // Constructor / Destructor
 MCMC(nlohmann::json& js, std::string name);
 ~MCMC();

 // MCMC Configuration
 unsigned int _s; /* Population Size */
 size_t _burnin; /* burn in generations */
 size_t _rejectionLevels; /* delayed rejection if > 1*/
 bool _adaptive; /* Using chain covariance for proposal */
 size_t _nonAdaptionPeriod; /* Period without Chain Cov Adaption (using initial Stddev) */
 double _cr; /* Learning rate or Chain Covariance */
 double _eps; /* Chain Covariance increment */
 //size_t _maxresamplings; /* Max number resamplings inside generation loop */
 bool _isTermCondMaxFunEvals;
 size_t _termCondMaxFunEvals; /* Max objective function evaluations */
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
 size_t chainLength; /* Number accepted samples + burnin accepted samples */
 size_t databaseEntries; /* Accepted Samples */
 double* databasePoints; /* Variable values of samples in DB */
 double* databaseFitness; /* Fitness of samples in DB */
 size_t countevals; /* Number of function evaluations */
 double* chainMean; /* Mean of mcmc chain */
 double* tmpC; /* Placeholder chain cov calculation */
 double* chainCov; /* Variance of mcmc chain */

 // MCMC Status variables
 double* _initialMeans; /* Initial Mean of Cov Proposal Distribution */
 double* _stdDevs; /* Diagonal of Cov Proposal Distribution */
 bool* _variableLogSpace; /* Apply log transform of variable before evaluation */
 double* _covarianceMatrix; /* Covariance of Proposal Distribution */

 // Korali Methods
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

 // Serialization Methods
 void getConfiguration(nlohmann::json& js) override;
 void setConfiguration(nlohmann::json& js) override;
 void setState(nlohmann::json& js) override;
 void saveState() const override;

 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_MCMC_H_
