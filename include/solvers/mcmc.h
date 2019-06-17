#ifndef _KORALI_SOLVER_MCMC_H_
#define _KORALI_SOLVER_MCMC_H_

#include "solvers/base.h"
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
 bool _adaptive; /* Using chain covariance for proposal */
 size_t _nonAdaptionPeriod; /* Period without Chain Cov Adaption (using initial Stddev) */
 double _cr; /* Learning rate or Chain Covariance */
 double _eps; /* Chain Covariance increment */
 size_t _maxresamplings; /* Max number resamplings inside generation loop */
 bool _isTermCondMaxFunEvals;
 size_t _termCondMaxFunEvals; /* Max objective function evaluations */
 bool _isTermCondMaxGenerations;
 size_t _termCondMaxGenerations; /* Max proposed samples */
 char _terminationReason[500];


 Variable::Gaussian* _gaussianGenerator; /* Gaussian random number generator */

 // MCMC Runtime Variables
 double* z; /* Placeholder random numbers */
 double* clPoint; /* Leader parameter values */
 double clLogLikelihood; /* Leader fitness value */
 double* ccPoint; /*  Candidate parameter values */
 double ccLogLikelihood; /* Candidate fitness value */
 double ccLogPrior; /* Candidate prior value */
 double acceptanceRateProposals; /* Ratio proposed to accepted Samples */
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
 double* _initialMean; /* Initial Mean of Cov Proposal Distribution */
 double* _initialStdDevs; /* Initial Diagonal of Cov Proposal Distribution */
 double* _covarianceMatrix; /* Covariance of Proposal Distribution */

 gsl_rng *_gslGen;
 
 // Korali Methods
 void run() override;
 void processSample(size_t c, double fitness) override;

  // Internal MCMC Methods
 void updateDatabase(double* point, double fitness);
 void generateCandidate(); 
 void sampleCandidate();
 void acceptReject();
 //void computeChainCovariances(double** chain_cov, size_t newchains) const;
 void updateState();
 bool setCandidatePriorAndCheck();
 bool checkTermination();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
 void setState(nlohmann::json& js) override;

 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_MCMC_H_
