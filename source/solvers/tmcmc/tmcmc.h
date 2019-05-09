#ifndef _KORALI_TMCMC_H_
#define _KORALI_TMCMC_H_

#include "solvers/base/base.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

namespace Korali::Solver
{

typedef struct fparam_s {
    const double *fj;
    size_t        fn;
    double        pj;
    double        cov;
} fparam_t;

class TMCMC : public Korali::Solver::Base
{
 public:

 // Constructor / Destructor
 TMCMC(nlohmann::json& js);
 ~TMCMC();

 // TMCMC Configuration
 double _tolCOV; /* Target coefficient of variation of weights */
 double _minStep; /* Min update of rho */
 double _beta2; /* Covariance scaling parameter */
 unsigned int _s; /* Population Size */
 bool _useLocalCov; /* Using local covariance instead of sample cov */
 size_t _burnin; /* burn in generations */
 size_t _maxGens; /* maximal number of generations */

 // TMCMC Runtime Variables
 gsl_rng  *range;
 gsl_rng** chainGSLRange;
 bool*   chainPendingFitness; /* Indicates that the fitness result for the chain is pending */
 double* ccPoints; /* Chain candidate parameter values */
 double* ccLogLikelihood; /* Chain candidate fitness value */
 double* clPoints; /* Chain leader parameter values */
 double* clLogLikelihood; /* Chain leader fitness */
 size_t  finishedChains; 
 size_t* chainCurrentStep;
 size_t* chainLength;

 // TMCMC Status variables
 size_t  _countevals; /* Number of function evaluations */
 size_t  _nChains; /* Unique selections after resampling (forming new chain) */
 size_t  _currentGeneration; /* Generation */
 double  _coefficientOfVariation; /* Actual coefficient of variation of weights */
 double  _annealingExponent; /* Annealing exponent */
 size_t  _uniqueEntries; /* Accepted samples after proposal */
 double  _logEvidence; /* Log of evidence of model */
 double  _acceptanceRate; /* Acceptance rate calculated from _nChains */
 double* _covarianceMatrix; /* Sample covariance of leader fitness values */
 double* _meanTheta; /* Mean of leader fitness values */
 size_t  _databaseEntries; /* Num samples in DB (must equal population size) */
 double* _databasePoints; /* Parameter values of samples in DB */
 double* _databaseFitness; /* Fitness of samples in DB */
 double **local_cov; /* Local covariances of leaders */

  // Korali Methods
 void run() override;

  // Internal TMCMC Methods
 void initializeSamples();
 void resampleGeneration();
 void updateDatabase(double* point, double fitness);
 void processSample(size_t c, double fitness) override;
 void generateCandidate(size_t c);
 void computeChainCovariances(double** chain_cov, size_t newchains) const;
 void minSearch(double const *fj, size_t fn, double pj, double objTol, double& xmin, double& fmin) const;
 static double tmcmc_objlogp(double x, const double *fj, size_t fn, double pj, double zero);
 static double objLog(const gsl_vector *v, void *param);

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
 void setState(nlohmann::json& js) override;

 // Print Methods
 void printGeneration() const;
};

} // namespace Korali

#endif // _KORALI_TMCMC_H_
