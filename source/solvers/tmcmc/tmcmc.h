#ifndef _KORALI_TMCMC_H_
#define _KORALI_TMCMC_H_

#include "solvers/base/base.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

namespace Korali::Solver
{

typedef struct fparam_s {
    const double *fj;
    int           fn;
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
 double _tolCOV;              /* Target coefficient of variation of weights */
 double _minStep;             /* Min update of rho */
 double _bbeta;               /* Covariance scaling parameter */
 size_t  _s; // Population Size
 bool _useLocalCov;
 size_t  _baseBurnIn;
 size_t _maxGens;

 // TMCMC Runtime Variables
 gsl_rng  *range;
 gsl_rng** chainGSLRange;
 bool*   chainPendingFitness; // Indicates that the fitness result for the chain is pending
 double* ccPoints;   // Chain Candidate Parameter Values
 double* ccFitness;  // Chain Candidate Fitness
 double* clPoints;   // Chain Leader Parameter Values
 double* clFitness;  // Chain Leader Fitness
 size_t  finishedChains;
 size_t* chainCurrentStep;
 size_t* chainLength;

 // TMCMC Status variables
 size_t  _nChains;
 size_t  _currentBurnIn;
 size_t  _currentGeneration;
 double  _coefficientOfVariation;
 double  _annealingRatio;
 size_t  _uniqueSelections; /* unique samples after reslection */
 size_t  _uniqueEntries;    /* accepted samples after proposal (TODO: not needed? (DW)) */
 double  _logEvidence;
 double  _acceptanceRate;
 double* _covarianceMatrix;
 double* _meanTheta;
 size_t  _databaseEntries;
 double* _databasePoints;
 double* _databaseFitness;
 double **local_cov;

  // Korali Methods
 void run() override;

  // Internal TMCMC Methods
 void resampleGeneration();
 void updateDatabase(double* point, double fitness);
 void processSample(size_t c, double fitness) override;
 void generateCandidate(int c);
 void computeChainCovariances(double** chain_cov, int newchains);
 void minSearch(double const *fj, int fn, double pj, double objTol, double *xmin, double *fmin);
 static double tmcmc_objlogp(double x, const double *fj, int fn, double pj, double zero);
 static double objLog(const gsl_vector *v, void *param);

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
 void setState(nlohmann::json& js) override;

 // Print Methods
 void printGeneration() const override;
 void printFinal() const override;
};

} // namespace Korali

#endif // _KORALI_TMCMC_H_
