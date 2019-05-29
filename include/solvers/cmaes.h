#ifndef _KORALI_SOLVER_CMAES_H_
#define _KORALI_SOLVER_CMAES_H_

#include "solvers/base.h"
#include "variables/gaussian.h"
#include <chrono>
#include <map>

namespace Korali::Solver
{

class CMAES : public Korali::Solver::Base
{
 public:

 // Constructor / Destructor
 CMAES(nlohmann::json& js);
 ~CMAES();

 // Runtime Methods (to be inherited from base class in the future)
 void prepareGeneration();
 bool checkTermination();
 void updateDistribution(const double *fitnessVector);
 void run() override;
 void processSample(size_t sampleId, double fitness) override;

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
 void setState(nlohmann::json& js) override;

 private:

 // Korali Runtime Variables
 std::string _objective; /* Maximize or Minimize */
 int _fitnessSign; /* Maximize = 1, Minimize = -1 */

 size_t _s; /* number of samples per generation */
 size_t _mu; /* number of best samples for mean / cov update */
 
 std::string _muType; /* Linear, Equal or Logarithmic */
 double* _muWeights; /* weights for mu best samples */
 double _muEffective; /* variance effective selection mass */
 double _muCovariance; /* parameter in cov update */

 double _sigmaCumulationFactor; /* default calculated from muEffective and dimension */
 double _dampFactor; /* dampening parameter determines controls step size adaption */
 double _cumulativeCovariance; /* default calculated from dimension */
 double _covarianceMatrixLearningRate; /* exponential learning rate cov */
 double _chiN; /* expectation of ||N(0,I)|| */

 bool   _isdiag; /* diagonal covariance matrix */
 size_t _covarianceEigenEvalFreq;

 Variable::Gaussian* _gaussianGenerator; /* random number generator */
 
 // Stop conditions
 size_t _maxGenenerations; // Max number of generations.
 size_t _maxFitnessEvaluations;   // Defines maximum number of fitness evaluations
 double _stopMinFitness; // Defines the minimum fitness allowed, otherwise it stops
 double _stopFitnessDiffThreshold; // Defines minimum function value differences before stopping
 double _stopMinDeltaX; // Defines minimum delta of input parameters among generations before it stops.
 double _stopTolUpXFactor; // Defines the minimum fitness allowed, otherwise it stops
 double _stopCovCond; // Defines the maximal condition number of the covariance matrix
 std::string _ignorecriteria; /* Termination Criteria(s) to ignore:
    Fitness Value, Fitness Diff Threshold, Max Standard Deviation,
    Max Kondition Covariance, No Effect Axis, No Effect Standard Deviation,
    Max Model Evaluations, Max Generations */

 // Private CMAES-Specific Variables
 size_t currentGeneration; /* generation count */
 double* fitnessVector; /* objective function values [_s] */
 double* samplePopulation; /* sample coordinates [_s x _k->N] */
 
 bool* initializedSample; /* flag if sample currently in evaluation */
 size_t finishedSamples; /* count evaluated samples in generation */

 double sigma;  /* step size */

 double bestEver; /* best ever fitness */
 double prevBest; /* best ever fitness from previous generation */
 double *rgxmean; /* mean "parent" */
 double *rgxold; /* mean "parent" previous generation */
 double *rgxbestever; /* bestever vector */
 double *curBestVector; /* current best vector */
 size_t *index; /* sorting index of current sample pop (index[0] idx of current best). */
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */
 double *histFuncValues; /* holding historical best function values */

 double **C; /* lower triangular matrix: i>=j for C[i][j] */
 double **B; /* matrix with eigenvectors in columns */
 double *rgD; /* axis lengths (sqrt(Evals)) */
 double *rgpc; /* evolution path for cov update */
 double *rgps; /* conjugate evolution path for sigma update */
 
 double *rgBDz; /* for B*D*z */
 double *rgdTmp; /* temporary (random) vector used in different places */

 size_t countevals; /* Number of function evaluations */
 size_t countinfeasible; /* Number of samples outside of domain given by bounds */
 double maxdiagC; /* max diagonal element of C */
 double mindiagC; /* min diagonal element of C */
 double maxEW; /* max Eigenwert of C */
 double minEW; /* min Eigenwert of C */
 double psL2; /* L2 norm of rgps */
 char terminationReason[500]; /* termination reason */

 bool flgEigensysIsUptodate; /* flag if eigen vec need update */

 // Private CMA-ES-Specific Methods
 void sampleSingle(size_t sampleIdx); /* sample individual */
 void evaluateSamples(); /* evaluate all samples until done */
 void adaptC(int hsig); /* CMA-ES covariance matrix adaption */
 void updateEigensystem(int flgforce);
 void eigen(size_t N, double **C, double *diag, double **Q) const;
 size_t maxIdx(const double *rgd, size_t len) const;
 size_t minIdx(const double *rgd, size_t len) const;
 void sort_index(const double *funVal, size_t *index, size_t n) const;
 bool isFeasible(size_t sampleIdx) const;
 double doubleRangeMax(const double *rgd, size_t len) const;
 double doubleRangeMin(const double *rgd, size_t len) const;
 bool isStoppingCriteriaActive(const char *criteria) const;

 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} // namespace Korali

#endif // _KORALI_SOLVER_CMAES_H_
