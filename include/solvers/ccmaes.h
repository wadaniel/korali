#ifndef _KORALI_SOLVERS_CCMAES_H_
#define _KORALI_SOLVERS_CCMAES_H_

#include "solvers/base.h"
#include "variables/gaussian.h"
#include <chrono>
#include <map>

namespace Korali { namespace Solver {

class CCMAES : public Base
{
 public:

 // Constructor / Destructor
 CCMAES(nlohmann::json& js);
 ~CCMAES();

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
 int _fitnessSign; /* maximizing vs optimizing (+- 1) */
 std::string _objective; /* Maximize or Minimize */ 
 double* _fitnessVector; /* objective function values [_s] */
 double* _samplePopulation; /* sample coordinates [_s x _k->N] */
 size_t _currentGeneration; /* generation count */
 bool* _initializedSample; /* flag to distribute work */
 char _terminationReason[500]; /* buffer for exit reason */

 size_t _finishedSamples; /* counter of evaluated samples to terminate evaluation */
 size_t _s; /* number of samples per generation */
 size_t _via_s; /* number of samples during start seach (viability regime) */
 size_t _current_s; /* number of samples active ( _s or _via_s ) */
 size_t _mu; /* number of best samples for mean / cov update */
 size_t _via_mu; /* number of best samples for mean (viability regime) */
 size_t _current_mu; /* number of samples active ( _mu or _mu_s ) */
 std::string _muType; /* Linear, Equal or Logarithmic */
 double* _muWeights; /* weights for mu best samples */
 double _muEffective; /* variance effective selection mass */
 double _muCovarianceIn; /* read from configuration, placeholder for reinit */
 double _muCovariance; /* internal parameter to calibrate updates */

 double _sigmaCumulationFactorIn; /* read from configuration, placeholder for reinit (see below) */
 double _sigmaCumulationFactor; /* increment for sigma, default calculated from muEffective and dimension */
 double _dampFactorIn; /* read from configuration, placeholder for reinit (see below) */
 double _dampFactor; /* dampening parameter determines controls step size adaption */
 double _cumulativeCovarianceIn; /* read from configuration, plcaeholder for reinit (see below) */
 double _cumulativeCovariance; /* default calculated from dimension */
 double _covMatrixLearningRateIn; /* read from configuration, placehoder for reinit (see below) */
 double _covarianceMatrixLearningRate; /* parameter to calibrate cov updates */
 double _chiN; /* expectation of ||N(0,I)||^2 */
 bool   _isdiag; /* diagonal covariance matrix */
 size_t _covarianceEigenEvalFreq;

 // Stop conditions
 size_t _termCondMaxGenerations; // Max number of generations.
 size_t _termCondMaxFitnessEvaluations;   // Defines maximum number of fitness evaluations
 double _termCondFitness; // Defines the maximum fitness allowed, otherwise it stops
 double _termCondFitnessDiffThreshold; // Defines minimum function value differences before stopping
 double _termCondMinDeltaX; // Defines minimum delta of input parameters among generations before it stops.
 double _termCondTolUpXFactor; // Defines the minimum fitness allowed, otherwise it stops
 double _termCondCovCond; // Defines the maximal condition number of the covariance matrix
 bool _isTermCondMaxGenerations, _isTermCondMaxFitnessEvaluations, _isTermCondFitness,
      _isTermCondFitnessDiffThreshold, _isTermCondMinDeltaX, _isTermCondTolUpXFactor,
      _isTermCondCovCond; // flgs to activate termination criteria

 // Private CMAES-Specific Variables
 double sigma;  /* step size */
 Variable::Gaussian* _gaussianGenerator;

 double bestEver; /* best ever fitness */
 double prevBest; /* best ever fitness from previous generation */
 double *rgxmean; /* mean "parent" */
 double *rgxbestever; /* bestever vector */
 double *curBestVector; /* current best vector */
 size_t *index; /* sorting index of current sample pop (index[0] idx of current best). */
 double currentFunctionValue; /* best fitness current generation */
 double prevFunctionValue; /* best fitness previous generation */

 double **C; /* Covariance Matrix */
 double **Ctmp; /* tmp Covariance Matrix for eigen decomp */
 double **B; /* matrix with eigenvectors in columns */
 double **Btmp; /* matrix for eigenvectors calculation */
 double *axisD; /* axis lengths (sqrt(Evals)) */
 double *axisDtmp; /* for axis lengths calculation */
 
 double **Z; /* randn() */
 double **BDZ; /* B*D*randn() */

 double *rgpc; /* evolution path for cov update */
 double *rgps; /* conjugate evolution path for sigma update */
 double *rgxold; /* mean "parent" previous generation */
 double *rgBDz; /* for B*D*z */
 double *rgdTmp; /* temporary (random) vector used in different places */
 double *histFuncValues; /* holding historical best function values */

 size_t countevals; /* Number of function evaluations */
 size_t countinfeasible; /* Number of samples outside of domain given by bounds */
 double maxdiagC; /* max diagonal element of C */
 double mindiagC; /* min diagonal element of C */
 double maxEW; /* max Eigenwert of C */
 double minEW; /* min Eigenwert of C */
 double psL2; /* L2 norm of rgps */

 bool flgEigensysIsUptodate;

 // Private CMA-ES-Specific Methods
 void sampleSingle(size_t sampleIdx); /* sample individual */
 void evaluateSamples(); /* evaluate all samples until done */
 void adaptC(int hsig); /* CMA-ES covariance matrix adaption */
 void updateEigensystem(double **M, int flgforce = 1);
 void eigen(size_t N, double **C, double *diag, double **Q) const;
 size_t maxIdx(const double *rgd, size_t len) const;
 size_t minIdx(const double *rgd, size_t len) const;
 void sort_index(const double *rgFunVal, size_t *index, size_t n) const;
 bool isFeasible(size_t sampleIdx) const; /* check if sample inside lower & upper bounds */
 double doubleRangeMax(const double *rgd, size_t len) const;
 double doubleRangeMin(const double *rgd, size_t len) const;

 // Private CCMA-ES-Specific Variables 
 size_t _numConstraints; /* number of constraints */
 size_t _maxCorrections; /* max cov adaptions per generation */
 size_t _maxResamplings; /* max resamplings per generation */
 double _targetSucRate; /* target success rate */
 double _adaptionSize; /* cov adaption size scaling*/
 double _beta; /* cov adaption size ( adaptionSize/(samples+2) ) */
 double _cv; /* learning rate in normal vector  update */
 double _cp; /* update rate global success estimate */
 
 bool isVia; /* true if mean violates constraints */
 int bestValidIdx; /* best sample with wo constraint violation (otherwise -1) */
 double globalSucRate; /* estim. global success rate */ 
 double fviability; /* viability func value */
 size_t resampled; /* number of resampled parameters due constraint violation */
 size_t correctionsC; /* number of cov matrix adaptions */
 size_t countcevals; /* number of constraint evaluations */
 double *sucRates; /* constraint success rates */
 double *viabilityBounds; /* viability boundaries */
 bool *viabilityImprovement; /* sample evaluations larger than fviability */ //TODO: not neeeded?
 size_t maxnumviolations; /* maximal amount of constraint violations */
 size_t *numviolations; /* number of constraint violations for each sample */
 bool **viabilityIndicator; /* constraint evaluation better than viability bound */
 double **constraintEvaluations; /* evaluation of each constraint for each sample  */
 double **v; /* normal approximation of constraints */
 double *besteverCeval; /* constraint evaluations for best ever */

 // Private CCMA-ES-Specific Methods
 void initInternals(size_t numsamples); /* init _muWeights, _muEffective and _muCov */
 void checkMeanAndSetRegime(); /* check if mean inside valid domain, if yes, update internal vars */
 void updateConstraints();
 void updateViabilityBoundaries(); /* update & shrink viability boundaries */
 void handleConstraints(); /* covariance adaption for invalid samples */
 void reEvaluateConstraints(); /* re evaluate constraints, in handleConstraints,  count violations etc.. */

 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVERS_CCMAES_H_
