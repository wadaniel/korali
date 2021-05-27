/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: LMCMAES.
*/

/** \dir solver/optimizer/LMCMAES
* @brief Contains code, documentation, and scripts for module: LMCMAES.
*/


#ifndef _KORALI_SOLVER_OPTIMIZER_LMCMAES_
#define _KORALI_SOLVER_OPTIMIZER_LMCMAES_


#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/solver/optimizer/optimizer.hpp"
#include <vector>

namespace korali
{
namespace solver
{
namespace optimizer
{


/**
* @brief Class declaration for module: LMCMAES.
*/
class LMCMAES : public Optimizer
{
  private:
  /**
  * @brief Prepares generation for the next set of evaluations
 */
  void prepareGeneration();

  /**
  * @brief Initializes the weights of the mu vector.
  * @param numsamples Length of mu vector.
 */
  void initMuWeights(size_t numsamples); /* init _muWeights and dependencies */

  /**
  * @brief Initialize Covariance Matrix and Cholesky Decomposition.
 */
  void initCovariance(); /* init sigma, C and B */

  /**
  * @brief Generate new Sample.
  */
  void sampleSingle(size_t sampleIdx);

  /**
  * @brief Updates set of historical samples, required for cholesky factor calculation.
  */
  void choleskyFactorUpdate(size_t sampleIdx);

  /**
  */
  void updateSet();

  /**
  * @brief Updates vectors for cholesky factor computation.
  */
  void updateInverseVectors();

  /**
 * @brief Updates mean and covariance of Gaussian proposal distribution.
 */
  void updateDistribution(std::vector<Sample> &samples);

  /**
  * @brief Update covariance matrix scaling
  */
  void updateSigma();

  /**
  * @brief Method that checks potential numerical issues and does correction. Not yet implemented.
 */
  void numericalErrorTreatment();

  /**
  * @brief Descending sort of vector elements, stores ordering in _sortingIndex.
  * @param vec Vector to sort, _sortingIndex Ordering of elements in vector
 */
  void sort_index(const std::vector<double> &vec, std::vector<size_t> &_sortingIndex) const;

  /**
  * @brief Flag if random number source is Normal or Uniform.
 */
  bool _normalRandomNumbers;

  public: 
  /**
  * @brief Specifies the number of samples to evaluate per generation (preferably $4+3*log(N)$, where $N$ is the number of variables).
  */
   size_t _populationSize;
  /**
  * @brief Number of best samples used to update the covariance matrix and the mean (by default it is half the Sample Count).
  */
   size_t _muValue;
  /**
  * @brief Weights given to the Mu best values to update the covariance matrix and the mean.
  */
   std::string _muType;
  /**
  * @brief Initial scaling factor for sample distribution.
  */
   double _initialSigma;
  /**
  * @brief Random Number to mutate
  */
   std::string _randomNumberDistribution;
  /**
  * @brief Sample every odd sample reflected in current mean.
  */
   int _symmetricSampling;
  /**
  * @brief Controls the learning rate of the conjugate evolution path (must be in (0,1]).
  */
   double _sigmaCumulationFactor;
  /**
  * @brief Controls the updates of the covariance matrix scaling factor (must be in (0,1]).
  */
   double _dampFactor;
  /**
  * @brief Sets an upper bound for the covariance matrix scaling factor. The upper bound is given by the average of the initial standard deviation of the variables.
  */
   int _isSigmaBounded;
  /**
  * @brief Controls the learning rate of the evolution path for the covariance update (must be in (0,1]).
  */
   double _cumulativeCovariance;
  /**
  * @brief Controls the learning rate of the Cholesky factor (must be in (0,1]).
  */
   double _choleskyMatrixLearningRate;
  /**
  * @brief Coefficients that define target distance between consecutive vectors in Evolution Path Matrix. Target distance calculated as a0 + a1 * ( (j+1)/Subset Size)^a2, where j corresponds to the jth oldest Evolution Path. By default target distance equals N.
  */
   std::vector<double> _targetDistanceCoefficients;
  /**
  * @brief Target population success rate. Sigma increases if population success rate is larger than target. Success rate estimated from comparison of previous and current function values.
  */
   double _targetSuccessRate;
  /**
  * @brief Intervals at which the Evolution Path Matrix and the Inverse Vectors are being updated.
  */
   size_t _setUpdateInterval;
  /**
  * @brief Number of vectors used to reconstruct the Cholesky factor (old version uses 4+3log(N)). Larger Subset Size increases internal cost but usually improves performance.
  */
   size_t _subsetSize;
  /**
  * @brief [Internal Use] Normal random number generator.
  */
   korali::distribution::univariate::Normal* _normalGenerator;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] Objective function values.
  */
   std::vector<double> _valueVector;
  /**
  * @brief [Internal Use] Weights for each of the Mu samples.
  */
   std::vector<double> _muWeights;
  /**
  * @brief [Internal Use] Variance effective selection mass.
  */
   double _effectiveMu;
  /**
  * @brief [Internal Use] Controls step size increment.
  */
   double _sigmaExponentFactor;
  /**
  * @brief [Internal Use] Determines the step size. Initialized by the larger value of either Initial Sigma or 30% of the domain size (averaged over all dimensions).
  */
   double _sigma;
  /**
  * @brief [Internal Use] Sample coordinate information.
  */
   std::vector<std::vector<double>> _samplePopulation;
  /**
  * @brief [Internal Use] Counter of evaluated samples to terminate evaluation.
  */
   size_t _finishedSampleCount;
  /**
  * @brief [Internal Use] Best value found as of previous generation.
  */
   double _previousBestValue;
  /**
  * @brief [Internal Use] Best variables of current generation.
  */
   std::vector<double> _currentBestVariables;
  /**
  * @brief [Internal Use] Index of the best sample in current generation.
  */
   size_t _bestSampleIndex;
  /**
  * @brief [Internal Use] Sorted indeces of samples according to their model evaluation.
  */
   std::vector<size_t> _sortingIndex;
  /**
  * @brief [Internal Use] Vector storing random numbers for sample generation.
  */
   std::vector<double> _randomVector;
  /**
  * @brief [Internal Use] Current column to replace in Evolution Path History and Inverse Vectors.
  */
   size_t _replacementIndex;
  /**
  * @brief [Internal Use] Historical column updates of Evolution Paths.
  */
   std::vector<size_t> _subsetHistory;
  /**
  * @brief [Internal Use] Stores timestamps of updated Evolution Paths.
  */
   std::vector<double> _subsetUpdateTimes;
  /**
  * @brief [Internal Use] Placeholder for Cholesky Factor product with Random Vector.
  */
   std::vector<double> _choleskyFactorVectorProduct;
  /**
  * @brief [Internal Use] Minimum entry in Cholesky Fector Vector Product vector.
  */
   double _minCholeskyFactorVectorProductEntry;
  /**
  * @brief [Internal Use] Maximum entry in Cholesky Fector Vector Product vector.
  */
   double _maxCholeskyFactorVectorProductEntry;
  /**
  * @brief [Internal Use] Vectors storing some of previous evolution paths.
  */
   std::vector<std::vector<double>> _evolutionPathHistory;
  /**
  * @brief [Internal Use] Matrix storing the inverse vectors.
  */
   std::vector<std::vector<double>> _inverseVectors;
  /**
  * @brief [Internal Use] Current mean of proposal distribution.
  */
   std::vector<double> _currentMean;
  /**
  * @brief [Internal Use] Previous mean of proposal distribution.
  */
   std::vector<double> _previousMean;
  /**
  * @brief [Internal Use] Update differential from previous to current mean.
  */
   std::vector<double> _meanUpdate;
  /**
  * @brief [Internal Use] Evolution path for Covariance Matrix update.
  */
   std::vector<double> _evolutionPath;
  /**
  * @brief [Internal Use] Weights for the calculation of the Cholesky Factor.
  */
   std::vector<double> _evolutionPathWeights;
  /**
  * @brief [Internal Use] L2 Norm of the conjugate evolution path.
  */
   double _conjugateEvolutionPathL2Norm;
  /**
  * @brief [Internal Use] Keeps count of the number of infeasible samples.
  */
   size_t _infeasibleSampleCount;
  /**
  * @brief [Internal Use] Shared variable to speed up computation.
  */
   double _sqrtInverseCholeskyRate;
  /**
  * @brief [Internal Use] Expectation of $||N(0,I)||^2$.
  */
   double _chiSquareNumber;
  /**
  * @brief [Internal Use] Scaling factors for samples (read from Initial Standard Deviation or calculated as 30% of the domain widths).
  */
   std::vector<double> _standardDeviation;
  /**
  * @brief [Termination Criteria] Specifies the minimum target fitness to stop minimization.
  */
   double _minValue;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  /**
 * @brief Configures LMCMA-ES.
 */
  void setInitialConfiguration() override;

  /**
 * @brief Executes sampling & evaluation generation.
 */
  void runGeneration() override;

  /**
 * @brief Console Output before generation runs.
 */
  void printGenerationBefore() override;

  /**
 * @brief Console output after generation.
 */
  void printGenerationAfter() override;

  /**
 * @brief Final console output at termination.
 */
  void finalize() override;
};

} //optimizer
} //solver
} //korali


#endif // _KORALI_SOLVER_OPTIMIZER_LMCMAES_

