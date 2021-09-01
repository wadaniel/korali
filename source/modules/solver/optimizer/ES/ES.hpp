/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: ES.
*/

/** \dir solver/optimizer/ES
* @brief Contains code, documentation, and scripts for module: ES.
*/

#pragma once

#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/solver/optimizer/optimizer.hpp"

namespace korali
{
namespace solver
{
namespace optimizer
{
;

/**
* @brief Class declaration for module: ES.
*/
class ES : public Optimizer
{
  public: 
  /**
  * @brief Specifies the number of samples to evaluate per generation (preferably $4+3*log(N)$, where $N$ is the number of variables).
  */
   size_t _populationSize;
  /**
  * @brief Covariance matrix updates will be optimized for diagonal matrices.
  */
   int _diagonalCovariance;
  /**
  * @brief Generate the negative counterpart of each random number during sampling.
  */
   int _mirroredSampling;
  /**
  * @brief [Internal Use] Normal random number generator.
  */
   korali::distribution::univariate::Normal* _normalGenerator;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] Descending Sorting Index of Value Vector.
  */
   std::vector<size_t> _sortingIndex;
  /**
  * @brief [Internal Use] Objective function values.
  */
   std::vector<double> _valueVector;
  /**
  * @brief [Internal Use] Sample coordinate information.
  */
   std::vector<std::vector<double>> _samplePopulation;
  /**
  * @brief [Internal Use] Counter of evaluated samples to terminate evaluation.
  */
   size_t _finishedSampleCount;
  /**
  * @brief [Internal Use] Best variables of current generation.
  */
   std::vector<double> _currentBestVariables;
  /**
  * @brief [Internal Use] Best variables of current generation.
  */
   std::vector<double> _bestEverVariables;
  /**
  * @brief [Internal Use] Best model evaluation from previous generation.
  */
   double _previousBestValue;
  /**
  * @brief [Internal Use] Index of the best sample in current generation.
  */
   size_t _bestSampleIndex;
  /**
  * @brief [Internal Use] Best ever model evaluation as of previous generation.
  */
   double _previousBestEverValue;
  /**
  * @brief [Internal Use] (Unscaled) covariance Matrix of proposal distribution.
  */
   std::vector<double> _covarianceMatrix;
  /**
  * @brief [Internal Use] Current mean of proposal distribution.
  */
   std::vector<double> _currentMean;
  /**
  * @brief [Internal Use] Previous mean of proposal distribution.
  */
   std::vector<double> _previousMean;
  /**
  * @brief [Internal Use] Keeps count of the number of infeasible samples.
  */
   size_t _infeasibleSampleCount;
  /**
  * @brief [Internal Use] Maximum diagonal element of the Covariance Matrix.
  */
   double _maximumDiagonalCovarianceMatrixElement;
  /**
  * @brief [Internal Use] Minimum diagonal element of the Covariance Matrix.
  */
   double _minimumDiagonalCovarianceMatrixElement;
  /**
  * @brief [Termination Criteria] Maximum number of resamplings per candidate per generation if sample is outside of Lower and Upper Bound.
  */
   size_t _maxInfeasibleResamplings;
  
 
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
  * @brief Prepares generation for the next set of evaluations
 */
  void prepareGeneration();

  /**
  * @brief Evaluates a single sample
  * @param sampleIdx Index of the sample to evaluate
  * @param randomNumbers Random numbers to generate sample
 */
  void sampleSingle(size_t sampleIdx, const std::vector<double>& randomNumbers);

  /**
 * @brief Updates mean and covariance of Gaussian proposal distribution.
 */
  void updateDistribution();

  /**
  * @brief Descending sort of vector elements, stores ordering in _sortingIndex.
  * @param _sortingIndex Ordering of elements in vector
  * @param vec Vector to sort
  * @param N Number of current samples.
 */
  void sort_index(const std::vector<double> &vec, std::vector<size_t> &_sortingIndex, size_t N) const;

  /**
 * @brief Configures CMA-ES.
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
;
