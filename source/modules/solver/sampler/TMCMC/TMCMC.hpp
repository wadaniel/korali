/** \namespace sampler
* @brief Namespace declaration for modules of type: sampler.
*/

/** \file
* @brief Header file for module: TMCMC.
*/

/** \dir solver/sampler/TMCMC
* @brief Contains code, documentation, and scripts for module: TMCMC.
*/


#ifndef _KORALI_SOLVER_SAMPLER_TMCMC_
#define _KORALI_SOLVER_SAMPLER_TMCMC_


#include "modules/distribution/distribution.hpp"
#include "modules/distribution/multivariate/normal/normal.hpp"
#include "modules/distribution/specific/multinomial/multinomial.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/solver/sampler/sampler.hpp"
#include <gsl/gsl_vector.h>


namespace korali
{
namespace solver
{
namespace sampler
{


/**
  * @brief Struct for TMCMC optimization operations
 */
typedef struct fparam_s
{
  /**
  * @brief Likelihood values in current generation
 */
  const double *loglike;

  /**
    * @brief Population size of current generation
   */
  size_t Ns;

  /**
    * @brief Annealing exponent of current generation
   */
  double exponent;

  /**
    * @brief Target coefficient of variation
   */
  double cov;
} fparam_t;


/**
* @brief Class declaration for module: TMCMC.
*/
class TMCMC : public Sampler
{
  private:
  /*
  * @brief Sets the burn in steps per generation
  */
  void setBurnIn();

  /*
  * @brief Prepare Generation before evaluation.
  */
  void prepareGeneration();

  /*
  * @brief Process Generation after receiving all results.
  */
  void processGeneration();

  /*
  * @brief Helper function for annealing exponent update/
  * @param fj Pointer to exponentiated probability values.
  * @param fn Current exponent.
  * @param pj Number of values in fj array.
  * @paran objTol
  * @param xmin Location of minimum, the new exponent.
  * @param fmin Found minimum in search.
  */
  void minSearch(double const *fj, size_t fn, double pj, double objTol, double &xmin, double &fmin);

  /*
  * @brief Collects results after sampleevaluation.
  */
  void processCandidate(const size_t sampleId);

  /*
  * @brief Calculate gradients of loglikelihood (only relevant for mTMCMC).
  */
  void calculateGradients(std::vector<Sample> &samples);

  /*
  * @brief Calculate sample wise proposal distributions (only relevant for mTMCMC).
  */
  void calculateProposals(std::vector<Sample> &samples);

  /*
  * @brief Generate candidate from leader.
  */
  void generateCandidate(const size_t sampleId);

  /*
  * @brief Add leader into sample database.
  */
  void updateDatabase(const size_t sampleId);

  /*
  * @brief Calculate acceptance probability.
  */
  double calculateAcceptanceProbability(const size_t sampleId);

  /*
  * @brief Helper function to calculate objective (CVaR) for min search.
  */
  static double tmcmc_objlogp(double x, const double *fj, size_t fn, double pj, double zero);

  /*
  * @brief Helper function to calculate objective (CVaR) for min search.
  */
  static double objLog(const gsl_vector *v, void *param);

  /*
  * @brief Number of variables to sample.
  */
  size_t N;

  public: 
  /**
  * @brief Indicates which variant of the TMCMC algorithm to use.
  */
   std::string _version;
  /**
  * @brief Specifies the number of samples drawn from the posterior distribution at each generation.
  */
   size_t _populationSize;
  /**
  * @brief Chains longer than Max Chain Length will be broken and samples will be duplicated (replacing samples associated with a chain length of 0). Max Chain Length of 1 corresponds to the BASIS algorithm [Wu2018].
  */
   size_t _maxChainLength;
  /**
  * @brief Specifies the number of additional TMCMC steps per chain per generation (except for generation 0 and 1).
  */
   size_t _defaultBurnIn;
  /**
  * @brief Specifies the number of additional TMCMC steps per chain at specified generations (this property will overwrite Default Burn In at specified generations). The first entry of the vector corresponds to the 2nd TMCMC generation.
  */
   std::vector<size_t> _perGenerationBurnIn;
  /**
  * @brief Target coefficient of variation of the plausibility weights to update the annealing exponent :math:`\rho` (by default, this value is 1.0 as suggested in [Ching2007]).
  */
   double _targetCoefficientOfVariation;
  /**
  * @brief Scaling factor :math:`\beta^2` of Covariance Matrix (by default, this value is 0.04 as suggested in [Ching2007]).
  */
   double _covarianceScaling;
  /**
  * @brief Minimum increment of the exponent :math:`\rho`. This parameter prevents TMCMC from stalling.
  */
   double _minAnnealingExponentUpdate;
  /**
  * @brief Maximum increment of the exponent :math:`\rho` (by default, this value is 1.0 (inactive)).
  */
   double _maxAnnealingExponentUpdate;
  /**
  * @brief Scaling factor of gradient and proposal distribution (only relevant for mTMCMC).
  */
   double _stepSize;
  /**
  * @brief Defines boundaries for eigenvalue adjustments of proposal distribution (only relevant for mTMCMC).
  */
   double _domainExtensionFactor;
  /**
  * @brief [Internal Use] Random number generator with a multinomial distribution.
  */
   korali::distribution::specific::Multinomial* _multinomialGenerator;
  /**
  * @brief [Internal Use] Random number generator with a multivariate normal distribution.
  */
   korali::distribution::multivariate::Normal* _multivariateGenerator;
  /**
  * @brief [Internal Use] Random number generator with a uniform distribution.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Internal Use] Actual placeholder for burn in steps per generation, calculated from Burn In Default, Burn In and Current Generation.
  */
   size_t _currentBurnIn;
  /**
  * @brief [Internal Use] Indicates that the model evaluation for the chain is pending.
  */
   std::vector<int> _chainPendingEvaluation;
  /**
  * @brief [Internal Use] Indicates that the gradient evaluation for the chain is pending (only relevant for mTMCMC).
  */
   std::vector<int> _chainPendingGradient;
  /**
  * @brief [Internal Use] All candidates of all chains to evaluate in order to advance the markov chains.
  */
   std::vector<std::vector<double>> _chainCandidates;
  /**
  * @brief [Internal Use] The loglikelihoods of the chain candidates.
  */
   std::vector<double> _chainCandidatesLogLikelihoods;
  /**
  * @brief [Internal Use] The logpriors of the chain candidates.
  */
   std::vector<double> _chainCandidatesLogPriors;
  /**
  * @brief [Internal Use] Candidate gradient of statistical model wrt. sample variables.
  */
   std::vector<std::vector<double>> _chainCandidatesGradients;
  /**
  * @brief [Internal Use] Shows if covariance calculation successfully terminated for candidate (only relevant for mTMCMC).
  */
   std::vector<int> _chainCandidatesErrors;
  /**
  * @brief [Internal Use] Candidates covariance of normal proposal distribution.
  */
   std::vector<std::vector<double>> _chainCandidatesCovariance;
  /**
  * @brief [Internal Use] Leading parameters of all chains to be accepted.
  */
   std::vector<std::vector<double>> _chainLeaders;
  /**
  * @brief [Internal Use] The loglikelihoods of the chain leaders.
  */
   std::vector<double> _chainLeadersLogLikelihoods;
  /**
  * @brief [Internal Use] The logpriors of the chain leaders.
  */
   std::vector<double> _chainLeadersLogPriors;
  /**
  * @brief [Internal Use] Leader gradient of statistical model wrt. sample variables.
  */
   std::vector<std::vector<double>> _chainLeadersGradients;
  /**
  * @brief [Internal Use] Shows if covariance calculation successfully terminated for leader (only relevant for mTMCMC).
  */
   std::vector<int> _chainLeadersErrors;
  /**
  * @brief [Internal Use] Leader covariance of normal proposal distribution.
  */
   std::vector<std::vector<double>> _chainLeadersCovariance;
  /**
  * @brief [Internal Use] Number of finished chains.
  */
   size_t _finishedChainsCount;
  /**
  * @brief [Internal Use] The current execution step for every chain.
  */
   std::vector<size_t> _currentChainStep;
  /**
  * @brief [Internal Use] Lengths for each of the chains.
  */
   std::vector<size_t> _chainLengths;
  /**
  * @brief [Internal Use] Actual coefficient of variation after :math:`\rho` has beed updated.
  */
   double _coefficientOfVariation;
  /**
  * @brief [Internal Use] Unique selections after resampling stage.
  */
   size_t _chainCount;
  /**
  * @brief [Internal Use] Exponent of the likelihood. If :math:`\rho` equals 1.0, TMCMC converged.
  */
   double _annealingExponent;
  /**
  * @brief [Internal Use] Previous Exponent of the likelihood. If :math:`\rho` equals 1.0, TMCMC converged.
  */
   double _previousAnnealingExponent;
  /**
  * @brief [Internal Use] Number of finite prior evaluations per gerneration.
  */
   size_t _numFinitePriorEvaluations;
  /**
  * @brief [Internal Use] Number of finite likelihood evaluations per gerneration.
  */
   size_t _numFiniteLikelihoodEvaluations;
  /**
  * @brief [Internal Use] Accepted candidates after proposal.
  */
   size_t _acceptedSamplesCount;
  /**
  * @brief [Internal Use] Calculated logEvidence of the model.
  */
   double _logEvidence;
  /**
  * @brief [Internal Use] Acceptance rate calculated from accepted samples.
  */
   double _proposalsAcceptanceRate;
  /**
  * @brief [Internal Use] Acceptance rate calculated from unique samples (chain count) after recombination.
  */
   double _selectionAcceptanceRate;
  /**
  * @brief [Internal Use] Sample covariance of the current leaders updated at every generation.
  */
   std::vector<double> _covarianceMatrix;
  /**
  * @brief [Internal Use] Max Loglikelihood found in current generation.
  */
   double _maxLoglikelihood;
  /**
  * @brief [Internal Use] Mean of the current leaders updated at every generation.
  */
   std::vector<double> _meanTheta;
  /**
  * @brief [Internal Use] Parameters stored in the database (taken from the chain leaders).
  */
   std::vector<std::vector<double>> _sampleDatabase;
  /**
  * @brief [Internal Use] LogLikelihood Evaluation of the parameters stored in the database.
  */
   std::vector<double> _sampleLogLikelihoodDatabase;
  /**
  * @brief [Internal Use] Log priors of the samples stored in the database.
  */
   std::vector<double> _sampleLogPriorDatabase;
  /**
  * @brief [Internal Use] Gradients stored in the database (taken from the chain leaders, only mTMCMC).
  */
   std::vector<std::vector<double>> _sampleGradientDatabase;
  /**
  * @brief [Internal Use] Shows if covariance calculation successfully terminated for sample (only relevant for mTMCMC).
  */
   std::vector<int> _sampleErrorDatabase;
  /**
  * @brief [Internal Use] Gradients stored in the database (taken from the chain leaders, only mTMCMC).
  */
   std::vector<std::vector<double>> _sampleCovariancesDatabase;
  /**
  * @brief [Internal Use] Calculated upper domain boundaries (only relevant for mTMCMC).
  */
   std::vector<double> _upperExtendedBoundaries;
  /**
  * @brief [Internal Use] Calculated lower domain boundaries (only relevant for mTMCMC).
  */
   std::vector<double> _lowerExtendedBoundaries;
  /**
  * @brief [Internal Use] Number of failed LU decompositions (only relevan for mTMCMC).
  */
   size_t _numLUDecompositionFailuresProposal;
  /**
  * @brief [Internal Use] Number of failed Eigenvalue problems (only relevan for mTMCMC).
  */
   size_t _numEigenDecompositionFailuresProposal;
  /**
  * @brief [Internal Use] Number of failed FIM inversions (only relevan for mTMCMC).
  */
   size_t _numInversionFailuresProposal;
  /**
  * @brief [Internal Use] Number of Fisher information matrices with negative eigenvalues (only relevan for mTMCMC).
  */
   size_t _numNegativeDefiniteProposals;
  /**
  * @brief [Internal Use] Number of failed chol. decomp. during proposal step (only relevant for mTMCMC).
  */
   size_t _numCholeskyDecompositionFailuresProposal;
  /**
  * @brief [Internal Use] Number of covariance adaptions (only relevant for mTMCMC).
  */
   size_t _numCovarianceCorrections;
  /**
  * @brief [Termination Criteria] Determines the annealing exponent :math:`\rho` to achieve before termination. TMCMC converges if :math:`\rho` equals 1.0.
  */
   double _targetAnnealingExponent;
  
 
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
 * @brief Configures TMCMC.
 */
  void setInitialConfiguration() override;

  /**
 * @brief Main solver loop.
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

} //sampler
} //solver
} //korali


#endif // _KORALI_SOLVER_SAMPLER_TMCMC_
