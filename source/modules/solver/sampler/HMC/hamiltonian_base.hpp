#ifndef HAMILTONIAN_BASE_H
#define HAMILTONIAN_BASE_H

#include "modules/conduit/conduit.hpp"

#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/sampler/MCMC/MCMC.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class Hamiltonian
* @brief Abstract base class for Hamiltonian objects.
*/
class Hamiltonian
{
  public:
  /**
  * @brief Default constructor.
  */
  Hamiltonian() = default;

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  Hamiltonian(const size_t stateSpaceDim) : _stateSpaceDim{stateSpaceDim} {}

  /**
  * @brief Purely abstract total energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param modelEvaluationCount Number of model evuations musst be increased.
  * @param numSamples Needed for Sample ID.
  * @param inverseMetric Inverse Metric must be provided.
  * @param _k Experiment object.
  * @return Total energy.
  */
  virtual double H(const std::vector<double> &q, const std::vector<double> &p, size_t &modelEvaluationCount, const size_t &numSamples, const std::vector<double> &inverseMetric, korali::Experiment *_k) = 0;

  /**
  * @brief Purely virtual kinetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param inverseMetric Inverse Metric must be provided.
  * @return Kinetic energy.
  */
  virtual double K(const std::vector<double> &q, const std::vector<double> &p, const std::vector<double> &inverseMetric) const = 0;

  /**
  * @brief Purely virtual gradient of kintetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param inverseMetric Inverse Metric must be provided.
  * @return Gradient of Kinetic energy with current momentum.
  */
  virtual std::vector<double> dK(const std::vector<double> &q, const std::vector<double> &p, const std::vector<double> &inverseMetric) const = 0;

  /**
  * @brief Potential Energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param modelEvaluationCount Number of model evuations musst be increased.
  * @param numSamples Needed for Sample ID.
  * @param _k Experiment object.
  * @return Potential energy.
  */
  virtual double U(const std::vector<double> &q, size_t &modelEvaluationCount, const size_t &numSamples, korali::Experiment *_k)
  {
    // get sample
    auto sample = Sample();
    ++modelEvaluationCount;
    sample["Parameters"] = q;
    sample["Sample Id"] = numSamples;
    sample["Module"] = "Problem";
    sample["Operation"] = "Evaluate";
    // _conduit->start(sample);
    // _conduit->wait(sample);
    // evaluate logP(x)
    // double evaluation = KORALI_GET(double, sample, "logP(x)");
    KORALI_START(sample);
    KORALI_WAIT(sample);

    double evaluation = KORALI_GET(double, sample, "logP(x)");
    // negate to get U
    evaluation *= -1.0;

    return evaluation;
  }

  /**
  * @brief Gradient of Potential Energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param modelEvaluationCount Number of model evuations musst be increased.
  * @param numSamples Needed for Sample ID.
  * @param _k Experiment object.
  * @return Gradient of Potential energy.
  */
  virtual std::vector<double> dU(const std::vector<double> &q, size_t &modelEvaluationCount, const size_t &numSamples, korali::Experiment *_k)
  {
    // TODO: REMOVE THIS SAMPLING PART
    // get sample
    auto sample = Sample();
    ++modelEvaluationCount;
    sample["Parameters"] = q;
    sample["Sample Id"] = numSamples;
    sample["Module"] = "Problem";
    sample["Operation"] = "Evaluate Gradient";
    KORALI_START(sample);
    KORALI_WAIT(sample);

    // evaluate grad(logP(x)) (extremely slow)
    std::vector<double> evaluation = KORALI_GET(std::vector<double>, sample, "grad(logP(x))");

    // negate to get dU
    std::transform(evaluation.cbegin(), evaluation.cend(), evaluation.begin(), std::negate<double>());

    return evaluation;
  }

  /**
  * @brief Updates Inverse Metric by using samples to approximate the covariance matrix via the Fisher information.
  * @param samples Contains samples. One row is one sample.
  * @param positionMean Mean of samples.
  * @param inverseMetric Inverse Metric to be approximated.
  * @param metric Metric which is calculated from inverMetric.
  * @return Error code of Cholesky decomposition needed for dense Metric.
  */
  virtual int updateInverseMetric(const std::vector<std::vector<double>> &samples, const std::vector<double> &positionMean, std::vector<double> &inverseMetric, std::vector<double> &metric) = 0;

  /**
  * @brief Getter function for State Space Dim.
  * @return Dimension of state space (i.e. dim(q)).
  */
  const size_t getStateSpaceDim() const
  {
    return _stateSpaceDim;
  }

  protected:
  /**
  * @brief _stateSpaceDim
State Space Dimension needed for Leapfrog integrator.
  */
  size_t _stateSpaceDim;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif