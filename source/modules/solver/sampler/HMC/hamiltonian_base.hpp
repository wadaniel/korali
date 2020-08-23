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
  * @param _k Experiment object.
  * @return Total energy.
  */
  virtual double H(const std::vector<double> &q, const std::vector<double> &p, size_t &modelEvaluationCount, const size_t &numSamples, korali::Experiment *_k) = 0;

  /**
  * @brief Purely virtual kinetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @return Kinetic energy.
  */
  virtual double K(const std::vector<double> &q, const std::vector<double> &p) const = 0;

  /**
  * @brief Purely virtual gradient of kintetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  virtual std::vector<double> dK(const std::vector<double> &q, const std::vector<double> &p) const = 0;

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
  * @brief Generates sample of momentum.
  * @return Sample of momentum from normal distribution with covariance matrix _metric.
  */
  virtual std::vector<double> sampleMomentum() const = 0;

  /**
  * @brief Calculates inner product induces by inverse metric.
  * @return pLeft.transpose * _inverseMetric * pRight.
  */
  virtual double innerProduct(std::vector<double> pLeft, std::vector<double> pRight) const = 0;

  /**
  * @brief Updates Inverse Metric by using samples to approximate the covariance matrix via the Fisher information.
  * @param samples Contains samples. One row is one sample.
  * @param positionMean Mean of samples.
  * @return Error code of Cholesky decomposition needed for dense Metric.
  */
  virtual int updateInverseMetric(const std::vector<std::vector<double>> &samples, const std::vector<double> &positionMean) = 0;

  /**
  * @brief Getter function for State Space Dim.
  * @return Dimension of state space (i.e. dim(q)).
  */
  const size_t getStateSpaceDim() const
  {
    return _stateSpaceDim;
  }

  /**
  * @brief Computes NUTS criterion on euclidean domain.
  * @param qLeft Leftmost position.
  * @param pLeft Leftmost momentum.
  * @param qRight Rightmost position.
  * @param pRight Rightmost momentum.
  * @return Returns if trees should be built further.
  */
  bool computeStandardCriterion(const std::vector<double> &qLeft, const std::vector<double> &pLeft, const std::vector<double> &qRight, const std::vector<double> &pRight)
  {
    std::vector<double> tmpVector(_stateSpaceDim, 0.0);

    std::transform(std::cbegin(qRight), std::cend(qRight), std::cbegin(qLeft), std::begin(tmpVector), std::minus<double>());
    double dotProductLeft = std::inner_product(std::cbegin(tmpVector), std::cend(tmpVector), std::cbegin(pLeft), 0.0);
    double dotProductRight = std::inner_product(std::cbegin(tmpVector), std::cend(tmpVector), std::cbegin(pRight), 0.0);

    return (dotProductLeft > 0) && (dotProductRight > 0);
  }

  /**
  * @brief Setter function for metric.
  * @param metric Metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setMetric(std::vector<double> &metric)
  {
    if (metric.size() != _metric.size())
    {
      return false;
    }
    else
    {
      _metric = metric;
      return true;
    }
  }

  /**
  * @brief Setter function for inverse metric.
  * @param inverseMetric Inverse metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setInverseMetric(std::vector<double> &inverseMetric)
  {
    if (inverseMetric.size() != _inverseMetric.size())
    {
      return false;
    }
    else
    {
      _inverseMetric = inverseMetric;
      return true;
    }
  }

  /**
  * @brief Getter function for metric.
  * @return Returns metric of hamiltonian.
  */
  std::vector<double> getMetric() const
  {
    return _metric;
  }

  /**
  * @brief Getter function for inverse metric.
  * @return Returns inverse metric of hamiltonian.
  */
  std::vector<double> getInverseMetric() const
  {
    return _inverseMetric;
  }

  protected:
  /**
  * @brief _stateSpaceDim
State Space Dimension needed for Leapfrog integrator.
  */
  size_t _stateSpaceDim;

  /**
  * @brief _inverseMetric
  */
  std::vector<double> _metric;

  /**
  * @brief _inverseMetric
  */
  std::vector<double> _inverseMetric;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif