#ifndef HAMILTONIAN_EUCLIDEAN_DIAG_H
#define HAMILTONIAN_EUCLIDEAN_DIAG_H

#include "hamiltonian_euclidean_base.hpp"
#include "modules/distribution/univariate/normal/normal.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class HamiltonianEuclideanDiag
* @brief Used for calculating energies with unit euclidean metric.
*/
class HamiltonianEuclideanDiag : public HamiltonianEuclidean
{
  public:
  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianEuclideanDiag(const size_t stateSpaceDim, korali::Experiment *k) : HamiltonianEuclidean{stateSpaceDim, k}
  {
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param normalGenerator Generator needed for momentum sampling.
  */
  HamiltonianEuclideanDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator, korali::Experiment *k) : HamiltonianEuclidean{stateSpaceDim, k}
  {
    _normalGenerator = normalGenerator;
  }

  /**
  * @brief Destructor of derived class.
  */
  ~HamiltonianEuclideanDiag()
  {
  }

  /**
  * @brief Total energy function used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Total energy.
  */
  double H(const std::vector<double> &momentum, const std::vector<double>& inverseMetric) override
  {
    return K(momentum, inverseMetric) + U();
  }

  /**
  * @brief Kinetic energy function K(q, p) = 0.5 * p.T * inverseMetric(q) * p + 0.5 * logDetMetric(q) used for Hamiltonian Dynamics. For Euclidean metric logDetMetric(q) := 0.0.
  * @param p Current momentum.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &momentum, const std::vector<double>& inverseMetric) override
  {
    double energy = 0.0;
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      energy += momentum[i] * inverseMetric[i] * momentum[i];
    }

    return 0.5 * energy;
  }

  /**
  * @brief Purely virtual gradient of kintetic energy function dK(q, p) = inverseMetric(q) * p + 0.5 * dlogDetMetric_dq(q) used for Hamiltonian Dynamics. For Euclidean metric logDetMetric(q) := 0.0.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dK(const std::vector<double> &momentum, const std::vector<double>& inverseMetric) override
  {
    std::vector<double> tmpVector(_stateSpaceDim, 0.0);
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpVector[i] = inverseMetric[i] * momentum[i];
    }
    return tmpVector;
  }

  /**
  * @brief Generates sample of momentum.
  * @return Sample of momentum from normal distribution with covariance matrix _metric. Only variance taken into account with diagonal metric.
  */
  std::vector<double> sampleMomentum(const std::vector<double>& metric) const override
  {
    std::vector<double> result(_stateSpaceDim);

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      result[i] = std::sqrt(metric[i]) * _normalGenerator->getRandomNumber();
    }

    return result;
  }
 
  /**
  * @brief Calculates inner product induces by inverse metric.
  * @param pLeft Left argument (momentum).
  * @param pRight Right argument (momentum).
  * @return pLeft.transpose * inverseMetric * pRight.
  */
  double innerProduct(const std::vector<double> &pLeft, const std::vector<double> &pRight, const std::vector<double>& inverseMetric) const override
  {
    double result = 0.0;

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
        result += pLeft[i] * inverseMetric[i] * pRight[i];
    }

    return result;
  }


  /**
  * @brief Updates diagonal Inverse Metric by using samples to approximate the Variance (diagonal of Fisher Information matrix).
  * @param samples Contains samples. One row is one sample.
  * @return Error code not needed here to set to 0.
  */
  int updateMetricMatricesEuclidean(const std::vector<std::vector<double>> &samples, std::vector<double>& metric, std::vector<double>& inverseMetric)
  {
    double mean, cov, sum;
    double sumOfSquares;
    double numSamples = samples.size();

    // calculate sample covariance
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      sum = 0.0;
      sumOfSquares = 0.0;
      for (size_t j = 0; j < numSamples; ++j)
      {
        sum += samples[j][i];
        sumOfSquares += samples[j][i] * samples[j][i];
      }
      mean = sum / (numSamples);
      cov = sumOfSquares / (numSamples)-mean * mean;
      inverseMetric[i] = cov;
      metric[i] = 1.0 / cov;
    }

    return 0;
  }

  private:
  /**
  * @brief One dimensional normal generator needed for sampling of momentum from diagonal _metric.
  */
  korali::distribution::univariate::Normal *_normalGenerator;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
