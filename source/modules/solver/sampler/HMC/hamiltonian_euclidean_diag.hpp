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
  HamiltonianEuclideanDiag(const size_t stateSpaceDim) : HamiltonianEuclidean{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim);
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param normalGenerator Generator needed for momentum sampling.
  */
  HamiltonianEuclideanDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator) : HamiltonianEuclidean{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim);
    _normalGenerator = normalGenerator;
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param sample Sample object.
  * @param normalGenerator Generator needed for momentum sampling.
  */
  HamiltonianEuclideanDiag(const size_t stateSpaceDim, korali::Sample *sample, korali::distribution::univariate::Normal *normalGenerator) : HamiltonianEuclidean{stateSpaceDim, sample}
  {
    _metric.resize(stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim);
    _normalGenerator = normalGenerator;
  }

  /**
  * @brief Total energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param _k Experiment object.
  * @return Total energy.
  */
  double H(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k) override
  {
    return K(q, p) + U(q, _k);
  }

  /**
  * @brief Kinetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &q, const std::vector<double> &p) const override
  {
    double tmpScalar = 0.0;
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar += p[i] * _inverseMetric[i] * p[i];
    }

    return 0.5 * tmpScalar;
  }

  /**
  * @brief Gradient of kintetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dK(const std::vector<double> &q, const std::vector<double> &p) const override
  {
    std::vector<double> tmpVector(_stateSpaceDim, 0.0);
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpVector[i] = _inverseMetric[i] * p[i];
    }
    return tmpVector;
  }

  /**
  * @brief Generates sample of momentum.
  * @return Sample of momentum from normal distribution with covariance matrix _metric. Only variance taken into account with diagonal metric.
  */
  std::vector<double> sampleMomentum() const override
  {
    std::vector<double> result(_stateSpaceDim);

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      result[i] = std::sqrt(_metric[i]) * _normalGenerator->getRandomNumber();
    }

    return result;
  }

  /**
  * @brief Calculates inner product induces by inverse metric.
  * @param pLeft Left argument (momentum).
  * @param pRight Right argument (momentum).
  * @return pLeft.transpose * _inverseMetric * pRight.
  */
  double innerProduct(std::vector<double> pLeft, std::vector<double> pRight) const
  {
    double result = 0.0;

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      result += pLeft[i] * _inverseMetric[i] * pRight[i];
    }

    return result;
  }

  /**
  * @brief Updates diagonal Inverse Metric by using samples to approximate the Variance (diagonal of Fisher Information matrix).
  * @param samples Contains samples. One row is one sample.
  * @param positionMean Mean of samples.
  * @return Error code not needed here to set to 0.
  */
  int updateInverseMetric(const std::vector<std::vector<double>> &samples, const std::vector<double> &positionMean) override
  {
    double tmpScalar;
    size_t numSamples = samples.size();

    // calculate covariance matrix of warmup sample via Fisher Infromation
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar = 0;
      for (size_t j = 0; j < numSamples; ++j)
      {
        tmpScalar += (samples[j][i] - positionMean[i]) * (samples[j][i] - positionMean[i]);
      }
      _inverseMetric[i] = tmpScalar / (numSamples - 1);
      _metric[i] = 1.0 / _inverseMetric[i];
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