#ifndef HAMILTONIAN_DIAG_H
#define HAMILTONIAN_DIAG_H

#include "hamiltonian_base.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class HamiltonianDiag
* @brief Used for calculating energies with unit euclidean metric.
*/
class HamiltonianDiag : public Hamiltonian
{
  public:
  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianDiag(const size_t stateSpaceDim) : Hamiltonian{stateSpaceDim} {}

  /**
  * @brief Total energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param modelEvaluationCount Number of model evuations musst be increased.
  * @param numSamples Needed for Sample ID.
  * @param inverseMetric Inverse Metric must be provided.
  * @return Total energy.
  */
  double H(const std::vector<double> &q, const std::vector<double> &p, size_t &modelEvaluationCount, const size_t &numSamples, const std::vector<double> &inverseMetric) override
  {
    return K(q, p, inverseMetric) + U(q, modelEvaluationCount, numSamples);
  }

  /**
  * @brief Kinetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param inverseMetric Inverse Metric must be provided.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &q, const std::vector<double> &p, const std::vector<double> &inverseMetric) const override
  {
    double tmpScalar = 0.0;
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar += p[i] * inverseMetric[i] * p[i];
    }

    // std::cout << "K(" << p[0] << ") = " << 0.5 * tmpScalar << std::endl;
    return 0.5 * tmpScalar;
  }

  /**
  * @brief Gradient of kintetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param inverseMetric Inverse Metric must be provided.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dK(const std::vector<double> &q, const std::vector<double> &p, const std::vector<double> &inverseMetric) const override
  {
    std::vector<double> tmpVector(_stateSpaceDim, 0.0);
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpVector[i] = inverseMetric[i] * p[i];
    }
    return tmpVector;
  }

  /**
  * @brief Updates diagonal Inverse Metric by using samples to approximate the Variance (diagonal of Fisher Information matrix).
  * @param samples Contains samples. One row is one sample.
  * @param positionMean Mean of samples.
  * @param inverseMetric Inverse Metric to be approximated.
  * @param metric Metric which is calculated from inverMetric.
  * @return Error code not needed here to set to 0.
  */
  int updateInverseMetric(const std::vector<std::vector<double>> &samples, const std::vector<double> &positionMean, std::vector<double> &inverseMetric, std::vector<double> &metric) override
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
      inverseMetric[i] = tmpScalar / (numSamples - 1);
      metric[i] = 1.0 / inverseMetric[i];
    }

    return 0;
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif