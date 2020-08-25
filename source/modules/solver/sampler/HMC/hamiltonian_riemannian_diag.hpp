#ifndef HAMILTONIAN_RIEMANNIAN_DIAG_H
#define HAMILTONIAN_RIEMANNIAN_DIAG_H

#include "hamiltonian_riemannian_base.hpp"
#include "modules/distribution/univariate/normal/normal.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class HamiltonianRiemannianDiag
* @brief Used for diagonal Riemannian metric.
*/
class HamiltonianRiemannianDiag : public HamiltonianRiemannian
{
  public:
  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianRiemannianDiag(const size_t stateSpaceDim) : HamiltonianRiemannian{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim);
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param normalGenerator Generator needed for momentum sampling.
  */
  HamiltonianRiemannianDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator) : HamiltonianRiemannian{stateSpaceDim}
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
    // updateHamiltonian(q);
    for (int i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar += p[i] * _inverseMetric[i] * p[i];
    }

    return 0.5 * (tmpScalar + _logDetMetric);
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
    for (int i = 0; i < _stateSpaceDim; ++i)
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

    for (int i = 0; i < _stateSpaceDim; ++i)
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

    for (int i = 0; i < _stateSpaceDim; ++i)
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
    for (int i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar = 0;
      for (int j = 0; j < numSamples; ++j)
      {
        tmpScalar += (samples[j][i] - positionMean[i]) * (samples[j][i] - positionMean[i]);
      }
      _inverseMetric[i] = tmpScalar / (numSamples - 1);
      _metric[i] = 1.0 / _inverseMetric[i];
    }

    return 0;
  }

  /**
  * @brief Hessian of Potential Energy function used for Riemannian metric.
  * @param q Current position.
  * @param _k Experiment object.
  * @return Gradient of Potential energy.
  */
  std::vector<double> hessianU(const std::vector<double> &q, korali::Experiment *_k)
  {
    updateHamiltonian(q, _k);

    ++_numHamiltonianObjectUpdates;

    // evaluate grad(logP(x)) (extremely slow)
    std::vector<std::vector<double>> evaluationMat = KORALI_GET(std::vector<std::vector<double>>, (*_sample), "H(logP(x))");
    std::vector<double> evaluation(_stateSpaceDim * _stateSpaceDim);
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        evaluation[i * _stateSpaceDim + j] = evaluationMat[i][j];
      }
    }

    // negate to get dU
    std::transform(evaluation.cbegin(), evaluation.cend(), evaluation.begin(), std::negate<double>());

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::hessianU :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    return evaluation;
  }

  // TODO: Remove
  /**
  * @brief Setter function for metric.
  * @param metric Metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setMetric(std::vector<double> &metric) override { return false; };

  // TODO: Remove
  /**
  * @brief Setter function for inverse metric.
  * @param inverseMetric Inverse metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setInverseMetric(std::vector<double> &inverseMetric) override { return false; };

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