#ifndef HAMILTONIAN_EUCLIDEAN_BASE_H
#define HAMILTONIAN_EUCLIDEAN_BASE_H

#include "hamiltonian_base.hpp"
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
* \class HamiltonianEuclidean
* @brief Abstract base class for Euclidean Hamiltonian objects.
*/
class HamiltonianEuclidean : public Hamiltonian
{
  public:
  /**
  * @brief Default constructor.
  */
  HamiltonianEuclidean() = default;

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianEuclidean(const size_t stateSpaceDim, korali::Experiment *k) : Hamiltonian{stateSpaceDim, k} {}

  /**
  * @brief Destructor of abstract base class.
  */
  virtual ~HamiltonianEuclidean()
  {
  }

  /**
  * @brief Purely virtual function tau(q, p) = 0.5 * p^T * inverseMetric(q) * p (no logDetMetric term)
  * @param momentum Current momentum.
  * @param inverseMetric Current inverse metric.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double tau(const std::vector<double> &momentum, const std::vector<double> &inverseMetric) override
  {
    return this->K(momentum, inverseMetric);
  }

  /**
  * @brief Purely virtual gradient of dtau_dq(q, p) = 0.5 * p^T * dinverseMetric_dq(q) * p used for Hamiltonian Dynamics.
  * @param momentum Current momentum.
  * @param inverseMetric Current inverseMetric.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dq(const std::vector<double> &momentum, const std::vector<double> &inverseMetric) override
  {
    return std::vector<double>(_stateSpaceDim, 0.0);
  }

  /**
  * @brief Purely virtual gradient of dtau_dp(q, p) = inverseMetric(q) * p used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dp(const std::vector<double> &p, const std::vector<double>& inverseMetric) override
  {
    return this->dK(momentum, inverseMetric);
  }

  /**
  * @brief Purely virtual gradient of phi(q) = 0.5 * logDetMetric(q) + U(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double phi() override
  {
    return this->U();
  }

  /**
  * @brief Purely virtual gradient of dphi_dq(q) = 0.5 * dlogDetMetric_dq(q) + dU(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dphi_dq() override
  {
    return this->dU();
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
