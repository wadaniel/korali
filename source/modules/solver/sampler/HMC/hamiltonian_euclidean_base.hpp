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
  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CONSTRUCTORS START /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Default constructor.
  */
  HamiltonianEuclidean() = default;

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianEuclidean(const size_t stateSpaceDim) : Hamiltonian{stateSpaceDim} {}

  //////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// CONSTRUCTORS END //////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// ENERGY FUNCTIONS START ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Purely virtual function tau(q, p) = 0.5 * p^T * inverseMetric(q) * p (no logDetMetric term)
  * @param q Current position.
  * @param p Current momentum.
  * @param _k Experiment object.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double tau(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k = 0) override
  {
    return this->K(q, p, _k);
  }

  /**
  * @brief Purely virtual gradient of dtau_dq(q, p) = 0.5 * p^T * dinverseMetric_dq(q) * p used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param _k Experiment object.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dq(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k = 0) override
  {
    return std::vector<double>(_stateSpaceDim, 0.0);
  }

  /**
  * @brief Purely virtual gradient of dtau_dp(q, p) = inverseMetric(q) * p used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param _k Experiment object.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dp(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k = 0) override
  {
    return this->dK(q, p, _k);
  }

  /**
  * @brief Purely virtual gradient of phi(q) = 0.5 * logDetMetric(q) + U(q) used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param _k Experiment object.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double phi(const std::vector<double> q, korali::Experiment *_k) override
  {
    return this->U(q, _k);
  }

  /**
  * @brief Purely virtual gradient of dphi_dq(q) = 0.5 * dlogDetMetric_dq(q) + dU(q) used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param _k Experiment object.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dphi_dq(const std::vector<double> q, korali::Experiment *_k) override
  {
    return this->dU(q, _k);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// ENERGY FUNCTIONS END ////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif