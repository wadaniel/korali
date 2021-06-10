#ifndef HAMILTONIAN_RIEMANNIAN_BASE_H
#define HAMILTONIAN_RIEMANNIAN_BASE_H

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
* \class HamiltonianRiemannian
* @brief Abstract base class for Hamiltonian objects.
*/
class HamiltonianRiemannian : public Hamiltonian
{
  public:
  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param k Pointer to Korali object.
  */
  HamiltonianRiemannian(const size_t stateSpaceDim, korali::Experiment *k) : Hamiltonian{stateSpaceDim, k} { _logDetMetric = 1.0; _currentHessian.resize(stateSpaceDim * stateSpaceDim); }

  /**
  * @brief Destructor of abstract base class.
  */
  virtual ~HamiltonianRiemannian() = default;

  protected:
  /**
  * @brief Hessian of potential energy function used for Riemannian metric.
  * @return Hessian of potential energy.
  */
  std::vector<double> hessianU() const
  {
    auto hessian = _currentHessian;

    // negate to get dU
    std::transform(hessian.cbegin(), hessian.cend(), hessian.begin(), std::negate<double>());

    return hessian;
  }

};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
