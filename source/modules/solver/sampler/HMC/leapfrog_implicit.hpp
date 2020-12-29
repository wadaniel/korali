#ifndef LEAPFROG_IMPLICIT_H
#define LEAPFROG_IMPLICIT_H

#include "engine.hpp"
#include "hamiltonian_base.hpp"
#include "hamiltonian_riemannian_base.hpp"
#include "leapfrog_base.hpp"

#include <limits>

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class LeapfrogImplicit
* @brief Used for time propagation according to hamiltonian dynamics via implicit Leapfrog integration.
*/
class LeapfrogImplicit : public Leapfrog
{
  private:
  /**
  * @brief Maximum fixed point iterations during each step.
  */
  size_t _maxNumFixedPointIter;

  public:
  /**
  * @brief Constructor for implicit leapfrog stepper.
  * @param maxNumFixedPointIter Maximum fixed point iterations.
  */
  LeapfrogImplicit(size_t maxNumFixedPointIter) : _maxNumFixedPointIter(maxNumFixedPointIter){};
  /**
  * @brief Implicit Leapfrog stepping scheme used for evolving Hamiltonian Dynamics.
  * @param q Position which is evolved.
  * @param p Momentum which is evolved.
  * @param stepSize Step Size used for Leap Frog Scheme.
  * @param hamiltonian Hamiltonian object to calulcate energies.
  * @param _k Experiment object.
  */
  void step(std::vector<double> &q, std::vector<double> &p, const double stepSize, Hamiltonian &hamiltonian, korali::Experiment *_k) override
  {
    size_t stateSpaceDim = hamiltonian.getStateSpaceDim();
    double delta = 1e-6 * stepSize;

    // half step of momentum
    hamiltonian.updateHamiltonian(q, _k);
    std::vector<double> dphi_dq = hamiltonian.dphi_dq();
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dphi_dq[i];
    }

    std::vector<double> rho = p;
    std::vector<double> pPrime(stateSpaceDim);
    double deltaP;

    size_t numIter = 0;
    do
    {
      deltaP = 0.0;
      hamiltonian.updateHamiltonian(q, _k);
      std::vector<double> dtau_dq = hamiltonian.dtau_dq(p);
      for (size_t i = 0; i < stateSpaceDim; ++i)
      {
        pPrime[i] = rho[i] - stepSize / 2.0 * dtau_dq[i];
      }

      // find max delta
      for (size_t i = 0; i < stateSpaceDim; ++i)
      {
        if (std::abs(p[i] - pPrime[i]) > deltaP)
        {
          deltaP = std::abs(p[i] - pPrime[i]);
        }
      }

      p = pPrime;
      ++numIter;

    } while (deltaP > delta && numIter < _maxNumFixedPointIter);
    if (_maxNumFixedPointIter <= numIter)
      _k->_logger->logInfo("Detailed", "Maximum number (%zu) of fixed point iterations reached during implicit leapfrog scheme.\n", _maxNumFixedPointIter);

    std::vector<double> qPrime(stateSpaceDim);
    std::vector<double> sigma = q;
    double deltaQ;

    numIter = 0;
    do
    {
      deltaQ = 0.0;
      hamiltonian.updateHamiltonian(sigma, _k);
      std::vector<double> dtau_dp_sigma = hamiltonian.dtau_dp(p);
      hamiltonian.updateHamiltonian(q, _k);
      std::vector<double> dtau_dp_q = hamiltonian.dtau_dp(p);
      for (size_t i = 0; i < stateSpaceDim; ++i)
      {
        qPrime[i] = sigma[i] + stepSize / 2.0 * dtau_dp_sigma[i] + stepSize / 2.0 * dtau_dp_q[i];
      }

      // find max delta
      for (size_t i = 0; i < stateSpaceDim; ++i)
      {
        if (std::abs(q[i] - qPrime[i]) > deltaQ)
        {
          deltaQ = std::abs(q[i] - qPrime[i]);
        }
      }

      q = qPrime;
      ++numIter;
    } while (deltaQ > delta && numIter < _maxNumFixedPointIter);

    if (_maxNumFixedPointIter <= numIter)
      _k->_logger->logInfo("Detailed", "Maximum number (%zu) of fixed point iterations reached during implicit leapfrog scheme.\n", _maxNumFixedPointIter);

    hamiltonian.updateHamiltonian(q, _k);
    std::vector<double> dtau_dq = hamiltonian.dtau_dq(p);
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dtau_dq[i];
    }

    dphi_dq = hamiltonian.dphi_dq();
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dphi_dq[i];
    }
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
