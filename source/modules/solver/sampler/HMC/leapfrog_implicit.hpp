#ifndef LEAPFROG_IMPLICIT_H
#define LEAPFROG_IMPLICIT_H

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
  public:
  /**
  * @brief Implicit Leapfrog stepping scheme used for evolving Hamiltonian Dynamics.
  * @param q Position which is evolved.
  * @param p Momentum which is evolved.
  * @param stepSize Step Size used for Leap Frog Scheme.
  * @param hamiltonian Hamiltonian object to calulcate energies.
  * @param _k Experiment object.
  */
  void step(std::vector<double> &q, std::vector<double> &p, const double stepSize, Hamiltonian *hamiltonian, korali::Experiment *_k) override
  {
    if (verbosity == true)
    {
      std::cout << "------------START OF LeapfrogImplicit Step-------------" << std::endl;
    }

    size_t maxNumFixedPointIter = 10;

    size_t stateSpaceDim = hamiltonian->getStateSpaceDim();
    double delta = 1e-8;

    // half step of momentum
    hamiltonian->updateHamiltonian(q, _k);
    std::vector<double> dphi_dq = hamiltonian->dphi_dq();
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dphi_dq[i];
    }

    if (verbosity == true)
    {
      std::cout << "after dphi_dq first init and set p" << std::endl;
    }

    std::vector<double> rho = p;
    std::vector<double> pPrime(stateSpaceDim);
    double deltaP;

    size_t numIter = 0;
    do
    {
      deltaP = 0.0;
      hamiltonian->updateHamiltonian(q, _k);
      std::vector<double> dtau_dq = hamiltonian->dtau_dq(p);
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

      if (verbosity == true)
      {
        std::cout << "deltaP = " << deltaP << std::endl;
      }

      p = pPrime;
      ++numIter;

      if (numIter > 2)
      {
        hamiltonian->verbosity = false;
      }

    } while (deltaP > delta && numIter < maxNumFixedPointIter);

    hamiltonian->verbosity = false;

    if (verbosity == true)
    {
      std::cout << "Total number of (momentum) iterations in FPI = " << numIter << std::endl;
      std::cout << "p = " << std::endl;
      __printVec(p);
    }

    std::vector<double> qPrime(stateSpaceDim);
    std::vector<double> sigma = q;
    double deltaQ;

    numIter = 0;
    do
    {
      deltaQ = 0.0;
      hamiltonian->updateHamiltonian(sigma, _k);
      std::vector<double> dtau_dp_sigma = hamiltonian->dtau_dp(p);
      hamiltonian->updateHamiltonian(q, _k);
      std::vector<double> dtau_dp_q = hamiltonian->dtau_dp(p);
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

      if (verbosity == true)
      {
        std::cout << "deltaQ = " << deltaQ << std::endl;
      }

      q = qPrime;
      ++numIter;
    } while (deltaQ > delta && numIter < maxNumFixedPointIter);

    if (verbosity == true)
    {
      std::cout << "Total number of (position) iterations in FPI = " << numIter << std::endl;
      std::cout << "q = " << std::endl;
      __printVec(q);
    }

    hamiltonian->updateHamiltonian(q, _k);
    std::vector<double> dtau_dq = hamiltonian->dtau_dq(p);
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dtau_dq[i];
    }

    if (verbosity == true)
    {
      std::cout << "after last step dtau_dq" << std::endl;
    }

    dphi_dq = hamiltonian->dphi_dq();
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dphi_dq[i];
    }

    if (verbosity == true)
    {
      std::cout << "-------------END OF LeapfrogImplicit Step--------------" << std::endl;
    }

    return;
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif