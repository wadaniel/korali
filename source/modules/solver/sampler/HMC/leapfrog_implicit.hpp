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
    std::cout << "Dont use this function" << std::endl;
    return;
  }
  
  void step(std::vector<double> &q, std::vector<double> &p, const double stepSize, HamiltonianRiemannianDiag *hamiltonian, korali::Experiment *_k)
  {
    std::cout << "start of step" << std::endl;
    hamiltonian->verbosity = true;
    size_t stateSpaceDim = hamiltonian->getStateSpaceDim();
    
    // half step of momentum
    std::vector<double> dphi_dq = hamiltonian->dphi_dq(q, _k);
    for(size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dphi_dq[i];
    }

    std::cout << "after dphi_dq first init and set p" << std::endl;

    std::vector<double> rho = p;
    std::vector<double> pPrime(stateSpaceDim);
    double delta = 0.05;
    double deltaP;
    do 
    {
      std::cout << "deltaP = " << deltaP << std::endl;
      std::vector<double> dtau_dq = hamiltonian->dtau_dq(q, p, _k);
      for(size_t i = 0; i < stateSpaceDim; ++i)
      {
        pPrime[i] = rho[i] - stepSize / 2.0 * dtau_dq[i];
      }

      // find max delta
      for(size_t i = 0; i < stateSpaceDim; ++i)
      {
        if(std::abs(p[i] - pPrime[i]) > deltaP)
        {
          deltaP = std::abs(p[i] - pPrime[i]);
        }
      }

      p = pPrime;
    } while(deltaP > delta);
    
    std::cout << "after first while (for the momentum)" << std::endl;

    std::vector<double> qPrime(stateSpaceDim);
    std::vector<double> sigma = q;
    double deltaQ = 0.0;
    do
    {
      std::vector<double> dtau_dp_sigma = hamiltonian->dtau_dp(sigma, p, _k);
      std::vector<double> dtau_dp_q = hamiltonian->dtau_dp(q, p, _k);
      for(size_t i = 0; i < stateSpaceDim; ++i)
      {
        qPrime[i] = sigma[i] + stepSize / 2.0 * dtau_dp_sigma[i] + stepSize / 2.0 * dtau_dp_q[i];
      }

      // find max delta
      for(size_t i = 0; i < stateSpaceDim; ++i)
      {
        if(std::abs(q[i] - qPrime[i]))
        {
          deltaQ = std::abs(q[i] - qPrime[i]);
        }
      }

      q = qPrime;
    } while(deltaQ > delta);
    
    std::cout << "after second while (for the position)" << std::endl;

    std::vector<double> dtau_dq = hamiltonian->dtau_dq(q, p, _k);
    for(size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dtau_dq[i];
    }
    
    std::cout << "after last step dtau_dq" << std::endl;

    dphi_dq = hamiltonian->dphi_dq(q, _k);
    for(size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dphi_dq[i];
    }
    
    std::cout << "end of step (after last step dphi_dq)" << std::endl;

    return;
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif