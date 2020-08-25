#ifndef LEAPFROG_EXPLICIT_H
#define LEAPFROG_EXPLICIT_H

#include "hamiltonian_base.hpp"
#include "leapfrog_base.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class LeapfrogExplicit
* @brief Used for time propagation according to hamiltonian dynamics via explicit Leapfrog integration.
*/
class LeapfrogExplicit : public Leapfrog
{
  public:
  /**
  * @brief Explicit Leapfrog stepping scheme used for evolving Hamiltonian Dynamics.
  * @param q Position which is evolved.
  * @param p Momentum which is evolved.
  * @param stepSize Step Size used for Leap Frog Scheme.
  * @param hamiltonian Hamiltonian object to calulcate energies.
  * @param _k Experiment object.
  */
  void step(std::vector<double> &q, std::vector<double> &p, const double stepSize, Hamiltonian *hamiltonian, korali::Experiment *_k) override
  {
    size_t stateSpaceDim = hamiltonian->getStateSpaceDim();
    std::vector<double> dU = hamiltonian->dU(q, _k);
    // std::cout << "dU[0] = " << dU[0] << std::endl;
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] -= 0.5 * stepSize * dU[i];
    }

    std::vector<double> dK = hamiltonian->dK(q, p);
    // std::cout << "dK[0] = " << dK[0] << std::endl;
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      q[i] += stepSize * dK[i];
    }

    dU = hamiltonian->dU(q, _k);
    // std::cout << "dU[0] = " << dU[0] << std::endl;
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] -= 0.5 * stepSize * dU[i];
    }

    return;
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif