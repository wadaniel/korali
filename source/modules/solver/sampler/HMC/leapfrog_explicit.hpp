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
  size_t numstep = 0;

  public:
  /**
  * @brief Explicit Leapfrog stepping scheme used for evolving Hamiltonian Dynamics.
  * @param q Position which is evolved.
  * @param p Momentum which is evolved.
  * @param stepSize Step Size used for Leap Frog Scheme.
  * @param hamiltonian Hamiltonian object to calulcate energies.
  * @param _k Experiment object.
  */
  void step(std::vector<double> &q, std::vector<double> &p, const double stepSize, Hamiltonian &hamiltonian, korali::Experiment *_k) override
  {
    numstep++;

    size_t stateSpaceDim = hamiltonian.getStateSpaceDim();
    hamiltonian.updateHamiltonian(q, _k);
    std::vector<double> dU = hamiltonian.dU();

    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] -= 0.5 * stepSize * dU[i];
    }

    // would need to update in Riemannian case
    std::vector<double> dK = hamiltonian.dK(p);

    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      q[i] += stepSize * dK[i];
    }

    hamiltonian.updateHamiltonian(q, _k);
    dU = hamiltonian.dU();

    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] -= 0.5 * stepSize * dU[i];
    }
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
