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
  LeapfrogExplicit(std::shared_ptr<Hamiltonian> hamiltonian) : Leapfrog(hamiltonian){};

  /**
  * @brief Explicit Leapfrog stepping scheme used for evolving Hamiltonian Dynamics.
  * @param q Position which is evolved.
  * @param p Momentum which is evolved.
  * @param stepSize Step Size used for Leap Frog Scheme.
  */
  void step(std::vector<double> &q, std::vector<double> &p, const double stepSize) override
  {
    _hamiltonian->updateHamiltonian(q);
    std::vector<double> dU = _hamiltonian->dU();

    for (size_t i = 0; i < dU.size(); ++i)
    {
      p[i] -= 0.5 * stepSize * dU[i];
    }

    // would need to update in Riemannian case
    std::vector<double> dK = _hamiltonian->dK(p);

    for (size_t i = 0; i < dK.size(); ++i)
    {
      q[i] += stepSize * dK[i];
    }

    _hamiltonian->updateHamiltonian(q);
    dU = _hamiltonian->dU();

    for (size_t i = 0; i < dU.size(); ++i)
    {
      p[i] -= 0.5 * stepSize * dU[i];
    }
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
