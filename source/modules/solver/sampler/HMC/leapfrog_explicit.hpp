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
    if (verbosity == true)
    {
      std::cout << "-------------START OF LeapfrogExplicit Step--------------" << std::endl;
    }

    size_t stateSpaceDim = hamiltonian->getStateSpaceDim();
    std::vector<double> dU = hamiltonian->dU(q, _k);

    if (verbosity == true)
    {
      std::cout << "dU(q) = " << std::endl;
      __printVec(dU);
    }

    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] -= 0.5 * stepSize * dU[i];
    }

    if (verbosity == true)
    {
      std::cout << "p = " << std::endl;
      __printVec(p);
    }

    std::vector<double> dK = hamiltonian->dK(q, p, _k);

    if (verbosity == true)
    {
      std::cout << "dK(q, p) = " << std::endl;
      __printVec(dK);
    }

    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      q[i] += stepSize * dK[i];
    }

    if (verbosity == true)
    {
      std::cout << "q = " << std::endl;
      __printVec(q);
    }

    dU = hamiltonian->dU(q, _k);

    if (verbosity == true)
    {
      std::cout << "dU(q) = " << std::endl;
      __printVec(dU);
    }

    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] -= 0.5 * stepSize * dU[i];
    }

    if (verbosity == true)
    {
      std::cout << "p = " << std::endl;
      __printVec(p);
    }

    if (verbosity == true)
    {
      std::cout << "-------------END OF LeapfrogExplicit Step--------------" << std::endl;
    }

    return;
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif