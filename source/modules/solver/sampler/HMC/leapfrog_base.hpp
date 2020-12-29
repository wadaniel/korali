#ifndef LEAPFROG_H
#define LEAPFROG_H

#include "hamiltonian_base.hpp"
#include <iostream>
#include <vector>
namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class Leapfrog
* @brief Abstract base class used for time propagation according to hamiltonian dynamics via Leapfrog integration schemes.
*/
class Leapfrog
{
  protected:
  /**
  * @brief Pointer to hamiltonian object to calculate energies..
  */
  std::shared_ptr<Hamiltonian> _hamiltonian;

  public:
  Leapfrog(std::shared_ptr<Hamiltonian> hamiltonian) : _hamiltonian(hamiltonian){};

  /**
  * @brief Abstract base class for Leapfrog integration.
  * @param q Position which is evolved.
  * @param p Momentum which is evolved.
  * @param stepSize Step Size used for Leap Frog Scheme.
  */
  virtual void step(std::vector<double> &q, std::vector<double> &p, const double stepSize) = 0;
};

} // namespace sampler
} // namespace solver
} // namespace korali
#endif
