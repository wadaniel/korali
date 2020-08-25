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
  public:
  /**
  * @brief Abstract base class for Leapfrog integration.
  * @param q Position which is evolved.
  * @param p Momentum which is evolved.
  * @param stepSize Step Size used for Leap Frog Scheme.
  * @param hamiltonian Hamiltonian object to calulcate energies.
  * @param _k Experiment object.
  */
  virtual void step(std::vector<double> &q, std::vector<double> &p, const double stepSize, Hamiltonian *hamiltonian, korali::Experiment *_k) = 0;
};

} // namespace sampler
} // namespace solver
} // namespace korali
#endif