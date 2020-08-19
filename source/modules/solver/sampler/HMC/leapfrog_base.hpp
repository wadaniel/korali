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
  * @param modelEvaluationCount Needed to keep track of number of model evaluations.
  * @param numSamples Needed for ID of Sample.
  * @param inverseMetric Needed for calculation of Kinetic Energy.
  * @param _k Experiment object.
  */
  virtual void step(std::vector<double> &q, std::vector<double> &p, const double stepSize, Hamiltonian *hamiltonian, size_t &modelEvaluationCount, const size_t &numSamples, std::vector<double> &inverseMetric, korali::Experiment *_k) = 0;
};

} // namespace sampler
} // namespace solver
} // namespace korali
#endif