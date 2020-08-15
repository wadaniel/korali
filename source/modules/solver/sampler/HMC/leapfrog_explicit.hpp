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
  * @param modelEvaluationCount Needed to keep track of number of model evaluations.
  * @param numSamples Needed for ID of Sample.
  * @param inverseMetric Needed for calculation of Kinetic Energy.
  */
  void step(std::vector<double> &q, std::vector<double> &p, const double stepSize, Hamiltonian *hamiltonian, size_t &modelEvaluationCount, const size_t &numSamples, std::vector<double> &inverseMetric) override
  {
    std::vector<double> dU = hamiltonian->dU(q, modelEvaluationCount, numSamples);
    size_t stateSpaceDim = hamiltonian->getStateSpaceDim();
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      p[i] -= 0.5 * stepSize * dU[i];
    }

    std::vector<double> dK = hamiltonian->dK(q, p, inverseMetric);
    for (size_t i = 0; i < stateSpaceDim; ++i)
    {
      q[i] += stepSize * dK[i];
    }

    dU = hamiltonian->dU(q, modelEvaluationCount, numSamples);
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