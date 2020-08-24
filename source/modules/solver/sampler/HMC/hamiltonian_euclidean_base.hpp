#ifndef HAMILTONIAN_EUCLIDEAN_BASE_H
#define HAMILTONIAN_EUCLIDEAN_BASE_H

#include "hamiltonian_base.hpp"
#include "modules/conduit/conduit.hpp"

#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/sampler/MCMC/MCMC.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class HamiltonianEuclidean
* @brief Abstract base class for Euclidean Hamiltonian objects.
*/
class HamiltonianEuclidean : public Hamiltonian
{
  public:
  /**
  * @brief Default constructor.
  */
  HamiltonianEuclidean() = default;

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianEuclidean(const size_t stateSpaceDim) : Hamiltonian{stateSpaceDim} {}

  /**
  * @brief Setter function for metric.
  * @param metric Metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setMetric(std::vector<double> &metric) override
  {
    if (metric.size() != _metric.size())
    {
      return false;
    }
    else
    {
      _metric = metric;
      return true;
    }
  }

  /**
  * @brief Setter function for inverse metric.
  * @param inverseMetric Inverse metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setInverseMetric(std::vector<double> &inverseMetric) override
  {
    if (inverseMetric.size() != _inverseMetric.size())
    {
      return false;
    }
    else
    {
      _inverseMetric = inverseMetric;
      return true;
    }
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif