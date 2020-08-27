#ifndef HAMILTONIAN_RIEMANNIAN_BASE_H
#define HAMILTONIAN_RIEMANNIAN_BASE_H

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
* \class HamiltonianRiemannian
* @brief Abstract base class for Hamiltonian objects.
*/
class HamiltonianRiemannian : public Hamiltonian
{
  public:
  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CONSTRUCTORS START /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianRiemannian(const size_t stateSpaceDim) : Hamiltonian{stateSpaceDim}, _logDetMetric{1.0} {}

  /**
  * @brief Destructor of abstract base class.
  */
  virtual ~HamiltonianRiemannian()
  {
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// CONSTRUCTORS END //////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  protected:
  /**
  * @brief normalization constant for kinetic energy (isn't constant compared to Euclidean Hamiltonian => have to include term in calculation)
  */
  double _logDetMetric;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif