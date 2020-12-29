#ifndef HAMILTONIAN_RIEMANNIAN_CONST_BASE_H
#define HAMILTONIAN_RIEMANNIAN_CONST_BASE_H

#include "hamiltonian_riemannian_base.hpp"
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
* \class HamiltonianRiemannianConst
* @brief Abstract base class for Hamiltonian objects.
*/
class HamiltonianRiemannianConst : public HamiltonianRiemannian
{
  public:
  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CONSTRUCTORS START /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianRiemannianConst(const size_t stateSpaceDim, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k} {}

  /**
  * @brief Destructor of abstract base class.
  */
  virtual ~HamiltonianRiemannianConst()
  {
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// CONSTRUCTORS END //////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
