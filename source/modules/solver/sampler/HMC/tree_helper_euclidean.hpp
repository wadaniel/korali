#ifndef TREE_HELPER_EUCLIDEAN_H
#define TREE_HELPER_EUCLIDEAN_H

#include "tree_helper_base.hpp"
namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \struct TreeHelperEuclidean
* @brief Euclidean child of helper class for long argument list of buildTree
*/
struct TreeHelperEuclidean : public TreeHelper
{
  /**
    * @brief Computes No U-Turn Sampling (NUTS) criterion
    * @param hamiltonian Hamiltonian object of system
    * @return Returns of tree should be built further.
    */
  bool computeCriterion(Hamiltonian *hamiltonian) override
  {
    return hamiltonian->computeStandardCriterion(qLeftOut, pLeftOut, qRightOut, pRightOut);
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif