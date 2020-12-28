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
* @brief Euclidean helper class for long argument list of buildTree
*/
struct TreeHelperEuclidean : public TreeHelper
{
  /**
    * @brief Computes No U-Turn Sampling (NUTS) criterion
    * @param hamiltonian Hamiltonian object of system
    * @return Returns of tree should be built further.
    */
  bool computeCriterion(Hamiltonian &hamiltonian) override
  {
    return hamiltonian.computeStandardCriterion(qLeftOut, pLeftOut, qRightOut, pRightOut);
  }

  /**
  * @brief Computes No U-Turn Sampling (NUTS) criterion
  * @param hamiltonian Hamiltonian object of system
  * @param pStart Starting momentum of trajectory
  * @param pEnd Ending momentum of trajsectory
  * @param rho Sum of momenta encountered in trajectory
  * @return Returns of tree should be built further.
  */
  bool computeCriterion(Hamiltonian &hamiltonian, const std::vector<double> pStart, const std::vector<double> pEnd, std::vector<double> rho) override
  {
    std::cout << "wrong termination criterion used in NUTS" << std::endl;
    return false;
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
