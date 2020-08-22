#ifndef TREE_HELPER_RIEMANNIAN_H
#define TREE_HELPER_RIEMANNIAN_H

#include "tree_helper_base.hpp"
namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \struct TreeHelperRiemannian
* @brief Riemmanian child of helper class for long argument list of buildTree
*/
struct TreeHelperRiemannian : public TreeHelper
{
  /**
    * @brief Leftmost rho.
    */
  std::vector<double> rhoLeft;
  /**
    * @brief Rightmost rho.
    */
  std::vector<double> rhoRight;

  /**
    * @brief Leftmost p.
    */
  std::vector<double> pLeftLeftOut;

  /**
    * @brief Middle-left p.
    */
  std::vector<double> pLeftRightOut;

  /**
    * @brief Middle-right p.
    */
  std::vector<double> pRightLeftOut;

  /**
    * @brief Rightmost p.
    */
  std::vector<double> pRightRightOut;

  /**
    * @brief Leftmost pSharp.
    */
  std::vector<double> pSharpLeftLeftOut;

  /**
    * @brief Middle-left pSharp.
    */
  std::vector<double> pSharpLeftRightOut;

  /**
    * @brief Middle-right pSharp.
    */
  std::vector<double> pSharpRightLeftOut;

  /**
    * @brief Rightmost pSharp.
    */
  std::vector<double> pSharpRightRightOut;

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