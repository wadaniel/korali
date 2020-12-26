#ifndef TREE_HELPER_BASE_H
#define TREE_HELPER_BASE_H

#include <vector>

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \struct TreeHelper
* @brief Abstract helper class for long argument list of buildTree
*/
struct TreeHelper
{
  /**
    * @brief Position input.
    */
  std::vector<double> qIn;
  /**
    * @brief Momentum input.
    */
  std::vector<double> pIn;
  /**
    * @brief Log of uni sample input.
    */
  double logUniSampleIn;
  /**
    * @brief Direction in which to propagate input.
    */
  int directionIn;
  /**
    * @brief Depth of binary tree inpu.
    */
  int depthIn;

  /**
    * @brief Energy of root of binary tree (i.e. starting position) input.
    */
  double rootHIn;
  /**
    * @brief Leftmost position output.
    */
  std::vector<double> qLeftOut;
  /**
    * @brief Leftmost momentum output.
    */
  std::vector<double> pLeftOut;
  /**
    * @brief Rightmost position output.
    */
  std::vector<double> qRightOut;
  /**
    * @brief Rightmost momentum output.
    */
  std::vector<double> pRightOut;
  /**
    * @brief Proposed position output.
    */
  std::vector<double> qProposedOut;
  /**
    * @brief Number of valid leaves output (needed for acceptance probability).
    */
  double numValidLeavesOut;
  /**
    * @brief No U-Turn Termination Sampling (NUTS) criterion output.
    */
  bool buildCriterionOut;
  /**
    * @brief Acceptance probability output.
    */
  double alphaOut;
  /**
    * @brief Number of valid leaves encountererd (needed for adaptive time stepping).
    */
  int numLeavesOut;

  /**
    * @brief Computes No U-Turn Sampling (NUTS) criterion
    * @param hamiltonian Hamiltonian object of system
    * @return Returns of tree should be built further.
    */
  bool computeCriterion(Hamiltonian &hamiltonian) { return true; };

  /**
    * @brief Computes No U-Turn Sampling (NUTS) criterion
    * @param hamiltonian Hamiltonian object of system
    * @param pStart Starting momentum of trajectory
    * @param pEnd Ending momentum of trajsectory
    * @param rho Sum of momenta encountered in trajectory
    * @return Returns of tree should be built further.
    */
  bool computeCriterion(Hamiltonian &hamiltonian, const std::vector<double> pStart, const std::vector<double> pEnd, std::vector<double> rho) { return true; };
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
