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
* @brief Riemmanian helper class for long argument list of buildTree
*/
struct TreeHelperRiemannian : public TreeHelper
{
  /**
    * @brief Computes No U-Turn Sampling (NUTS) criterion
    * @param hamiltonian Hamiltonian object of system
    * @return Returns of tree should be built further.
    */
  bool computeCriterion(const Hamiltonian &hamiltonian) const override
  {
    std::cout << "wrong termination criterion used in NUTS" << std::endl;
    return false;
  }

  /**
    * @brief Computes No U-Turn Sampling (NUTS) criterion
    * @param hamiltonian Hamiltonian object of system
    * @param pStart Starting momentum of trajectory
    * @param pEnd Ending momentum of trajsectory
    * @param rho Sum of momenta encountered in trajectory
    * @return Returns of tree should be built further.
    */
  bool computeCriterion(const Hamiltonian &hamiltonian, const std::vector<double> &pStart, const std::vector<double> &pEnd, const std::vector<double> &rho) const override
  {
    size_t dim = rho.size();
    std::vector<double> tmpVectorOne(dim, 0.0);
    std::transform(std::cbegin(rho), std::cend(rho), std::cbegin(pStart), std::begin(tmpVectorOne), std::minus<double>());

    std::vector<double> tmpVectorTwo(dim, 0.0);
    std::transform(std::cbegin(rho), std::cend(rho), std::cbegin(pStart), std::begin(tmpVectorTwo), std::minus<double>());

    return hamiltonian.innerProduct(pStart, tmpVectorOne) > 0.0 && hamiltonian.innerProduct(pEnd, tmpVectorTwo) > 0.0;
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
