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
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// HELPERS START ///////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Hessian of potential energy function used for Riemannian metric.
  * @return Hessian of potential energy.
  */
  std::vector<double> __hessianU()
  {
    auto hessian = KORALI_GET(std::vector<double>, (*_sample), "H(logP(x))");

    // negate to get dU
    std::transform(hessian.cbegin(), hessian.cend(), hessian.begin(), std::negate<double>());

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::__hessianU :" << std::endl;
      std::cout << "__hessianU() = " << std::endl;
      __printVec(hessian);
    }

    return hessian;
  }

  /**
  * @brief Helper function f(x) = x * coth(alpha * x) for SoftAbs metric.
  * @param x Point of evaluation.
  * @param alpha Hyperparameter.
  * @return function value at x.
  */
  double __softAbsFunc(const double x, const double alpha)
  {
    double result;
    if (std::abs(x * alpha) < 0.5)
    {
      double a2 = 1.0 / 3.0;
      double a4 = -1.0 / 45.0;
      result = 1.0 / alpha + a2 * std::pow(x, 2) * alpha + a4 * std::pow(x, 4) * std::pow(alpha, 3);
    }
    else
    {
      result = x * 1.0 / std::tanh(alpha * x);
    }
    return result;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// HELPERS END ////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief normalization constant for kinetic energy (isn't constant compared to Euclidean Hamiltonian => have to include term in calculation)
  */
  double _logDetMetric;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif