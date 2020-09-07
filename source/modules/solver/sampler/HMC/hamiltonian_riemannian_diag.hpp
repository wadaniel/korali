#ifndef HAMILTONIAN_RIEMANNIAN_DIAG_H
#define HAMILTONIAN_RIEMANNIAN_DIAG_H

#include "hamiltonian_riemannian_base.hpp"
#include "modules/distribution/univariate/normal/normal.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class HamiltonianRiemannianDiag
* @brief Used for diagonal Riemannian metric.
*/
class HamiltonianRiemannianDiag : public HamiltonianRiemannian
{
  public:
  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CONSTRUCTORS START /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianRiemannianDiag(const size_t stateSpaceDim) : HamiltonianRiemannian{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim);
    _inverseRegularizationParam = 1.0;
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param normalGenerator Generator needed for momentum sampling.
  */
  HamiltonianRiemannianDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator) : HamiltonianRiemannian{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim);
    _normalGenerator = normalGenerator;
    _inverseRegularizationParam = 1.0;
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param normalGenerator Generator needed for momentum sampling.
  * @param inverseRegularizationParam Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values _inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  HamiltonianRiemannianDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator, const double inverseRegularizationParam) : HamiltonianRiemannian{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim);
    _normalGenerator = normalGenerator;
    _inverseRegularizationParam = inverseRegularizationParam;
  }

  /**
  * @brief Destructor of derived class.
  */
  ~HamiltonianRiemannianDiag()
  {
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// CONSTRUCTORS END //////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// ENERGY FUNCTIONS START ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Total energy function used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Total energy.
  */
  double H(const std::vector<double> &p) override
  {
    return this->K(p) + this->U();
  }

  /**
  * @brief Purely virtual kinetic energy function K(q, p) = 0.5 * p.T * inverseMetric(q) * p + 0.5 * logDetMetric(q) used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &p) override
  {
    double result = tau(p) + 0.5 * _logDetMetric;
    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::K :" << std::endl;
      // printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      std::cout << "K(p) = " << result << std::endl;
    }

    return result;
  }

  /**
  * @brief Purely virtual gradient of kintetic energy function dK(q, p) = inverseMetric(q) * p + 0.5 * dlogDetMetric_dq(q) used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dK(const std::vector<double> &p) override
  {
    std::vector<double> tmpVector(_stateSpaceDim, 0.0);
    for (int i = 0; i < _stateSpaceDim; ++i)
    {
      tmpVector[i] = _inverseMetric[i] * p[i];
    }
    return tmpVector;
  }

  /**
  * @brief Purely virtual function tau(q, p) = 0.5 * p^T * inverseMetric(q) * p (no logDetMetric term)
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double tau(const std::vector<double> &p) override
  {
    double tmpScalar = 0.0;

    // this->updateHamiltonian(q, _k);

    for (int i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar += p[i] * _inverseMetric[i] * p[i];
    }

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::tau :" << std::endl;
      // printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      std::cout << "tau(p) = " << 0.5 * tmpScalar << std::endl;
    }

    return 0.5 * tmpScalar;
  }

  /**
  * @brief Purely virtual gradient of dtau_dq(q, p) = 0.5 * p^T * dinverseMetric_dq(q) * p used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dq(const std::vector<double> &p) override
  {
    std::vector<double> result(_stateSpaceDim, 0.0);
    std::vector<double> gradU = this->dU();
    std::vector<double> hessianU = this->__hessianU();

    for (size_t j = 0; j < _stateSpaceDim; ++j)
    {
      result[j] = 0.0;
      for (size_t i = 0; i < _stateSpaceDim; ++i)
      {
        double arg = _inverseRegularizationParam * (gradU[i] * gradU[i]);
        result[j] += hessianU[i * _stateSpaceDim + j] * this->__taylorSeriesTauFunc(gradU[i], _inverseRegularizationParam) * p[i] * p[i];
      }
    }

    if (verbosity == true)
    {
      std::cout << "dtau_dq(p) = ";
      __printVec(result);
      std::cout << "with p = " << std::endl;
      __printVec(p);
    }

    return result;
  }

  /**
  * @brief Purely virtual gradient of dtau_dp(q, p) = inverseMetric(q) * p used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dp(const std::vector<double> &p) override
  {
    std::vector<double> result = this->dK(p);
    if (verbosity == true)
    {
      std::cout << "dtau_dp(p) = ";
      __printVec(result);
      std::cout << "with p = " << std::endl;
      __printVec(p);
    }

    return this->dK(p);
  }

  /**
  * @brief Purely virtual gradient of phi(q) = 0.5 * logDetMetric(q) + U(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double phi() override
  {
    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::phi :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
    }

    return this->U() + 0.5 * _logDetMetric;
  }

  /**
  * @brief Purely virtual gradient of dphi_dq(q) = 0.5 * dlogDetMetric_dq(q) + dU(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dphi_dq() override
  {
    std::vector<double> result(_stateSpaceDim, 0.0);
    std::vector<double> gradU = this->dU();
    std::vector<double> hessianU = this->__hessianU();

    std::vector<double> dLogDetMetric_dq(_stateSpaceDim, 0.0);

    for (size_t j = 0; j < _stateSpaceDim; ++j)
    {
      dLogDetMetric_dq[j] = 0.0;
      for (size_t i = 0; i < _stateSpaceDim; ++i)
      {
        double arg = _inverseRegularizationParam * (gradU[i] * gradU[i]);
        // dLogDetMetric_dq[j] += 1.0 / _metric[i] * ( 2.0 * hessianU[i*_stateSpaceDim + j] * gradU[i] * 1.0 / std::tanh(arg) - 2.0 * _inverseRegularizationParam * hessianU[i*_stateSpaceDim + j] * gradU[i] / (std::sinh(arg) * std::sinh(arg)) );
        dLogDetMetric_dq[j] += 2.0 * hessianU[i * _stateSpaceDim + j] * this->__taylorSeriesPhiFunc(gradU[i], _inverseRegularizationParam);
      }
    }

    for (size_t j = 0; j < _stateSpaceDim; ++j)
    {
      result[j] = gradU[j] + 0.5 * dLogDetMetric_dq[j];
    }

    if (verbosity == true)
    {
      std::cout << "dphi_dq() = ";
      __printVec(result);
      std::cout << "with gradU = " << std::endl;
      __printVec(gradU);
      std::cout << "with hessianU = " << std::endl;
      __printVec(hessianU);
    }

    return result;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// ENERGY FUNCTIONS END ////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// GENERAL FUNCTIONS START //////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Updates current position of hamiltonian.
  * @param q Current position.
  * @param _k Experiment object.
  */
  void updateHamiltonian(const std::vector<double> &q, korali::Experiment *_k) override
  {
    (*_sample)["Parameters"] = q;

    KORALI_START((*_sample));
    KORALI_WAIT((*_sample));

    // TODO: remove hack, evaluate Gradient only when required by the solver (D.W.)
    (*_sample)["Operation"] = "Evaluate Gradient";
    KORALI_START((*_sample));
    KORALI_WAIT((*_sample));
    (*_sample)["Operation"] = "Evaluate";

    // TODO: remove hack, evaluate Hessian only when required by the solver (D.W.)
    (*_sample)["Operation"] = "Evaluate Hessian";
    KORALI_START((*_sample));
    KORALI_WAIT((*_sample));
    (*_sample)["Operation"] = "Evaluate";

    // constant for condition number of _metric
    double detMetric = 1.0;

    auto grad = KORALI_GET(std::vector<double>, (*_sample), "grad(logP(x))");

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::updateHamiltonian after getting gradient :" << std::endl;
      std::cout << "g = " << std::endl;
      __printVec(grad);
    }

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      _metric[i] = this->__softAbsFunc(grad[i], _inverseRegularizationParam);
      _inverseMetric[i] = 1.0 / _metric[i];
      detMetric *= _metric[i];
    }
    _logDetMetric = std::log(detMetric);

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::updateHamiltonian end :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      std::cout << "_logDetMetric = " << _logDetMetric << std::endl;
      std::cout << "_metric = " << std::endl;
      __printVec(_metric);
      std::cout << "_inverseMetric = " << std::endl;
      __printVec(_inverseMetric);
    }
  }

  /**
  * @brief Generates sample of momentum.
  * @return Sample of momentum from normal distribution with covariance matrix _metric. Only variance taken into account with diagonal metric.
  */
  std::vector<double> sampleMomentum() const override
  {
    std::vector<double> result(_stateSpaceDim);

    for (int i = 0; i < _stateSpaceDim; ++i)
    {
      result[i] = std::sqrt(_metric[i]) * _normalGenerator->getRandomNumber();
    }

    return result;
  }

  /**
  * @brief Calculates inner product induces by inverse metric.
  * @param pLeft Left argument (momentum).
  * @param pRight Right argument (momentum).
  * @return pLeft.transpose * _inverseMetric * pRight.
  */
  double innerProduct(std::vector<double> pLeft, std::vector<double> pRight) const
  {
    double result = 0.0;

    for (int i = 0; i < _stateSpaceDim; ++i)
    {
      result += pLeft[i] * _inverseMetric[i] * pRight[i];
    }

    return result;
  }

  /**
  * @brief Updates diagonal Inverse Metric by using samples to approximate the Variance (diagonal of Fisher Information matrix).
  * @param samples Contains samples. One row is one sample.
  * @param positionMean Mean of samples.
  * @return Error code not needed here to set to 0.
  */
  int updateInverseMetric(const std::vector<std::vector<double>> &samples, const std::vector<double> &positionMean) override
  {
    double tmpScalar;
    size_t numSamples = samples.size();

    // calculate covariance matrix of warmup sample via Fisher Infromation
    for (int i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar = 0;
      for (int j = 0; j < numSamples; ++j)
      {
        tmpScalar += (samples[j][i] - positionMean[i]) * (samples[j][i] - positionMean[i]);
      }
      _inverseMetric[i] = tmpScalar / (numSamples - 1);
      _metric[i] = 1.0 / _inverseMetric[i];
    }

    return 0;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// GENERAL FUNCTIONS END ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values _inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  double _inverseRegularizationParam;

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// HELPERS START ///////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  private:
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
  * @brief Helper function f(x) = x^2 * coth(alpha * x^2) for SoftAbs metric.
  * @param x Point of evaluation.
  * @param alpha Hyperparameter.
  * @return function value at x.
  */
  double __softAbsFunc(const double x, const double alpha)
  {
    double result;
    if (std::abs(x) < 0.5)
    {
      double a4 = 1.0 / 3.0;
      double a8 = -1.0 / 45.0;
      result = 1.0 / alpha + a4 * std::pow(x, 4) * alpha + a8 * std::pow(x, 8) * std::pow(alpha, 3);
    }
    else
    {
      result = x * x * 1.0 / std::tanh(alpha * x * x);
    }
    return result;
  }

  /**
  * @brief Helper function f(x) = 1/x - alpha * x / (sinh(alpha * x^2) * cosh(alpha * x^2)) for SoftAbs metric.
  * @param x Point of evaluation.
  * @param alpha Hyperparameter.
  * @return function value at x.
  */
  double __taylorSeriesPhiFunc(const double x, const double alpha)
  {
    double result;

    if (std::abs(x) < 0.5)
    {
      double a3 = 2.0 / 3.0;
      double a7 = -14.0 / 45.0;
      double a11 = 124.0 / 945.0;

      result = a3 * std::pow(x, 3) * std::pow(alpha, 2) + a7 * std::pow(x, 7) * std::pow(alpha, 4) + a11 * std::pow(x, 11) * std::pow(alpha, 6);
    }
    else
    {
      result = 1.0 / x - alpha * x / (std::sinh(alpha * x * x) * std::cosh(alpha * x * x));
    }

    return result;
  }

  /**
  * @brief Helper function f(x) = 1/x * (alpha / cosh(alha * x^2)^2 - tanh(alpha * x^2) / x^2) for SoftAbs metric.
  * @param x Point of evaluation.
  * @param alpha Hyperparameter.
  * @return function value at x.
  */
  double __taylorSeriesTauFunc(const double x, const double alpha)
  {
    double result;

    if (std::abs(x) < 0.5)
    {
      double a3 = -2.0 / 3.0;
      double a7 = 8.0 / 15.0;
      double a11 = -34.0 / 105.0;

      result = a3 * std::pow(x, 3) * std::pow(alpha, 3) + a7 * std::pow(x, 7) * std::pow(alpha, 5) + a11 * std::pow(x, 11) * std::pow(alpha, 7);
    }
    else
    {
      result = 1.0 / x * (alpha / (std::cosh(alpha * x * x) * std::cosh(alpha * x * x)) - std::tanh(alpha * x * x) / (x * x));
    }

    return result;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// HELPERS END ////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief One dimensional normal generator needed for sampling of momentum from diagonal _metric.
  */
  korali::distribution::univariate::Normal *_normalGenerator;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
