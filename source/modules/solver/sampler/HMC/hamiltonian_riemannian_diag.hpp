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
  HamiltonianRiemannianDiag(const size_t stateSpaceDim, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k}
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
  HamiltonianRiemannianDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k}
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
  HamiltonianRiemannianDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator, const double inverseRegularizationParam, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k}
  {
    _metric.resize(stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim);
    _normalGenerator = normalGenerator;
    _inverseRegularizationParam = inverseRegularizationParam;
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param normalGenerator Generator needed for momentum sampling.
  * @param metric Metric for initialization. 
  * @param inverseMetric Inverse Metric for initialization. 
  * @param inverseRegularizationParam Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values _inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  HamiltonianRiemannianDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator, const std::vector<double> metric, const std::vector<double> inverseMetric, const double inverseRegularizationParam, korali::Experiment *k) : HamiltonianRiemannianDiag{stateSpaceDim, normalGenerator, k}
  {
    _metric = metric;
    _inverseMetric = inverseMetric;
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
    for (size_t i = 0; i < _stateSpaceDim; ++i)
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

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar += p[i] * _inverseMetric[i] * p[i];
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
        result[j] += hessianU[i * _stateSpaceDim + j] * this->__taylorSeriesTauFunc(gradU[i], _inverseRegularizationParam) * p[i] * p[i];
      }
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

    return this->dK(p);
  }

  /**
  * @brief Purely virtual gradient of phi(q) = 0.5 * logDetMetric(q) + U(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double phi() override
  {
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
        dLogDetMetric_dq[j] += 2.0 * hessianU[i * _stateSpaceDim + j] * this->__taylorSeriesPhiFunc(gradU[i], _inverseRegularizationParam);
      }
    }

    for (size_t j = 0; j < _stateSpaceDim; ++j)
    {
      result[j] = gradU[j] + 0.5 * dLogDetMetric_dq[j];
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
  void updateHamiltonian(const std::vector<double> &q) override
  {
    auto sample = korali::Sample();
    sample["Sample Id"] = _modelEvaluationCount;
    sample["Module"] = "Problem";
    sample["Operation"] = "Evaluate";
    sample["Parameters"] = q;

    KORALI_START(sample);
    KORALI_WAIT(sample);
    _modelEvaluationCount++;
    _currentEvaluation = KORALI_GET(double, sample, "logP(x)");

    if (samplingProblemPtr != nullptr)
    {
      samplingProblemPtr->evaluateGradient(sample);
      samplingProblemPtr->evaluateHessian(sample);
    }
    else
    {
      bayesianProblemPtr->evaluateGradient(sample);
      bayesianProblemPtr->evaluateHessian(sample);
    }

    _currentGradient = sample["grad(logP(x))"].get<std::vector<double>>();
    _currentHessian = sample["H(logP(x))"].get<std::vector<double>>();

    // constant for condition number of _metric
    _logDetMetric = 0.0;
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      _metric[i] = this->__softAbsFunc(_currentGradient[i] * _currentGradient[i], _inverseRegularizationParam);
      _inverseMetric[i] = 1.0 / _metric[i];
      _logDetMetric += std::log(_metric[i]);
    }

    return;
  }

  /**
  * @brief Generates sample of momentum.
  * @return Sample of momentum from normal distribution with covariance matrix _metric. Only variance taken into account with diagonal metric.
  */
  std::vector<double> sampleMomentum() const override
  {
    std::vector<double> result(_stateSpaceDim);

    for (size_t i = 0; i < _stateSpaceDim; ++i)
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
  double innerProduct(const std::vector<double> &pLeft, const std::vector<double> &pRight) const
  {
    double result = 0.0;

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      result += pLeft[i] * _inverseMetric[i] * pRight[i];
    }

    return result;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// GENERAL FUNCTIONS END ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values _inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  double _inverseRegularizationParam;

  private:
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// HELPERS START ///////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Helper function f(x) = 1/x - alpha * x / (sinh(alpha * x^2) * cosh(alpha * x^2)) for SoftAbs metric.
  * @param x Point of evaluation.
  * @param alpha Hyperparameter.
  * @return function value at x.
  */
  double __taylorSeriesPhiFunc(const double x, const double alpha)
  {
    double result;

    if (std::abs(x * alpha) < 0.5)
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

    if (std::abs(x * alpha) < 0.5)
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
