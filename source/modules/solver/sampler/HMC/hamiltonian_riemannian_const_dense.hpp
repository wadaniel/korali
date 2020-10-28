#ifndef HAMILTONIAN_RIEMANNIAN_CONST_DENSE_H
#define HAMILTONIAN_RIEMANNIAN_CONST_DENSE_H

#include "hamiltonian_riemannian_const_base.hpp"
#include "modules/distribution/multivariate/normal/normal.hpp"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class HamiltonianRiemannianConstDense
* @brief Used for dense Riemannian metric.
*/
class HamiltonianRiemannianConstDense : public HamiltonianRiemannianConst
{
  public:
  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CONSTRUCTORS START /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianRiemannianConstDense(const size_t stateSpaceDim) : HamiltonianRiemannianConst{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim * stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim * stateSpaceDim);
    _inverseRegularizationParam = 1.0;

    // Initialize multivariate normal distribution
    _multivariateGenerator->_meanVector = std::vector<double>(_stateSpaceDim, 0.0);
    _multivariateGenerator->_sigma = std::vector<double>(_stateSpaceDim * _stateSpaceDim, 0.0);

    // Cholesky Decomposition
    for (size_t d = 0; d < _stateSpaceDim; ++d)
      _multivariateGenerator->_sigma[d * _stateSpaceDim + d] = sqrt(_metric[d * _stateSpaceDim + d]);

    _multivariateGenerator->updateDistribution();

    // Memory allocation
    Q = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    lambda = gsl_vector_alloc(stateSpaceDim);
    w = gsl_eigen_symmv_alloc(stateSpaceDim);
    lambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    inverseLambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatOne = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatTwo = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatThree = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatFour = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param multivariateGenerator Multivariate generator needed for momentum sampling.
  */
  HamiltonianRiemannianConstDense(const size_t stateSpaceDim, korali::distribution::multivariate::Normal *multivariateGenerator) : HamiltonianRiemannianConst{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim * stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim * stateSpaceDim);

    _multivariateGenerator = multivariateGenerator;
    _inverseRegularizationParam = 1.0;

    // Memory allocation
    Q = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    lambda = gsl_vector_alloc(stateSpaceDim);
    w = gsl_eigen_symmv_alloc(stateSpaceDim);
    lambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    inverseLambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatOne = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatTwo = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatThree = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatFour = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param multivariateGenerator Generator needed for momentum sampling.
  * @param inverseRegularizationParam Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values _inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  HamiltonianRiemannianConstDense(const size_t stateSpaceDim, korali::distribution::multivariate::Normal *multivariateGenerator, const double inverseRegularizationParam) : HamiltonianRiemannianConst{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim * stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim * stateSpaceDim);

    _multivariateGenerator = multivariateGenerator;
    _inverseRegularizationParam = inverseRegularizationParam;

    // Memory allocation
    Q = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    lambda = gsl_vector_alloc(stateSpaceDim);
    w = gsl_eigen_symmv_alloc(stateSpaceDim);
    lambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    inverseLambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatOne = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatTwo = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatThree = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatFour = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param multivariateGenerator Generator needed for momentum sampling.
  * @param metric Metric for initialization. 
  * @param inverseMetric Inverse Metric for initialization. 
  * @param inverseRegularizationParam Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values _inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  HamiltonianRiemannianConstDense(const size_t stateSpaceDim, korali::distribution::multivariate::Normal *multivariateGenerator, const std::vector<double> metric, const std::vector<double> inverseMetric, const double inverseRegularizationParam) : HamiltonianRiemannianConstDense{stateSpaceDim, multivariateGenerator, inverseRegularizationParam}
  {
    assert(metric.size() == stateSpaceDim * stateSpaceDim);
    assert(inverseMetric.size() == stateSpaceDim * stateSpaceDim);

    _metric = metric;
    _inverseMetric = inverseMetric;

    std::vector<double> mean(stateSpaceDim, 0.0);

    // Initialize multivariate normal distribution
    _multivariateGenerator->_meanVector = std::vector<double>(stateSpaceDim, 0.0);
    _multivariateGenerator->_sigma = std::vector<double>(stateSpaceDim * stateSpaceDim, 0.0);

    // Cholesky Decomposition
    for (size_t d = 0; d < stateSpaceDim; ++d)
    {
      _multivariateGenerator->_sigma[d * stateSpaceDim + d] = sqrt(metric[d * stateSpaceDim + d]);
    }

    _multivariateGenerator->updateDistribution();

    Q = gsl_matrix_alloc(_stateSpaceDim, _stateSpaceDim);
    lambda = gsl_vector_alloc(_stateSpaceDim);
    w = gsl_eigen_symmv_alloc(_stateSpaceDim);
    lambdaSoftAbs = gsl_matrix_alloc(_stateSpaceDim, _stateSpaceDim);
    inverseLambdaSoftAbs = gsl_matrix_alloc(_stateSpaceDim, _stateSpaceDim);
    tmpMatOne = gsl_matrix_alloc(_stateSpaceDim, _stateSpaceDim);
    tmpMatTwo = gsl_matrix_alloc(_stateSpaceDim, _stateSpaceDim);
    tmpMatThree = gsl_matrix_alloc(_stateSpaceDim, _stateSpaceDim);
    tmpMatFour = gsl_matrix_alloc(_stateSpaceDim, _stateSpaceDim);
  }

  /**
  * @brief Destructor of derived class.
  */
  ~HamiltonianRiemannianConstDense()
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
    double result = this->tau(p) + 0.5 * _logDetMetric;
    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianConstDense::K :" << std::endl;
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
    double tmpScalar = 0.0;

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar = 0.0;
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        tmpScalar += _inverseMetric[i * _stateSpaceDim + j] * p[j];
      }
      tmpVector[i] = tmpScalar;
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

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar += p[i] * _inverseMetric[i] * p[i];
    }

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianConstDense::tau :" << std::endl;
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

    return result;
  }

  /**
  * @brief Purely virtual gradient of phi(q) = 0.5 * logDetMetric(q) + U(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double phi() override
  {
    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianConstDense::phi :" << std::endl;
      //printf("%s\n", _sample->_js.getJson().dump(2).c_str());
    }

    return this->U() + 0.5 * _logDetMetric;
  }

  /**
  * @brief Purely virtual gradient of dphi_dq(q) = 0.5 * dlogDetMetric_dq(q) + dU(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dphi_dq() override
  {
    std::vector<double> result = this->dU();

    if (verbosity == true)
    {
      std::cout << "dphi_dq() = ";
      __printVec(result);
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
    // TODO: I think this is the same function as in hamiltonian base (D.W.)
    auto sample = korali::Sample();
    sample["Sample Id"] = _numHamiltonianObjectUpdates++;
    sample["Module"] = "Problem";
    sample["Operation"] = "Evaluate";
    sample["Parameters"] = q;

    KORALI_START(sample);
    KORALI_WAIT(sample);
    _currentEvaluation = KORALI_GET(double, sample, "logP(x)");

    // TODO: remove hack, evaluate Gradient only when required by the solver (D.W.)
    sample["Operation"] = "Evaluate Gradient";

    KORALI_START(sample);
    KORALI_WAIT(sample);
    _currentGradient = KORALI_GET(std::vector<double>, sample, "grad(logP(x))");

    sample["Operation"] = "Evaluate Hessian";

    KORALI_START(sample);
    KORALI_WAIT(sample);

    _currentHessian = KORALI_GET(std::vector<double>, sample, "H(logP(x))");
  }

  /**
  * @brief Generates sample of momentum.
  * @return Sample of momentum from normal distribution with covariance matrix _metric. Only variance taken into account with diagonal metric.
  */
  std::vector<double> sampleMomentum() const override
  {
    std::vector<double> result(_stateSpaceDim, 0.0);
    _multivariateGenerator->getRandomVector(&result[0], _stateSpaceDim);
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

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        result += pLeft[i] * _inverseMetric[i * _stateSpaceDim + j] * pRight[j];
      }
    }

    return result;
  }

  /**
  * @brief Updates Metric and Inverse Metric according to SoftAbs.
  * @param q Current position.
  * @param _k Experiment object.
  * @return Returns error code of Cholesky decomposition of GSL.
  */
  int updateMetricMatricesRiemannian(const std::vector<double> &q, korali::Experiment *_k) override
  {
    // constant for condition number of _metric
    // double detMetric = 1.0; // unused

    auto hessian = _currentHessian;
    gsl_matrix_view Xv = gsl_matrix_view_array(hessian.data(), _stateSpaceDim, _stateSpaceDim);
    gsl_matrix *X = &Xv.matrix;

    gsl_eigen_symmv(X, lambda, Q, w);

    gsl_matrix_set_all(lambdaSoftAbs, 0.0);

    gsl_matrix_set_all(inverseLambdaSoftAbs, 0.0);

    _logDetMetric = 0.0;
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      double lambdaSoftAbs_i = __softAbsFunc(gsl_vector_get(lambda, i), _inverseRegularizationParam);
      gsl_matrix_set(lambdaSoftAbs, i, i, lambdaSoftAbs_i);
      gsl_matrix_set(inverseLambdaSoftAbs, i, i, 1.0 / lambdaSoftAbs_i);
      _logDetMetric += std::log(lambdaSoftAbs_i);
    }

    gsl_matrix_set_all(tmpMatOne, 0.0);
    gsl_matrix_set_all(tmpMatTwo, 0.0);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Q, lambdaSoftAbs, 0.0, tmpMatOne);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, lambdaSoftAbs, Q, 0.0, tmpMatTwo);

    // gsl_matrix_set_all(tmpMatThree, 0.0); // unused
    gsl_matrix_set_all(tmpMatFour, 0.0);

    // gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Q, lambdaSoftAbs, 0.0, tmpMatThree); // unused
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, inverseLambdaSoftAbs, Q, 0.0, tmpMatFour);

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        _metric[i + j * _stateSpaceDim] = gsl_matrix_get(tmpMatTwo, i, j);
        _inverseMetric[i + j * _stateSpaceDim] = gsl_matrix_get(tmpMatFour, i, j);
      }
    }

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianConstDense::updateMetricMatricesRiemannian after getting hessian :" << std::endl;
      std::cout << "H = " << std::endl;
      __printVec(hessian);
    }

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianConstDense::updateMetricMatricesRiemannian end :" << std::endl;
      //printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      std::cout << "_logDetMetric = " << _logDetMetric << std::endl;
      std::cout << "_metric = " << std::endl;
      __printVec(_metric);
      std::cout << "_inverseMetric = " << std::endl;
      __printVec(_inverseMetric);
    }

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      _metric[i * _stateSpaceDim + i] = 1.0;
      _inverseMetric[i * _stateSpaceDim + i] = 1.0;
    }

    _multivariateGenerator->_sigma = _metric;

    // Cholesky Decomp
    gsl_matrix_view sigma = gsl_matrix_view_array(&_multivariateGenerator->_sigma[0], _stateSpaceDim, _stateSpaceDim);

    int err = gsl_linalg_cholesky_decomp(&sigma.matrix);
    if (err != GSL_EDOM)
    {
      _multivariateGenerator->updateDistribution();
    }

    return err;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// GENERAL FUNCTIONS END ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values _inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  double _inverseRegularizationParam;

  private:
  /**
  * @brief Multi dimensional normal generator needed for sampling of momentum from dense _metric.
  */
  korali::distribution::multivariate::Normal *_multivariateGenerator;

  gsl_matrix *Q;
  gsl_vector *lambda;
  gsl_eigen_symmv_workspace *w;
  gsl_matrix *lambdaSoftAbs;
  gsl_matrix *inverseLambdaSoftAbs;

  gsl_matrix *tmpMatOne;
  gsl_matrix *tmpMatTwo;
  gsl_matrix *tmpMatThree;
  gsl_matrix *tmpMatFour;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
