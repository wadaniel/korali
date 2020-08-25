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
  * @brief Total energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param _k Experiment object.
  * @return Total energy.
  */
  double H(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k) override
  {
    return K(q, p, _k) + U(q, _k);
  }

  /**
  * @brief Kinetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k = 0) override
  {
    // make sure hamiltonian updated in tau
    if(_k == 0)
    {
      std::cout << "Error in RiemannianHamiltonianDiag::K : Experiment pointer _k initialized with nullptr" << std::endl;
    }
    
    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::K :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    return 0.5 * (tau(q, p, _k) + _logDetMetric);
  }

  /**
  * @brief Gradient of kintetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dK(const std::vector<double> &q, const std::vector<double> &p) override
  {
    std::vector<double> tmpVector(_stateSpaceDim, 0.0);
    for (int i = 0; i < _stateSpaceDim; ++i)
    {
      tmpVector[i] = _inverseMetric[i] * p[i];
    }
    return tmpVector;
  }

  double tau(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k)
  {
    double tmpScalar = 0.0;

    updateHamiltonian(q, _k);

    for (int i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar += p[i] * _inverseMetric[i] * p[i];
    }

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::tau :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    return 0.5 * tmpScalar;
  }

  std::vector<double> dtau_dq(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k)
  {
    std::vector<double> result(_stateSpaceDim, 0.0);
    std::vector<double> gradU = dU(q, _k);
    std::vector<double> H_U = hessianU(q, _k);
    
    for(size_t j = 0; j < _stateSpaceDim; ++j)
    {
      result[j] = 0.0;
      for(size_t i = 0; i < _stateSpaceDim; ++i)
      {
        double arg = _inverseRegularizationParam * (gradU[i] * gradU[i]);
        result[j] += H_U[i*_stateSpaceDim + j] / (std::pow(gradU[i], 3)) * (_inverseRegularizationParam / std::cosh(arg) - std::tanh(arg)) * p[i] * p[i];
      }
    }

    if(verbosity == true)
    {
      std::cout << "dtau_dq(q, p, _k) = ";
      __printVec(result);
      std::cout << "with q = " << std::endl;
      __printVec(q);
      std::cout << "with p = " << std::endl;
      __printVec(p);
    }

    return result;
  }

  std::vector<double> dtau_dp(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k)
  {
    updateHamiltonian(q, _k);

    
    if(verbosity == true)
    {
      std::cout << "dtau_dp(q, p, _k) = ";
      __printVec(dK(q, p));
      std::cout << "with q = " << std::endl;
      __printVec(q);
      std::cout << "with p = " << std::endl;
      __printVec(p);
    }

    return dK(q, p);
  }

  double phi(const std::vector<double> q, korali::Experiment *_k)
  {
    // make sure hamiltonian updated in U 
    
    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::phi :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    return U(q, _k) + 0.5 * _logDetMetric;
  }

  std::vector<double> dphi_dq(const std::vector<double> q, korali::Experiment *_k)
  {
    std::vector<double> result(_stateSpaceDim, 0.0);
    std::vector<double> gradU = dU(q, _k);
    std::vector<double> H_U = hessianU(q, _k);

    std::vector<double> dLogDetMetric_dq(_stateSpaceDim, 0.0);
    
    for(size_t j = 0; j < _stateSpaceDim; ++j)
    {
      dLogDetMetric_dq[j] = 0.0;
      for(size_t i = 0; i < _stateSpaceDim; ++i)
      {
        double arg = _inverseRegularizationParam * (gradU[i] * gradU[i]);
        dLogDetMetric_dq[j] += 1.0 / _metric[i] * ( 2.0 * H_U[i*_stateSpaceDim + j] * gradU[i] * 1.0 / std::tanh(arg) - 2.0 * _inverseRegularizationParam * H_U[i*_stateSpaceDim + j] * gradU[i] / (std::sinh(arg) * std::sinh(arg)) );
      }
    }

    for(size_t j = 0; j < _stateSpaceDim; ++j)
    {
      result[j] = gradU[j] + 0.5 * dLogDetMetric_dq[j];
    }

    if(verbosity == true)
    {
      std::cout << "dphi_dq(q, _k) = ";
      __printVec(result);
      std::cout << "with q = " << std::endl;
      __printVec(q);
    }
    
    return result;
  }

  /**
  * @brief Updates current position of hamiltonian.
  * @param q Current position.
  * @param _k Experiment object.
  */
  void updateHamiltonian(const std::vector<double> &q, korali::Experiment *_k) override
  {
    if(verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::updateHamiltonian before adaptMetrics :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    if(verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::updateHamiltonian before setting Paramters :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    (*_sample)["Parameters"] = q;

    if(verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::updateHamiltonian after setting Paramters :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    KORALI_START((*_sample));
    KORALI_WAIT((*_sample));


    if(verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::updateHamiltonian after KORALI_WAIT :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    // constant for condition number of _metric
    double detMetric = 1.0;

    std::vector<double> g = KORALI_GET(std::vector<double>, (*_sample), "grad(logP(x))");

    if(verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::updateHamiltonian after getting gradient :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      double gi2 = g[i] * g[i];
      double coth_alpha_gi_2 = 1.0 / std::tanh(_inverseRegularizationParam * gi2);
      _metric[i] = gi2 * coth_alpha_gi_2;
      _inverseMetric[i] = 1.0 / _metric[i];
      detMetric *= _metric[i];
    }
    _logDetMetric = std::log(detMetric);

    if(verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::updateHamiltonian end :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
      std::cout << "_logDetMetric = " << _logDetMetric << std::endl;
      std::cout << "_metric = " << std::endl;
      __printVec(_metric);
      std::cout << "_inverseMetric = " << std::endl;
      __printVec(_inverseMetric);
    }
  }

  void adaptMetrics(const std::vector<double> &q, korali::Experiment *_k)
  {

    return;
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

  /**
  * @brief Hessian of Potential Energy function used for Riemannian metric.
  * @param q Current position.
  * @param _k Experiment object.
  * @return Gradient of Potential energy.
  */
  std::vector<double> hessianU(const std::vector<double> &q, korali::Experiment *_k)
  {
    updateHamiltonian(q, _k);

    ++_numHamiltonianObjectUpdates;

    // evaluate grad(logP(x)) (extremely slow)
    std::vector<std::vector<double>> evaluationMat = KORALI_GET(std::vector<std::vector<double>>, (*_sample), "H(logP(x))");
    std::vector<double> evaluation(_stateSpaceDim * _stateSpaceDim);
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        evaluation[i * _stateSpaceDim + j] = evaluationMat[i][j];
      }
    }

    // negate to get dU
    std::transform(evaluation.cbegin(), evaluation.cend(), evaluation.begin(), std::negate<double>());

    if (verbosity == true)
    {
      std::cout << "In HamiltonianRiemannianDiag::hessianU :" << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
      fflush(stdout);
    }

    return evaluation;
  }

  // TODO: Remove
  /**
  * @brief Setter function for metric.
  * @param metric Metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setMetric(std::vector<double> &metric) override { return false; };

  // TODO: Remove
  /**
  * @brief Setter function for inverse metric.
  * @param inverseMetric Inverse metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setInverseMetric(std::vector<double> &inverseMetric) override { return false; };

  private:
  /**
  * @brief Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values _inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  double _inverseRegularizationParam;

  /**
  * @brief One dimensional normal generator needed for sampling of momentum from diagonal _metric.
  */
  korali::distribution::univariate::Normal *_normalGenerator;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif