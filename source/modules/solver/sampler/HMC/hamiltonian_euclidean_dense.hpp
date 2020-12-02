#ifndef HAMILTONIAN_EUCLIDEAN_DENSE_H
#define HAMILTONIAN_EUCLIDEAN_DENSE_H

#include "hamiltonian_euclidean_base.hpp"
#include "modules/distribution/multivariate/normal/normal.hpp"

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics.h>

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class HamiltonianEuclideanDense
* @brief Used for calculating energies with euclidean metric.
*/
class HamiltonianEuclideanDense : public HamiltonianEuclidean
{
  public:
  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CONSTRUCTORS START /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianEuclideanDense(const size_t stateSpaceDim) : HamiltonianEuclidean{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim * stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim * stateSpaceDim);
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param multivariateGenerator Generator needed for momentum sampling.
  */
  HamiltonianEuclideanDense(const size_t stateSpaceDim, korali::distribution::multivariate::Normal *multivariateGenerator) : HamiltonianEuclidean{stateSpaceDim}
  {
    _metric.resize(stateSpaceDim * stateSpaceDim);
    _inverseMetric.resize(stateSpaceDim * stateSpaceDim);

    _multivariateGenerator = multivariateGenerator;
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param multivariateGenerator Generator needed for momentum sampling.
  * @param metric Metric for initialization. 
  * @param inverseMetric Inverse Metric for initialization. 
  */
  HamiltonianEuclideanDense(const size_t stateSpaceDim, korali::distribution::multivariate::Normal *multivariateGenerator, const std::vector<double> metric, const std::vector<double> inverseMetric) : HamiltonianEuclideanDense{stateSpaceDim, multivariateGenerator}
  {
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
  }

  /**
  * @brief Destructor of derived class.
  */
  ~HamiltonianEuclideanDense()
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
    return K(p) + U();
  }

  /**
  * @brief Kinetic energy function K(q, p) = 0.5 * p.T * inverseMetric(q) * p + 0.5 * logDetMetric(q) used for Hamiltonian Dynamics. For Euclidean metric logDetMetric(q) := 0.0.
  * @param p Current momentum.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &p) override
  {
    double tmpScalar = 0.0;
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        tmpScalar += p[i] * _inverseMetric[i * _stateSpaceDim + j] * p[j];
      }
    }

    return 0.5 * tmpScalar;
  }

  /**
  * @brief Purely virtual gradient of kintetic energy function dK(q, p) = inverseMetric(q) * p + 0.5 * dlogDetMetric_dq(q) used for Hamiltonian Dynamics. For Euclidean metric logDetMetric(q) := 0.0.
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

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// ENERGY FUNCTIONS END ////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// GENERAL FUNCTIONS START //////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Generates sample of momentum.
  * @return Sample of momentum from normal distribution with covariance matrix _metric.
  */
  std::vector<double> sampleMomentum() const override
  {
    // TODO: Change
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
  * @brief Updates Inverse Metric by using samples to approximate the covariance matrix via the Fisher information.
  * @param samples Contains samples. One row is one sample.
  * @param positionMean Mean of samples.
  * @return Error code of Cholesky decomposition used to invert matrix.
  */
  int updateMetricMatricesEuclidean(const std::vector<std::vector<double>> &samples) override
  {
    double sumk, sumi, sumOfSquares;
    double cov;
    double numSamples = samples.size();

    // calculate sample covariance
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t k = i; k < _stateSpaceDim; ++k)
      {
        sumk = 0.;
        sumi = 0.;
        sumOfSquares = 0.;
        for (size_t j = 0; j < numSamples; ++j)
        {
          sumi += samples[j][i];
          sumk += samples[j][k];
          sumOfSquares += samples[j][i]*samples[j][k];
        }
        cov = (sumOfSquares-(sumk*sumi)/numSamples)/(numSamples-1.);
        _inverseMetric[i * _stateSpaceDim + k] = cov;
        _inverseMetric[k * _stateSpaceDim + i] = cov;
      }
    }

    // update Metric to be consisitent with Inverse Metric
    int err = __invertMatrix(_inverseMetric, _metric);
    if (err > 0) return err;

    std::vector<double> sig = _metric;
    gsl_matrix_view sigView = gsl_matrix_view_array(&sig[0], _stateSpaceDim, _stateSpaceDim);

    // Cholesky Decomp
    err = gsl_linalg_cholesky_decomp(&sigView.matrix);
    if (err == 0) 
    {
        _multivariateGenerator->_sigma = sig;
        _multivariateGenerator->updateDistribution();
    }


    return err;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// GENERAL FUNCTIONS END ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// HELPERS START ///////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  protected:
  // inverts mat via cholesky decomposition and writes inverted Matrix to inverseMat
  // TODO: Avoid calculating cholesky decompisition twice

  /**
  * @brief Inverts s.p.d. matrix via Cholesky decomposition.
  * @param mat Input matrix interpreted as square symmetric matrix.
  * @param inverseMat Result of inversion.
  * @return Error code of Cholesky decomposition used to invert matrix.
  */
  int __invertMatrix(const std::vector<double> &matrix, std::vector<double> &inverseMat)
  {
    const size_t dim = (size_t)std::sqrt(matrix.size());
    gsl_matrix_view invView = gsl_matrix_view_array(&inverseMat[0], dim, dim);
    gsl_matrix_const_view matView = gsl_matrix_const_view_array(&matrix[0], dim, dim);

    gsl_permutation *p = gsl_permutation_alloc(dim);
    int s;
    
    gsl_matrix *luMat = gsl_matrix_alloc(dim, dim);
    gsl_matrix_memcpy(luMat, &matView.matrix);
    gsl_linalg_LU_decomp(luMat, p, &s);
    int err = gsl_linalg_LU_invert(luMat, p, &invView.matrix);

    // free up memory of gsl matrix
    gsl_permutation_free(p);
    gsl_matrix_free(luMat);

    return err;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// HELPERS END ////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  private:
  /**
  * @brief Multivariate normal generator needed for sampling of momentum from dense _metric.
  */
  korali::distribution::multivariate::Normal *_multivariateGenerator;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
