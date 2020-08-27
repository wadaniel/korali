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
  * @param q Current position.
  * @param p Current momentum.
  * @param _k Experiment object.
  * @return Total energy.
  */
  double H(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k) override
  {
    return K(q, p) + U(q, _k);
  }

  /**
  * @brief Kinetic energy function K(q, p) = 0.5 * p.T * inverseMetric(q) * p + 0.5 * logDetMetric(q) used for Hamiltonian Dynamics. For Euclidean metric logDetMetric(q) := 0.0.
  * @param q Current position.
  * @param p Current momentum.
  * @param _k Experiment object.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k = 0) override
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
  * @param q Current position.
  * @param p Current momentum.
  * @param _k Experiment object.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dK(const std::vector<double> &q, const std::vector<double> &p, korali::Experiment *_k = 0) override
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
  int updateInverseMetric(const std::vector<std::vector<double>> &samples, const std::vector<double> &positionMean) override
  {
    double tmpScalar;
    size_t numSamples = samples.size();

    // calculate covariance matrix of warmup sample via Fisher Infromation
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t k = i; k < _stateSpaceDim; ++k)
      {
        tmpScalar = 0;
        for (size_t j = 0; j < numSamples; ++j)
        {
          tmpScalar += (samples[j][i] - positionMean[i]) * (samples[j][k] - positionMean[k]);
        }
        _inverseMetric[i * _stateSpaceDim + k] = tmpScalar / (numSamples - 1);
        _inverseMetric[k * _stateSpaceDim + i] = _inverseMetric[i * _stateSpaceDim + k];
      }
    }

    // update Metric to be consisitent with Inverse Metric
    int err = __invertMatrix(_inverseMetric, _metric);

    _multivariateGenerator->_sigma = _metric;

    // Cholesky Decomp
    gsl_matrix_view sigma = gsl_matrix_view_array(&_multivariateGenerator->_sigma[0], _stateSpaceDim, _stateSpaceDim);

    err = gsl_linalg_cholesky_decomp(&sigma.matrix);
    if (err == GSL_EDOM)
    {
      // Do nothing if error occurs
    }
    else
    {
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
  int __invertMatrix(std::vector<double> &mat, std::vector<double> &inverseMat)
  {
    int _stateSpaceDim = (int)std::sqrt(mat.size());
    gsl_matrix *A = gsl_matrix_alloc(_stateSpaceDim, _stateSpaceDim);

    // copy mat to gsl matrix
    for (size_t d = 0; d < _stateSpaceDim; ++d)
    {
      for (size_t e = 0; e < d; ++e)
      {
        gsl_matrix_set(A, d, e, mat[d * _stateSpaceDim + e]);
        gsl_matrix_set(A, e, d, mat[e * _stateSpaceDim + d]);
      }
      gsl_matrix_set(A, d, d, mat[d * _stateSpaceDim + d]);
    }

    // calculate cholesky decomposition
    int err = gsl_linalg_cholesky_decomp(A);
    if (err == GSL_EDOM)
    {
      // error handling for non s.p.d. matrices
    }
    else
    {
      // Invert matrix
      gsl_linalg_cholesky_invert(A);

      // copy gsl matrix to inverseMat
      // TODO: Find out if there is a better way to do this
      for (size_t d = 0; d < _stateSpaceDim; ++d)
      {
        for (size_t e = 0; e < d; ++e)
        {
          inverseMat[d * _stateSpaceDim + e] = gsl_matrix_get(A, d, e);
          inverseMat[e * _stateSpaceDim + d] = gsl_matrix_get(A, d, e);
        }
        inverseMat[d * _stateSpaceDim + d] = gsl_matrix_get(A, d, d);
      }
    }

    // free up memory of gsl matrix
    gsl_matrix_free(A);

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