#ifndef HAMILTONIAN_DENSE_H
#define HAMILTONIAN_DENSE_H

#include "hamiltonian_base.hpp"

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
* \class HamiltonianDense
* @brief Used for calculating energies with euclidean metric.
*/
class HamiltonianDense : public Hamiltonian
{
  public:
  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianDense(const size_t stateSpaceDim) : Hamiltonian{stateSpaceDim} {}

  /**
  * @brief Total energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param modelEvaluationCount Number of model evuations musst be increased.
  * @param numSamples Needed for Sample ID.
  * @param inverseMetric Inverse Metric must be provided.
  * @return Total energy.
  */
  double H(const std::vector<double> &q, const std::vector<double> &p, size_t &modelEvaluationCount, const size_t &numSamples, const std::vector<double> &inverseMetric) override
  {
    return K(q, p, inverseMetric) + U(q, modelEvaluationCount, numSamples);
  }

  /**
  * @brief Kinetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param inverseMetric Inverse Metric must be provided.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &q, const std::vector<double> &p, const std::vector<double> &inverseMetric) const override
  {
    double tmpScalar = 0.0;
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        tmpScalar += p[i] * inverseMetric[i * _stateSpaceDim + j] * p[j];
      }
    }

    // std::cout << "K(" << p[0] << ") = " << 0.5 * tmpScalar << std::endl;
    return 0.5 * tmpScalar;
  }

  /**
  * @brief Gradient of kintetic energy function used for Hamiltonian Dynamics.
  * @param q Current position.
  * @param p Current momentum.
  * @param inverseMetric Inverse Metric must be provided.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dK(const std::vector<double> &q, const std::vector<double> &p, const std::vector<double> &inverseMetric) const override
  {
    std::vector<double> tmpVector(_stateSpaceDim, 0.0);
    double tmpScalar = 0.0;

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar = 0.0;
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        tmpScalar += inverseMetric[i * _stateSpaceDim + j] * p[j];
      }
      tmpVector[i] = tmpScalar;
    }
    return tmpVector;
  }

  /**
  * @brief Updates Inverse Metric by using samples to approximate the covariance matrix via the Fisher information.
  * @param samples Contains samples. One row is one sample.
  * @param positionMean Mean of samples.
  * @param inverseMetric Inverse Metric to be approximated.
  * @param metric Metric which is calculated from inverMetric.
  * @return Error code of Cholesky decomposition used to invert matrix.
  */
  int updateInverseMetric(const std::vector<std::vector<double>> &samples, const std::vector<double> &positionMean, std::vector<double> &inverseMetric, std::vector<double> &metric) override
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
        inverseMetric[i * _stateSpaceDim + k] = tmpScalar / (numSamples - 1);
        inverseMetric[k * _stateSpaceDim + i] = inverseMetric[i * _stateSpaceDim + k];
      }
    }

    // update Metric to be consisitent with Inverse Metric
    int err = invertMatrix(inverseMetric, metric);

    return err;
  }

  protected:
  // inverts mat via cholesky decomposition and writes inverted Matrix to inverseMat
  // TODO: Avoid calculating cholesky decompisition twice

  /**
  * @brief Inverts s.p.d. matrix via Cholesky decomposition.
  * @param mat Input matrix interpreted as square symmetric matrix.
  * @param inverseMat Result of inversion.
  * @return Error code of Cholesky decomposition used to invert matrix.
  */
  int invertMatrix(std::vector<double> &mat, std::vector<double> &inverseMat)
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
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif