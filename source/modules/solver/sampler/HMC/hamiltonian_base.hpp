#ifndef HAMILTONIAN_BASE_H
#define HAMILTONIAN_BASE_H

#include "modules/conduit/conduit.hpp"

#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/bayesian/bayesian.hpp"
#include "modules/problem/bayesian/reference/reference.hpp"
#include "modules/problem/problem.hpp"
#include "modules/problem/sampling/sampling.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class Hamiltonian
* @brief Abstract base class for Hamiltonian objects.
*/
class Hamiltonian
{
  public:
  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// CONSTRUCTORS START /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Default constructor.
  */
  Hamiltonian() = default;

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  Hamiltonian(const size_t stateSpaceDim, korali::Experiment *k) : _stateSpaceDim{stateSpaceDim}, _numHamiltonianObjectUpdates{0} {
  
    _k = k;
    samplingProblemPtr = dynamic_cast<korali::problem::Sampling *>(k->_problem);
    bayesianProblemPtr = dynamic_cast<korali::problem::Bayesian *>(k->_problem);

    if (samplingProblemPtr != nullptr && bayesianProblemPtr != nullptr)
        KORALI_LOG_ERROR("Problem type not compatible with Hamiltonian object.");
 
  };

  /**
  * @brief Destructor of abstract base class.
  */
  virtual ~Hamiltonian()
  {
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////// CONSTRUCTORS END //////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// ENERGY FUNCTIONS START ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Purely abstract total energy function used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Total energy.
  */
  virtual double H(const std::vector<double> &p) = 0;

  /**
  * @brief Purely virtual kinetic energy function K(q, p) = 0.5 * p.T * inverseMetric(q) * p + 0.5 * logDetMetric(q) used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Kinetic energy.
  */
  virtual double K(const std::vector<double> &p) = 0;

  /**
  * @brief Purely virtual gradient of kintetic energy function dK(q, p) = inverseMetric(q) * p + 0.5 * dlogDetMetric_dq(q) used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  virtual std::vector<double> dK(const std::vector<double> &p) = 0;

  /**
  * @brief Potential Energy function U(q) = -log(pi(q)) used for Hamiltonian Dynamics.
  * @return Potential energy.
  */
  virtual double U()
  {
    double evaluation = _currentEvaluation;
    evaluation *= -1.0;

    return evaluation;
  }

  /**
  * @brief Gradient of Potential Energy function dU(q) = -grad(log(pi(q))) used for Hamiltonian Dynamics.
  * @return Gradient of Potential energy.
  */
  virtual std::vector<double> dU()
  {
    auto grad = _currentGradient;

    // negate to get dU
    std::transform(grad.cbegin(), grad.cend(), grad.begin(), std::negate<double>());

    return grad;
  }

  /**
  * @brief Purely virtual function tau(q, p) = 0.5 * p^T * inverseMetric(q) * p (no logDetMetric term)
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  virtual double tau(const std::vector<double> &p) = 0;

  /**
  * @brief Purely virtual gradient of dtau_dq(q, p) = 0.5 * p^T * dinverseMetric_dq(q) * p used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  virtual std::vector<double> dtau_dq(const std::vector<double> &p) = 0;

  /**
  * @brief Purely virtual gradient of dtau_dp(q, p) = inverseMetric(q) * p used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  virtual std::vector<double> dtau_dp(const std::vector<double> &p) = 0;

  /**
  * @brief Purely virtual gradient of phi(q) = 0.5 * logDetMetric(q) + U(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  virtual double phi() = 0;

  /**
  * @brief Purely virtual gradient of dphi_dq(q) = 0.5 * dlogDetMetric_dq(q) + dU(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  virtual std::vector<double> dphi_dq() = 0;

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
  virtual void updateHamiltonian(const std::vector<double> &q)
  {
    auto sample = korali::Sample();
    sample["Sample Id"] = _numHamiltonianObjectUpdates++;
    sample["Module"] = "Problem";
    sample["Operation"] = "Evaluate";
    sample["Parameters"] = q;

    KORALI_START(sample);
    KORALI_WAIT(sample);
    _currentEvaluation = KORALI_GET(double, sample, "logP(x)");
 
    if(samplingProblemPtr != nullptr)
      samplingProblemPtr->evaluateGradient(sample);
    else
      bayesianProblemPtr->evaluateGradient(sample);
 
    _currentGradient = sample["grad(logP(x))"].get<std::vector<double>>();
  }

  /**
  * @brief Generates sample of momentum.
  * @return Sample of momentum from normal distribution with covariance matrix _metric.
  */
  virtual std::vector<double> sampleMomentum() const = 0;

  /**
  * @brief Calculates inner product induces by inverse metric.
  * @param pLeft Left argument (momentum).
  * @param pRight Right argument (momentum).
  * @return pLeft.transpose * _inverseMetric * pRight.
  */
  virtual double innerProduct(std::vector<double> pLeft, std::vector<double> pRight) const = 0;

  /**
  * @brief Computes NUTS criterion on euclidean domain.
  * @param qLeft Leftmost position.
  * @param pLeft Leftmost momentum.
  * @param qRight Rightmost position.
  * @param pRight Rightmost momentum.
  * @return Returns if trees should be built further.
  */
  bool computeStandardCriterion(const std::vector<double> &qLeft, const std::vector<double> &pLeft, const std::vector<double> &qRight, const std::vector<double> &pRight)
  {
    std::vector<double> tmpVector(_stateSpaceDim, 0.0);

    std::transform(std::cbegin(qRight), std::cend(qRight), std::cbegin(qLeft), std::begin(tmpVector), std::minus<double>());
    double dotProductLeft = std::inner_product(std::cbegin(tmpVector), std::cend(tmpVector), std::cbegin(pLeft), 0.0);
    double dotProductRight = std::inner_product(std::cbegin(tmpVector), std::cend(tmpVector), std::cbegin(pRight), 0.0);

    return (dotProductLeft >= 0) && (dotProductRight >= 0);
  }

  /**
  * @brief Updates Inverse Metric by using samples to approximate the covariance matrix via the Fisher information.
  * @param samples Contains samples. One row is one sample.
  * @return Error code of Cholesky decomposition needed for dense Metric.
  */
  virtual int updateMetricMatricesEuclidean(const std::vector<std::vector<double>> &samples)
  {
    return -1;
  };

  /**
  * @brief Updates Inverse Metric by using hessian.
  * @param q current position
  * @param _k Korali experiment object
  * @return Error code to indicate if update was successful.
  */
  virtual int updateMetricMatricesRiemannian(const std::vector<double> &q)
  {
    return 0;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// GENERAL FUNCTIONS END ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// GETTERS START ///////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Getter function for State Space Dim.
  * @return Dimension of state space (i.e. dim(q)).
  */
  const size_t getStateSpaceDim() const
  {
    return _stateSpaceDim;
  }

  /**
  * @brief Getter function for metric.
  * @return Returns metric of hamiltonian.
  */
  std::vector<double> getMetric() const
  {
    return _metric;
  }

  /**
  * @brief Getter function for inverse metric.
  * @return Returns inverse metric of hamiltonian.
  */
  std::vector<double> getInverseMetric() const
  {
    return _inverseMetric;
  }

  /**
  * @brief Getter function for number of hamiltonian object updates.
  * @return Number of hamiltonian object updates.
  */
  size_t getNumHamiltonianObjectUpdates() const
  {
    return _numHamiltonianObjectUpdates;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// GETTERS END ////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////// SETTERS START ///////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Setter function for metric.
  * @param metric Metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setMetric(std::vector<double> &metric)
  {
    if (metric.size() != _metric.size())
    {
      return false;
    }
    else
    {
      _metric = metric;
      return true;
    }
  }

  /**
  * @brief Setter function for inverse metric.
  * @param inverseMetric Inverse metric which is set.
  * @return Returns true if dimensions are compatible. Returns false if dimension mismatch found.
  */
  bool setInverseMetric(std::vector<double> &inverseMetric)
  {
    if (inverseMetric.size() != _inverseMetric.size())
    {
      return false;
    }
    else
    {
      _inverseMetric = inverseMetric;
      return true;
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// SETTERS END ////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  protected:
 
  /**
  @brief Pointer to the korali experiment.
  */
  korali::Experiment * _k;

  /**
  @brief Pointer to the sampling problem (might be NULL)
  */
  korali::problem::Sampling* samplingProblemPtr;
 
  /**
  @brief Pointer to the Bayesian problem (might be NULL)
  */
  korali::problem::Bayesian* bayesianProblemPtr;

  /**
  @brief Current evaluation of objective (return value of sample evaluation).
  */
  double _currentEvaluation;

  /**
  * @brief Current gradient of objective (return value of sample evaluation).
  */
  std::vector<double> _currentGradient;

  /**
  * @brief _stateSpaceDim
State Space Dimension needed for Leapfrog integrator.
  */
  size_t _stateSpaceDim;

  /**
  * @brief _inverseMetric
  */
  std::vector<double> _metric;

  /**
  * @brief _inverseMetric
  */
  std::vector<double> _inverseMetric;

  /**
  * @brief Number of model evaluations
  */
  size_t _numHamiltonianObjectUpdates;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
