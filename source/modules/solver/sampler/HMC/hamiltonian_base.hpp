#ifndef HAMILTONIAN_BASE_H
#define HAMILTONIAN_BASE_H

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
  Hamiltonian(const size_t stateSpaceDim) : _stateSpaceDim{stateSpaceDim}, _numHamiltonianObjectUpdates{0}
  {
    _sample = new korali::Sample();
    (*_sample)["Sample Id"] = 0;
    (*_sample)["Module"] = "Problem";
    (*_sample)["Operation"] = "Evaluate";
    verbosity = false;
  }

  /**
  * @brief Destructor of abstract base class.
  */
  virtual ~Hamiltonian()
  {
    delete _sample;
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
    ++_numHamiltonianObjectUpdates;
    double evaluation = KORALI_GET(double, (*_sample), "logP(x)");
    evaluation *= -1.0;

    if (verbosity == true)
    {
      std::cout << "In Hamiltonian::U :" << std::endl;
      std::cout << "U() = " << evaluation << std::endl
                << std::endl;
    }

    return evaluation;
  }

  /**
  * @brief Gradient of Potential Energy function dU(q) = -grad(log(pi(q))) used for Hamiltonian Dynamics.
  * @return Gradient of Potential energy.
  */
  virtual std::vector<double> dU()
  {
    // evaluate grad(logP(x)) (extremely slow)
    auto grad = KORALI_GET(std::vector<double>, (*_sample), "grad(logP(x))");

    // negate to get dU
    std::transform(grad.cbegin(), grad.cend(), grad.begin(), std::negate<double>());

    if (verbosity == true)
    {
      std::cout << "In Hamiltonian::dU :" << std::endl;
      std::cout << "dU() = " << std::endl;
      __printVec(grad);
      std::cout << std::endl;
    }

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
  virtual void updateHamiltonian(const std::vector<double> &q, korali::Experiment *_k)
  {
    (*_sample)["Parameters"] = q;

    KORALI_START((*_sample));
    KORALI_WAIT((*_sample));

    // TODO: remove hack, evaluate Gradient only when required by the solver (D.W.)
    (*_sample)["Operation"] = "Evaluate Gradient";
    KORALI_START((*_sample));
    KORALI_WAIT((*_sample));
    (*_sample)["Operation"] = "Evaluate";

    if (verbosity == true)
    {
      std::cout << "In Hamiltonian::updateHamiltonian " << std::endl;
      printf("%s\n", _sample->_js.getJson().dump(2).c_str());
    }
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

    if (verbosity)
    {
      std::cout << "In Hamiltonian::computeStandardCriterion :" << std::endl;
      std::cout << "qLeft = " << std::endl;
      __printVec(qLeft);
      std::cout << "pLeft = " << std::endl;
      __printVec(pLeft);
      std::cout << "qRight = " << std::endl;
      __printVec(qRight);
      std::cout << "pRight = " << std::endl;
      __printVec(pRight);
      std::cout << "dotProductLeft = " << dotProductLeft << std::endl;
      std::cout << "dotProductRight = " << dotProductRight << std::endl;
      bool __returnVal = (dotProductLeft > 0) && (dotProductRight > 0);
      std::cout << "return (dotProductLeft > 0) && (dotProductRight > 0) = " << __returnVal << std::endl;
    }

    return (dotProductLeft > 0) && (dotProductRight > 0);
  }

  /**
  * @brief Updates Inverse Metric by using samples to approximate the covariance matrix via the Fisher information.
  * @param samples Contains samples. One row is one sample.
  * @param positionMean Mean of samples.
  * @return Error code of Cholesky decomposition needed for dense Metric.
  */
  virtual int updateMetricMatricesEuclidean(const std::vector<std::vector<double>> &samples, const std::vector<double> &positionMean) 
  {
    return -1;
  };

  /**
  * @brief Updates Inverse Metric by using hessian.
  * @param q current position
  * @param _k Korali experiment object
  */
  virtual void updateMetricMatricesRiemannian(const std::vector<double> &q, korali::Experiment *_k) 
  {
    if (verbosity == true)
    {
      std::cout  << "in Hamiltonian::updateMetricMatricesRiemannian" << std::endl;      
    }

    return;
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

  //////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////// DEBUGGER MEMBERS START ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief verbosity boolean for debuggin purposes.
  */
  bool verbosity;
  
  /**
  * @brief Debug printer function for std::vector. TODO: REMOVE
  * @param vec Vector to be printed.
  */
  void __printVec(std::vector<double> vec)
  {
    for (size_t i = 0; i < vec.size(); ++i)
    {
      std::cout << vec[i] << std::endl;
    }
    return;
  }

  protected:
  //////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// DEBUGGER MEMBERS END ////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////

  /**
  * @brief Pointer to current sample object.
  */
  korali::Sample *_sample;

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
