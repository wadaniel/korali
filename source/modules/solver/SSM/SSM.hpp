/** \namespace solver
* @brief Namespace declaration for modules of type: solver.
*/

/** \file
* @brief Header file for module: SSM.
*/

/** \dir solver/SSM
* @brief Contains code, documentation, and scripts for module: SSM.
*/

#pragma once

#include "modules/solver/solver.hpp"
#include "modules/problem/reaction/reaction.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"

namespace korali
{
namespace solver
{
;

/**
* @brief Class declaration for module: SSM.
*/
class SSM : public Solver
{
  public: 
  /**
  * @brief Total duration of a stochastic reaction simulation.
  */
   double _simulationLength;
  /**
  * @brief Number of bins to calculate the mean trajectory at termination.
  */
   size_t _diagnosticsNumBins;
  /**
  * @brief [Internal Use] The current time of the simulated trajectory.
  */
   double _time;
  /**
  * @brief [Internal Use] The current number of reactants in the simulated trajectory.
  */
   std::vector<int> _numReactants;
  /**
  * @brief [Internal Use] Uniform random number generator.
  */
   korali::distribution::univariate::Uniform* _uniformGenerator;
  /**
  * @brief [Termination Criteria] Max number of trajectory simulations.
  */
   size_t _maxNumSimulations;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  /**
   * @brief Storage for the pointer to the (continuous) learning problem
   */
  problem::Reaction *_problem;
  
  /**
   * @brief Resets the initial conditions of a new trajectory simulation.
   */
  void reset(std::vector<int> numReactants, double time = 0.);
  
  /**
   * @brief Simulates a trajectory for all reactants based on provided reactions.
   */
  virtual void advance() = 0;
  

  void initialize() override;
  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;

};

} //solver
} //korali
;
