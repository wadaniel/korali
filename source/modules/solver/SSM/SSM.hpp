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
#include "modules/distribution/univariate/exponential/exponential.hpp"

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
  * @brief Number of bins to calculate the mean trajectory at termination.
  */
   size_t _numBins;
  /**
  * @brief [Internal Use] Exponential random number generator.
  */
   korali::distribution::univariate::Exponential* _exponentialGenerator;
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
   * @brief Simulates a trajectory for all reactants based on provided reactions.
   */
  virtual void simulateTrajectory() = 0;

  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;

};

} //solver
} //korali
;
