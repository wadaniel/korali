/** \namespace solver
* @brief Namespace declaration for modules of type: solver.
*/

/** \file
* @brief Header file for module: Executor.
*/

/** \dir solver/executor
* @brief Contains code, documentation, and scripts for module: Executor.
*/


#ifndef _KORALI_SOLVER_EXECUTOR_
#define _KORALI_SOLVER_EXECUTOR_


#include "modules/solver/solver.hpp"

__startNamespace__;

/**
* @brief Class declaration for module: Executor.
*/
class Executor : public Solver
{
  private:
  public: 
  /**
  * @brief Specifies the number of model executions per generation. By default this setting is 0, meaning that all executions will be performed in the first generation. For values greater 0, executions will be split into batches and split int generations for intermediate output.
  */
   size_t _executionsPerGeneration;
  /**
  * @brief [Internal Use] Number of samples to execute.
  */
   size_t _sampleCount;
  
 
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
  * @brief Generate a sample and evaluate it.
  */
  void runGeneration() override;

  /**
  * @brief Console Output before generation runs.
  */
  void printGenerationBefore() override;

  /**
  * @brief Console output after generation.
  */
  void printGenerationAfter() override;
};

__endNamespace__;

#endif // _KORALI_SOLVER_EXECUTOR_
