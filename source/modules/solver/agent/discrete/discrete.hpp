/** \namespace agent
* @brief Namespace declaration for modules of type: agent.
*/

/** \file
* @brief Header file for module: Discrete.
*/

/** \dir solver/agent/discrete
* @brief Contains code, documentation, and scripts for module: Discrete.
*/

#pragma once

#include "modules/problem/reinforcementLearning/discrete/discrete.hpp"
#include "modules/solver/agent/agent.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
;

/**
* @brief Class declaration for module: Discrete.
*/
class Discrete : public Agent
{
  public: 
  /**
  * @brief Specifies the probability of taking a random action for the epsilon-greedy strategy.
  */
   float _randomActionProbability;
  
 
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
 * @brief Storage for the pointer to the (discrete) learning problem
 */
  problem::reinforcementLearning::Discrete *_problem;

  float calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy) override;

  /**
   * @brief Calculates the gradient of importance weight wrt to NN output
   * @param actionIdx Action from memory
   * @param curPvalues todo
   * @param oldPvalues todo
   * @return gradient of importance weight wrt NN output
   */
  std::vector<float> calculateImportanceWeightGradient(const size_t actionIdx, const std::vector<float> &curPvalues, const std::vector<float> &oldPvalues);

  /**
   * @brief Calculates the gradient of KL(p_old, p_cur) wrt to the parameter of the 2nd (current) distribution.
   * @param oldPvalues todo
   * @param curPvalues todo
   * @return gradient of KL wrt curParamsOne and curParamsTwo
   */
  std::vector<float> calculateKLDivergenceGradient(const std::vector<float> &oldPvalues, const std::vector<float> &curPvalues);

  void getAction(korali::Sample &sample) override;
  virtual void initializeAgent() override;
};

} //agent
} //solver
} //korali
;
