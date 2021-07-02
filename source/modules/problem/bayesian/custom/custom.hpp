/** \namespace bayesian
* @brief Namespace declaration for modules of type: bayesian.
*/

/** \file
* @brief Header file for module: Custom.
*/

/** \dir problem/bayesian/custom
* @brief Contains code, documentation, and scripts for module: Custom.
*/


#ifndef _KORALI_PROBLEM_BAYESIAN_CUSTOM_
#define _KORALI_PROBLEM_BAYESIAN_CUSTOM_


#include "modules/problem/bayesian/bayesian.hpp"

__startNamespace__;

/**
* @brief Class declaration for module: Custom.
*/
class Custom : public Bayesian
{
  private:
  public: 
  /**
  * @brief Stores the user-defined likelihood model. It should return the value of the Log Likelihood of the given sample.
  */
   std::uint64_t _likelihoodModel;
  
 
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
  

  void evaluateLoglikelihood(korali::Sample &sample) override;
  void evaluateLoglikelihoodGradient(korali::Sample &sample) override;
  void evaluateFisherInformation(korali::Sample &sample) override;
  void initialize() override;
};

__endNamespace__;

#endif // _KORALI_PROBLEM_BAYESIAN_CUSTOM_
