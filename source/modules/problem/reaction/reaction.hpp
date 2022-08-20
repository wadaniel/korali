/** \namespace problem
* @brief Namespace declaration for modules of type: problem.
*/

/** \file
* @brief Header file for module: Reaction.
*/

/** \dir problem/reaction
* @brief Contains code, documentation, and scripts for module: Reaction.
*/

#pragma once

#include "auxiliar/reactionParser.hpp"
#include "modules/problem/problem.hpp"

namespace korali
{
namespace problem
{
;

/**
 * @brief Structure to store reaction information
 */
struct reaction_t
{
  /**
   * @brief The rate of the reaction.
   */
  double rate;

  /**
   * @brief TODO
   */
  std::vector<int> reactantIds;

  /**
   * @brief TODO
   */
  std::vector<int> reactantStoichiometries;

  /**
   * @brief TODO
   */
  std::vector<int> productIds;

  /**
   * @brief TODO
   */
  std::vector<int> productStoichiometries;

  /**
   * @brief TODO
   */
  std::vector<bool> isReactantReservoir = {};

  /**
   * @brief Constructor for type reaction_t.
   */
  reaction_t(double rate,
             std::vector<int> reactantIds,
             std::vector<int> reactantSCs,
             std::vector<int> productIds,
             std::vector<int> productSCs,
             std::vector<bool> isReactantReservoir)
    : rate(rate), reactantIds(std::move(reactantIds)), reactantStoichiometries(std::move(reactantSCs)), productIds(std::move(productIds)), productStoichiometries(std::move(productSCs)), isReactantReservoir(std::move(isReactantReservoir))
  {
    if (this->reactantIds.size() > 0 && this->isReactantReservoir.size() == 0)
    {
      this->isReactantReservoir.resize(this->reactantIds.size());
      this->isReactantReservoir.assign(this->reactantIds.size(), false);
    }
  }
};

/**
* @brief Class declaration for module: Reaction.
*/
class Reaction : public Problem
{
  public: 
  /**
  * @brief [Internal Use] Complete description of all reactions.
  */
   knlohmann::json _reactions;
  /**
  * @brief [Internal Use] Maps the reactants name to an internal index.
  */
   std::map<std::string, int> _reactantNameToIndexMap;
  /**
  * @brief [Internal Use] Maps the reactants name to an internal index.
  */
   std::vector<int> _initialReactantNumbers;
  
 
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
  

  std::vector<reaction_t> _reactionVector;

  void initialize() override;

  double computePropensity(size_t reactionIndex, const std::vector<int> &reactantNumbers) const;
  double calculateMaximumAllowedFirings(size_t reactionIndex, const std::vector<int> &reactantNumbers) const;

  void applyChanges(size_t reactionIndex, std::vector<int> &reactantNumbers, int numFirings = 1) const;
};

} //problem
} //korali
;
