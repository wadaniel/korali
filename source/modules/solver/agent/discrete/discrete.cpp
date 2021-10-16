#include "engine.hpp"
#include "modules/solver/agent/discrete/discrete.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
;

void Discrete::initializeAgent()
{
  // Getting discrete problem pointer
  _problem = dynamic_cast<problem::reinforcementLearning::Discrete *>(_k->_problem);
}
void Discrete::getAction(korali::Sample &sample)
{
  // Get action for all the agents in the environment

  // Getting current state

  for (size_t i = 0; i < _problem->_agentsPerEnvironment; i++)
  {
    // Getting current state
    auto state = sample["State"][i].get<std::vector<float>>();

    _stateTimeSequence.add(state);

    // Getting the probability of the actions given by the agent's policy
    auto policy = runPolicy({_stateTimeSequence.getVector()})[0];
    const auto &pActions = policy.distributionParameters;

    // Storage for the action index to use
    size_t actionIdx = 0;

    /*****************************************************************************
     * During training, we follow the Epsilon-greedy strategy. Choose, given a
     * probability (pEpsilon), one from the following:
     *  - Uniformly random action among all possible actions
     *  - Sample action guided by the policy's probability distribution
     ****************************************************************************/

    if (sample["Mode"] == "Training")
    {
      // Getting pGreedy = U[0,1] for the epsilon-greedy strategy
      float pEpsilon = _uniformGenerator->getRandomNumber();

      // Producing random (uniform) number for the selection of the action
      float x = _uniformGenerator->getRandomNumber();

      // If p < e, then we choose the action randomly, with a uniform probability, among all possible actions.
      if (pEpsilon < _randomActionProbability)
      {
        actionIdx = floor(x * _problem->_possibleActions.size());
      }
      else // else we select guided by the policy's probability distribution
      {
        // Categorical action sampled from action probabilites (from ACER paper [Wang2017])
        float curSum = 0.0;
        for (actionIdx = 0; actionIdx < pActions.size() - 1; actionIdx++)
        {
          curSum += pActions[actionIdx];
          if (x < curSum) break;
        }

        // NOTE: In original DQN paper [Minh2015] we choose max
        // actionIdx = std::distance(pActions.begin(), std::max_element(pActions.begin(), pActions.end()));
      }
    }

    /*****************************************************************************
     * During testing, we just select the action with the largest probability
     * given by the policy.
     ****************************************************************************/

    // Finding the best action index from the probabilities
    if (sample["Mode"] == "Testing")
      actionIdx = std::distance(pActions.begin(), std::max_element(pActions.begin(), pActions.end()));

    /*****************************************************************************
     * Storing the action itself
     ****************************************************************************/

    // Storing action itself, its idx, and probabilities
    sample["Policy"]["Distribution Parameters"][i] = pActions;
    sample["Policy"]["Action Index"][i] = actionIdx;
    sample["Action"][i] = _problem->_possibleActions[actionIdx];
    sample["Policy"]["State Value"][i] = policy.stateValue;
  }
}

float Discrete::calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy)
{
  const auto &pVectorCurPolicy = curPolicy.distributionParameters;
  const auto &pVectorOldPolicy = oldPolicy.distributionParameters;
  auto actionIdx = oldPolicy.actionIndex;

  // Getting probability density of action for current policy
  float pCurPolicy = pVectorCurPolicy[actionIdx];

  // Getting probability density of action for old policy
  float pOldPolicy = pVectorOldPolicy[actionIdx];

  // Now calculating importance weight for the old s,a experience
  float constexpr epsilon = 0.00000001f;
  float importanceWeight = pCurPolicy / (pOldPolicy + epsilon);

  // Safety checks
  if (importanceWeight > 1024.0f) importanceWeight = 1024.0f;
  if (importanceWeight < -1024.0f) importanceWeight = -1024.0f;

  return importanceWeight;
}

std::vector<float> Discrete::calculateImportanceWeightGradient(const size_t actionIdx, const std::vector<float> &curPvals, const std::vector<float> &oldPvals)
{
  std::vector<float> grad(_problem->_possibleActions.size(), 0.0);

  float importanceWeight = curPvals[actionIdx] / oldPvals[actionIdx];

  // Safety checks
  if (importanceWeight > 1024.0f) importanceWeight = 1024.0f;
  if (importanceWeight < -1024.0f) importanceWeight = -1024.0f;

  // calculate gradient of categorical distribution normalized by old pvals
  for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
  {
    if (i == actionIdx)
      grad[i] = importanceWeight * (1. - curPvals[i]);
    else
      grad[i] = -importanceWeight * curPvals[i];
  }

  return grad;
}

std::vector<float> Discrete::calculateKLDivergenceGradient(const std::vector<float> &oldPvalues, const std::vector<float> &curPvalues)
{
  std::vector<float> klGrad(_problem->_possibleActions.size(), 0.0);

  // Gradient wrt NN output i
  for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
  {
    // Iterate over all pvalues
    for (size_t j = 0; j < _problem->_possibleActions.size(); j++)
    {
      if (i == j)
        klGrad[i] -= oldPvalues[j] * (1.0 - curPvalues[i]);
      else
        klGrad[i] += oldPvalues[j] * curPvalues[i];
    }
  }

  return klGrad;
}

void Discrete::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Random Action Probability"))
 {
 try { _randomActionProbability = js["Random Action Probability"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ discrete ] \n + Key:    ['Random Action Probability']\n%s", e.what()); } 
   eraseValue(js, "Random Action Probability");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Random Action Probability'] required by discrete.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Agent::setConfiguration(js);
 _type = "agent/discrete";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: discrete: \n%s\n", js.dump(2).c_str());
} 

void Discrete::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Random Action Probability"] = _randomActionProbability;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Agent::getConfiguration(js);
} 

void Discrete::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Random Action Probability\": 0.05}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Agent::applyModuleDefaults(js);
} 

void Discrete::applyVariableDefaults() 
{

 Agent::applyVariableDefaults();
} 

bool Discrete::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Agent::checkTermination();
 return hasFinished;
}

;

} //agent
} //solver
} //korali
;
