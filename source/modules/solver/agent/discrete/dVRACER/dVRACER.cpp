#include "engine.hpp"
#include "modules/solver/agent/discrete/dVRACER/dVRACER.hpp"
#include "omp.h"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
namespace discrete
{
;

void dVRACER::initializeAgent()
{
  // Initializing common discrete agent configuration
  Discrete::initializeAgent();

  // Init statistics
  _statisticsAverageActionSigmas.resize(_problem->_actionVectorSize);

  /*********************************************************************
   * Initializing Critic/Policy Neural Network Optimization Experiment
   *********************************************************************/
  _criticPolicyLearner.resize(_problem->_policiesPerEnvironment);
  _criticPolicyExperiment.resize(_problem->_policiesPerEnvironment);
  _criticPolicyProblem.resize(_problem->_policiesPerEnvironment);

  for(size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
  {
    _criticPolicyExperiment[p]["Problem"]["Type"] = "Supervised Learning";
    _criticPolicyExperiment[p]["Problem"]["Max Timesteps"] = _timeSequenceLength;
    if(_problem->_policiesPerEnvironment == 1)
      _criticPolicyExperiment[p]["Problem"]["Training Batch Size"] = _miniBatchSize * _problem->_agentsPerEnvironment;
    else
      _criticPolicyExperiment[p]["Problem"]["Training Batch Size"] = _miniBatchSize;
    _criticPolicyExperiment[p]["Problem"]["Inference Batch Size"] = 1;
    _criticPolicyExperiment[p]["Problem"]["Input"]["Size"] = _problem->_stateVectorSize;
    _criticPolicyExperiment[p]["Problem"]["Solution"]["Size"] = 1 + _policyParameterCount; // The value function, action q values, and inverse temperatur

    _criticPolicyExperiment[p]["Solver"]["Type"] = "Learner/DeepSupervisor";
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
    _criticPolicyExperiment[p]["Solver"]["Learning Rate"] = _currentLearningRate;
    _criticPolicyExperiment[p]["Solver"]["Loss Function"] = "Direct Gradient";
    _criticPolicyExperiment[p]["Solver"]["Steps Per Generation"] = 1;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Engine"] = _neuralNetworkEngine;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
    _criticPolicyExperiment[p]["Solver"]["Output Weights Scaling"] = 0.001;

    // No transformations for the state value output
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][0] = 1.0f;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][0] = 0.0f;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][0] = "Identity";

    // No transofrmation for the q values
    for (size_t i = 0; i < _problem->_possibleActions.size(); ++i)
    {
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][i + 1] = 1.0f;
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][i + 1] = 0.0f;
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][i + 1] = "Identity";
    }

    // Transofrmation for the inverse temperature
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][1 + _problem->_possibleActions.size()] = 1.0f;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][1 + _problem->_possibleActions.size()] = 0.0f;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][1 + _problem->_possibleActions.size()] = "Sigmoid";

    // Running initialization to verify that the configuration is correct
    _criticPolicyExperiment[p].initialize();
    _criticPolicyProblem[p] = dynamic_cast<problem::SupervisedLearning *>(_criticPolicyExperiment[p]._problem);
    _criticPolicyLearner[p] = dynamic_cast<solver::learner::DeepSupervisor *>(_criticPolicyExperiment[p]._solver);
  }
}

void dVRACER::trainPolicy()
{
  // Obtaining Minibatch experience ids
  const auto miniBatch = generateMiniBatch(_miniBatchSize);

  // Gathering state sequences for selected minibatch
  const auto stateSequence = getMiniBatchStateSequence(miniBatch);

  // Running policy NN on the Minibatch experiences
  const auto policyInfo = runPolicy(stateSequence);

  // Using policy information to update experience's metadata
  updateExperienceMetadata(miniBatch, policyInfo);

  // Now calculating policy gradients
  calculatePolicyGradients(miniBatch);

  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
  {
    // Updating learning rate for critic/policy learner guided by REFER
    _criticPolicyLearner[p]->_learningRate = _currentLearningRate;

    // Now applying gradients to update policy NN
    _criticPolicyLearner[p]->runGeneration();
  }
}

void dVRACER::calculatePolicyGradients(const std::vector<size_t> &miniBatch)
{
  const size_t miniBatchSize = miniBatch.size();

#pragma omp parallel for
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current experiment
    size_t expId = miniBatch[b];

    // Getting experience policy data
    const auto &expPolicy = _expPolicyVector[expId];
    // Getting current policy data
    const auto &curPolicy = _curPolicyVector[expId];

    // Getting value evaluation
    const std::vector<float> V = _stateValueVector[expId];
    const std::vector<float> expVtbc = _retraceValueVector[expId];

    if (_relationship == "individual")
    {
      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
      {
        // Storage for the update gradient
        std::vector<float> gradientLoss(1 + _policyParameterCount, 0.0f);

        // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
        gradientLoss[0] = expVtbc[d] - V[d];

        // Compute policy gradient only if inside trust region (or offPolicy disabled)
        if (_isOnPolicyVector[expId][d])
        {
          // Qret for terminal state is just reward
          float Qret = getScaledReward(_rewardVector[expId][d], d);

          // If experience is non-terminal, add Vtbc
          if (_terminationVector[expId] == e_nonTerminal)
          {
            float nextExpVtbc = _retraceValueVector[expId + 1][d];
            Qret += _discountFactor * nextExpVtbc;
          }

          // If experience is truncated, add truncated state value
          if (_terminationVector[expId] == e_truncated)
          {
            float nextExpVtbc = _truncatedStateValueVector[expId][d];
            Qret += _discountFactor * nextExpVtbc;
          }

          // Compute Off-Policy Objective (eq. 5)
          float lossOffPolicy = Qret - V[d];

          // Compute Policy Gradient wrt Params
          auto polGrad = calculateImportanceWeightGradient(curPolicy[d], expPolicy[d]);

          // Set Gradient of Loss wrt Params
          for (size_t i = 0; i < _policyParameterCount; i++)
          {
            // '-' because the optimizer is maximizing
            gradientLoss[1 + i] = _experienceReplayOffPolicyREFERBeta[d] * lossOffPolicy * polGrad[i];
          }
        }

        // Compute derivative of kullback-leibler divergence wrt current distribution params
        auto klGrad = calculateKLDivergenceGradient(expPolicy[d], curPolicy[d]);

        for (size_t i = 0; i < _policyParameterCount; i++)
        {
          // Step towards old policy (gradient pointing to larger difference between old and current policy)
          gradientLoss[1 + i] -= (1.0f - _experienceReplayOffPolicyREFERBeta[d]) * klGrad[i];

          if (std::isfinite(gradientLoss[i]) == false)
            KORALI_LOG_ERROR("Gradient loss returned an invalid value: %f\n", gradientLoss[i]);
        }

        // Set Gradient of Loss as Solution
        if (_problem->_policiesPerEnvironment == 1)
          _criticPolicyProblem[0]->_solutionData[b * _problem->_agentsPerEnvironment + d] = gradientLoss;
        else
          _criticPolicyProblem[d]->_solutionData[b] = gradientLoss;
      }
    }
    else if (_relationship== "collaborator")
    {
      //Calculating sum or product terms which will be needed for gradients
      float sumV = 0.0;
      float prodIW= 1.0;
      float sumQret = 0.0;
      float sumNextExpVtbc = 0.0;

      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
      {
        sumV += V[d];
        prodIW *= _importanceWeightVector[expId][d];
        sumQret += getScaledReward(_rewardVector[expId][d], d);
        if (_terminationVector[expId] == e_truncated)
          sumNextExpVtbc += _truncatedStateValueVector[expId][d];
      }

      for (size_t d = 0; d < _problem->_agentsPerEnvironment; d++)
      {
        // Storage for the update gradient
        std::vector<float> gradientLoss(1 + _policyParameterCount, 0.0f);

        // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
        gradientLoss[0] = expVtbc[d] - sumV;

        // Compute policy gradient only if inside trust region (or offPolicy disabled)
        if (_isOnPolicyVector[expId][d])
        {
          // Qret for terminal state is just reward
          float Qret = sumQret;

          // If experience is non-terminal, add Vtbc
          if (_terminationVector[expId] == e_nonTerminal)
          {
            float nextExpVtbc = _retraceValueVector[expId + 1][d];
            Qret += _discountFactor * nextExpVtbc;
          }

          // If experience is truncated, add truncated state value
          if (_terminationVector[expId] == e_truncated)
          {
            Qret += _discountFactor * sumNextExpVtbc;
          }

          // Compute Off-Policy Objective (eq. 12)
          float lossOffPolicy = (Qret - sumV) * prodIW / _importanceWeightVector[expId][d];

          // Compute Policy Gradient wrt Params
          auto polGrad = calculateImportanceWeightGradient(curPolicy[d], expPolicy[d]);

          // Set Gradient of Loss wrt Params
          for (size_t i = 0; i < _policyParameterCount; i++)
          {
            // '-' because the optimizer is maximizing
            gradientLoss[1 + i] = _experienceReplayOffPolicyREFERBeta[d] * lossOffPolicy * polGrad[i];
          }
        }

        // Compute derivative of kullback-leibler divergence wrt current distribution params
        auto klGrad = calculateKLDivergenceGradient(expPolicy[d], curPolicy[d]);

        for (size_t i = 0; i < _policyParameterCount; i++)
        {
          // Step towards old policy (gradient pointing to larger difference between old and current policy)
          gradientLoss[1 + i] -= (1.0f - _experienceReplayOffPolicyREFERBeta[d]) * klGrad[i];

          if (std::isfinite(gradientLoss[i]) == false)
            KORALI_LOG_ERROR("Gradient loss returned an invalid value: %f\n", gradientLoss[i]);
        }

        _criticPolicyProblem[0]->_solutionData[b * _problem->_agentsPerEnvironment + d] = gradientLoss;
      }

    }
    else
      KORALI_LOG_ERROR("Defined Relationship: %s is neither individual nor collaborator \n", _relationship);
  }

  // Compute average action stadard deviation
  for (size_t j = 0; j < _problem->_actionVectorSize; j++) _statisticsAverageActionSigmas[j] /= (float)miniBatchSize;
}

std::vector<policy_t> dVRACER::runPolicy(const std::vector<std::vector<std::vector<float>>> &stateBatch)
{
  // Getting batch size
  size_t batchSize = stateBatch.size();

  if (_problem->_policiesPerEnvironment == 1)
  {
    // Storage for policy
    std::vector<policy_t> policyVector(batchSize);

    // Forward the neural network for this state
    const auto evaluation = _criticPolicyLearner[0]->getEvaluation(stateBatch);

  #pragma omp parallel for
    for (size_t b = 0; b < batchSize; b++)
    {
      // Getting state value
      policyVector[b].stateValue = evaluation[b][0];

      // Storage for action probabilities
      float maxq = -korali::Inf;
      std::vector<float> qValAndInvTemp(_policyParameterCount);
      std::vector<float> pActions(_problem->_possibleActions.size());

      // Get the inverse of the temperature for the softmax distribution
      const float invTemperature = evaluation[b][_policyParameterCount];

      // Iterating all Q(s,a)
      for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
      {
        // Computing Q(s,a_i)
        qValAndInvTemp[i] = evaluation[b][1 + i];

        // Extracting max Q(s,a_i)
        if (qValAndInvTemp[i] > maxq) maxq = qValAndInvTemp[i];
      }

      // Storage for the cumulative e^Q(s,a_i)/maxq
      float sumExpQVal = 0.0;

      for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
      {
        // Computing e^(Q(s,a_i) - maxq)
        float expCurQVal = std::exp(invTemperature * (qValAndInvTemp[i] - maxq));

        // Computing Sum_i(e^Q(s,a_i)/e^maxq)
        sumExpQVal += expCurQVal;

        // Storing partial value of the probability of the action
        pActions[i] = expCurQVal;
      }

      // Calculating inverse of Sum_i(e^Q(s,a_i))
      float invSumExpQVal = 1.0f / sumExpQVal;

      // Normalizing action probabilities
      for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
        pActions[i] *= invSumExpQVal;

      // Set inverse temperature parameter
      qValAndInvTemp[_problem->_possibleActions.size()] = invTemperature;

      // Storing the action probabilities into the policy
      policyVector[b].actionProbabilities = pActions;
      policyVector[b].distributionParameters = qValAndInvTemp;
    }

    return policyVector;

  }
  else if ((_problem->_policiesPerEnvironment != 1) & (batchSize==1)) //TODO:CHECK with P&D,since it is unclear which policy is needed when runPolicy is called in continuous.cpp.base, I feed it through all NN and decide in cont. which one to take
  {
    // Storage for policy
    std::vector<policy_t> policyVector(_problem->_policiesPerEnvironment);
    for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    {
      const auto evaluation = _criticPolicyLearner[p]->getEvaluation(stateBatch);

      // Getting state value
      policyVector[p].stateValue = evaluation[0][0];

      // Storage for action probabilities
      float maxq = -korali::Inf;
      std::vector<float> qValAndInvTemp(_policyParameterCount);
      std::vector<float> pActions(_problem->_possibleActions.size());

      // Get the inverse of the temperature for the softmax distribution
      const float invTemperature = evaluation[0][_policyParameterCount];

      // Iterating all Q(s,a)
      for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
      {
        // Computing Q(s,a_i)
        qValAndInvTemp[i] = evaluation[0][1 + i];

        // Extracting max Q(s,a_i)
        if (qValAndInvTemp[i] > maxq) maxq = qValAndInvTemp[i];
      }

      // Storage for the cumulative e^Q(s,a_i)/maxq
      float sumExpQVal = 0.0;

      for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
      {
        // Computing e^(Q(s,a_i) - maxq)
        float expCurQVal = std::exp(invTemperature * (qValAndInvTemp[i] - maxq));

        // Computing Sum_i(e^Q(s,a_i)/e^maxq)
        sumExpQVal += expCurQVal;

        // Storing partial value of the probability of the action
        pActions[i] = expCurQVal;
      }

      // Calculating inverse of Sum_i(e^Q(s,a_i))
      float invSumExpQVal = 1.0f / sumExpQVal;

      // Normalizing action probabilities
      for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
        pActions[i] *= invSumExpQVal;

      // Set inverse temperature parameter
      qValAndInvTemp[_problem->_possibleActions.size()] = invTemperature;

      // Storing the action probabilities into the policy
      policyVector[p].actionProbabilities = pActions;
      policyVector[p].distributionParameters = qValAndInvTemp;

    }
    return policyVector;
  }
  else
  {
    // Storage for policy
    std::vector<policy_t> policyVector(batchSize);

    // Storage for each sample for all Policies
    std::vector<std::vector<std::vector<std::vector<float>>>> stateBatchDistributed(_problem->_policiesPerEnvironment);
    for (size_t b = 0; b < batchSize; b++)
    {
      stateBatchDistributed[b % _problem->_agentsPerEnvironment].push_back(stateBatch[b]);
    }

    for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    {
      const auto evaluation = _criticPolicyLearner[p]->getEvaluation(stateBatchDistributed[p]);
    #pragma omp parallel for
      for (size_t b = 0; b < evaluation.size(); b++)
      {
        // Getting state value
        policyVector[b * _problem->_agentsPerEnvironment + p].stateValue = evaluation[b][0];

        // Storage for action probabilities
        float maxq = -korali::Inf;
        std::vector<float> qValAndInvTemp(_policyParameterCount);
        std::vector<float> pActions(_problem->_possibleActions.size());

        // Get the inverse of the temperature for the softmax distribution
        const float invTemperature = evaluation[b][_policyParameterCount];

        // Iterating all Q(s,a)
        for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
        {
          // Computing Q(s,a_i)
          qValAndInvTemp[i] = evaluation[b][1 + i];

          // Extracting max Q(s,a_i)
          if (qValAndInvTemp[i] > maxq) maxq = qValAndInvTemp[i];
        }

        // Storage for the cumulative e^Q(s,a_i)/maxq
        float sumExpQVal = 0.0;

        for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
        {
          // Computing e^(Q(s,a_i) - maxq)
          float expCurQVal = std::exp(invTemperature * (qValAndInvTemp[i] - maxq));

          // Computing Sum_i(e^Q(s,a_i)/e^maxq)
          sumExpQVal += expCurQVal;

          // Storing partial value of the probability of the action
          pActions[i] = expCurQVal;
        }

        // Calculating inverse of Sum_i(e^Q(s,a_i))
        float invSumExpQVal = 1.0f / sumExpQVal;

        // Normalizing action probabilities
        for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
          pActions[i] *= invSumExpQVal;

        // Set inverse temperature parameter
        qValAndInvTemp[_problem->_possibleActions.size()] = invTemperature;

        // Storing the action probabilities into the policy
        policyVector[b * _problem->_agentsPerEnvironment + p].actionProbabilities = pActions;
        policyVector[b * _problem->_agentsPerEnvironment + p].distributionParameters = qValAndInvTemp;

      }
    }
    return policyVector;
  }
}

knlohmann::json dVRACER::getAgentPolicy()
{
  knlohmann::json hyperparameters;
  for(size_t p = 0; p < _problem->_policiesPerEnvironment;p++)
    hyperparameters["Policy Hyperparameters"][p]= _criticPolicyLearner[p]->getHyperparameters();
  return hyperparameters;
}

void dVRACER::setAgentPolicy(const knlohmann::json &hyperparameters)
{
  for(size_t p = 0; p < _problem->_policiesPerEnvironment;p++)
    _criticPolicyLearner[p]->setHyperparameters(hyperparameters[p].get<std::vector<float>>());
}

void dVRACER::printAgentInformation()
{
  _k->_logger->logInfo("Normal", " + [dVRACER] Policy Learning Rate: %.3e\n", _currentLearningRate);
}

void dVRACER::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Statistics", "Average Action Sigmas"))
 {
 try { _statisticsAverageActionSigmas = js["Statistics"]["Average Action Sigmas"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ dVRACER ] \n + Key:    ['Statistics']['Average Action Sigmas']\n%s", e.what()); } 
   eraseValue(js, "Statistics", "Average Action Sigmas");
 }

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Exploration Noise"))
 {
 try { _k->_variables[i]->_initialExplorationNoise = _k->_js["Variables"][i]["Initial Exploration Noise"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ dVRACER ] \n + Key:    ['Initial Exploration Noise']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Exploration Noise");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Exploration Noise'] required by dVRACER.\n"); 

 } 
 Discrete::setConfiguration(js);
 _type = "agent/discrete/dVRACER";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: dVRACER: \n%s\n", js.dump(2).c_str());
} 

void dVRACER::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Statistics"]["Average Action Sigmas"] = _statisticsAverageActionSigmas;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Initial Exploration Noise"] = _k->_variables[i]->_initialExplorationNoise;
 } 
 Discrete::getConfiguration(js);
} 

void dVRACER::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Discrete::applyModuleDefaults(js);
} 

void dVRACER::applyVariableDefaults() 
{

 std::string defaultString = "{\"Initial Exploration Noise\": -1.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Discrete::applyVariableDefaults();
} 

bool dVRACER::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Discrete::checkTermination();
 return hasFinished;
}

;

} //discrete
} //agent
} //solver
} //korali
;
