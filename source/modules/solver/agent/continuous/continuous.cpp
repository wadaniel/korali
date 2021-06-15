#include "engine.hpp"
#include "modules/solver/agent/continuous/continuous.hpp"
#include "sample/sample.hpp"

#include <gsl/gsl_sf_psi.h>

namespace korali
{
namespace solver
{
namespace agent
{


void Continuous::initializeAgent()
{
  // Getting continuous problem pointer
  _problem = dynamic_cast<problem::reinforcementLearning::Continuous *>(_k->_problem);

  // Obtaining action shift and scales for bounded distributions
  _actionShifts.resize(_problem->_actionVectorSize);
  _actionScales.resize(_problem->_actionVectorSize);
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    // For bounded distributions, infinite bounds should result in an error message
    if (_policyDistribution == "Squashed Normal" || _policyDistribution == "Beta" || _policyDistribution == "Clipped Normal" || _policyDistribution == "Truncated Normal")
    {
      if (isfinite(_actionLowerBounds[i]) == false)
        KORALI_LOG_ERROR("Provided lower bound (%f) for action variable %lu is non-finite, but the distribution (%s) is bounded.\n", _actionLowerBounds[i], i, _policyDistribution.c_str());

      if (isfinite(_actionUpperBounds[i]) == false)
        KORALI_LOG_ERROR("Provided upper bound (%f) for action variable %lu is non-finite, but the distribution (%s) is bounded.\n", _actionUpperBounds[i], i, _policyDistribution.c_str());

      _actionShifts[i] = (_actionUpperBounds[i] + _actionLowerBounds[i]) * 0.5f;
      _actionScales[i] = (_actionUpperBounds[i] - _actionLowerBounds[i]) * 0.5f;
    }
  }

  // Obtaining policy parameter transformations (depends on which policy distribution chosen)
  if (_policyDistribution == "Normal" || _policyDistribution == "Squashed Normal" || _policyDistribution == "Clipped Normal" || _policyDistribution == "Truncated Normal")
  {
    _policyParameterCount = 2 * _problem->_actionVectorSize; // Means and Sigmas

    // Allocating space for the required transformations
    _policyParameterTransformationMasks.resize(_policyParameterCount);
    _policyParameterScaling.resize(_policyParameterCount);
    _policyParameterShifting.resize(_policyParameterCount);

    // Establishing transformations for the Normal policy
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      auto varIdx = _problem->_actionVectorIndexes[i];
      float sigma = _k->_variables[varIdx]->_initialExplorationNoise;

      // Checking correct noise configuration
      if (sigma <= 0.0f) KORALI_LOG_ERROR("Provided initial noise (%f) for action variable %lu is not defined or negative.\n", sigma, varIdx);

      // Identity mask for Means
      _policyParameterScaling[i] = 1.0f;
      _policyParameterShifting[i] = 0.0f;
      _policyParameterTransformationMasks[i] = "Identity";

      // Softplus mask for Sigmas
      _policyParameterTransformationMasks[_problem->_actionVectorSize + i] = "Softplus";
      _policyParameterScaling[_problem->_actionVectorSize + i] = sigma * 2.0f;
      _policyParameterShifting[_problem->_actionVectorSize + i] = 0.0f;
    }
  }

  if (_policyDistribution == "Beta")
  {
    _policyParameterCount = 2 * _problem->_actionVectorSize; // Mu and Variance

    // Allocating space for the required transformations
    _policyParameterTransformationMasks.resize(_policyParameterCount);
    _policyParameterScaling.resize(_policyParameterCount);
    _policyParameterShifting.resize(_policyParameterCount);

    // Establishing transformations for the Normal policy
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      auto varIdx = _problem->_actionVectorIndexes[i];
      float variance = _k->_variables[varIdx]->_initialExplorationNoise;

      // Checking correct noise configuration
      if (variance <= 0.0f) KORALI_LOG_ERROR("Provided initial noise (%f) for action variable %lu is not defined or negative.\n", variance, varIdx);

      // Identity mask for Means
      _policyParameterScaling[i] = 1.0f;
      _policyParameterShifting[i] = 0.0f;
      _policyParameterTransformationMasks[i] = "Identity";

      // Sigmoid Mask for Variance
      _policyParameterTransformationMasks[_problem->_actionVectorSize + i] = "Sigmoid";
      _policyParameterScaling[_problem->_actionVectorSize + i] = variance * 2.0f;
      _policyParameterShifting[_problem->_actionVectorSize + i] = 0.0f;
    }
  }
}

void Continuous::getAction(korali::Sample &sample)
{
  // Getting current state
  auto state = sample["State"];

  // Adding state to the state time sequence
  _stateTimeSequence.add(state);

  // Storage for the action to select
  std::vector<float> action(_problem->_actionVectorSize);

  // Forward state sequence to get the Gaussian means and sigmas from policy
  auto policy = runPolicy({_stateTimeSequence.getVector()})[0];

  /*****************************************************************************
  * During Training we select action according to policy's probability
  * distribution
  ****************************************************************************/

  if (sample["Mode"] == "Training") action = generateTrainingAction(policy);

  /*****************************************************************************
  * During testing, we select the means (point of highest density) for all
  * elements of the action vector
  ****************************************************************************/

  if (sample["Mode"] == "Testing") action = generateTestingAction(policy);

  /*****************************************************************************
  * Storing the action and its policy
  ****************************************************************************/

  sample["Policy"]["Distribution Parameters"] = policy.distributionParameters;
  sample["Policy"]["State Value"] = policy.stateValue;
  sample["Policy"]["Unbounded Action"] = policy.unboundedAction;
  sample["Action"] = action;
}

std::vector<float> Continuous::generateTrainingAction(policy_t &curPolicy)
{
  std::vector<float> action(_problem->_actionVectorSize);

  // Creating the action based on the selected policy
  if (_policyDistribution == "Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mean = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      action[i] = mean + sigma * _normalGenerator->getRandomNumber();
    }
  }

  if (_policyDistribution == "Squashed Normal")
  {
    std::vector<float> unboundedAction(_problem->_actionVectorSize);
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mean = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float scale = _actionScales[i]; // verify if half the domain width is correct (DW)
      const float shift = _actionShifts[i];
      
      unboundedAction[i] = mean + sigma * _normalGenerator->getRandomNumber();
      action[i] = (std::tanh(unboundedAction[i]) * scale) + shift;
    }
    curPolicy.unboundedAction = unboundedAction;
  }
  
  if (_policyDistribution == "Clipped Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mu = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      action[i] = mu + sigma * _normalGenerator->getRandomNumber();
      if (action[i] >= _actionUpperBounds[i]) action[i] = _actionUpperBounds[i];
      if (action[i] <= _actionLowerBounds[i]) action[i] = _actionLowerBounds[i];
    }
  }
 
  if (_policyDistribution == "Truncated Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mu = curPolicy.distributionParameters[i];
      const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      
      const float alpha = (_actionLowerBounds[i]-mu)/sigma;
      const float beta = (_actionUpperBounds[i]-mu)/sigma;
      
      // Sampling via inverse sampling
      const float u = _uniformGenerator->getRandomNumber();
      const float x = u*normalCDF(beta, mu, sigma) + (1.-u)*normalCDF(alpha, mu, sigma);
      action[i] = mu + 2*ierf( 2.*x - 1. )*sigma;
 
      // Keep this for the moment (D.W.)
      assert(action[i] <= _actionUpperBounds[i]);
      assert(action[i] >= _actionLowerBounds[i]);
    }
  }



  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      action[i] = ranBetaAlt(_normalGenerator->_range, curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
    }
  }

  return action;
}

std::vector<float> Continuous::generateTestingAction(const policy_t &curPolicy)
{
  std::vector<float> action(_problem->_actionVectorSize);

  if (_policyDistribution == "Normal")
  {
    // Take only the means without noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
      action[i] = curPolicy.distributionParameters[i];
  }

  if (_policyDistribution == "Squashed Normal")
  {
    // Take only the transformed means without noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      const float mean = curPolicy.distributionParameters[i];
      const float scale = _actionScales[i]; // verify if half the domain width is correct (DW)
      const float shift = _actionShifts[i];
      action[i] = (std::tanh(mean) * scale) + shift;
    }
  }
 
  if (_policyDistribution == "Clipped Normal")
  {
    // Take only the modes of the Clipped Normal without noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      action[i] = curPolicy.distributionParameters[i];
  
      // In case the 'mode' is outside of the bounds
      if (action[i] >= _actionUpperBounds[i]) action[i] = _actionUpperBounds[i];
      if (action[i] <= _actionLowerBounds[i]) action[i] = _actionLowerBounds[i];
    }
  }
 
  if (_policyDistribution == "Truncated Normal")
  {
    // Take only the modes of the Truncated Normal noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      action[i] = curPolicy.distributionParameters[i];
  
      // In case the 'mode' is outside of the bounds
      if (action[i] >= _actionUpperBounds[i]) action[i] = _actionUpperBounds[i];
      if (action[i] <= _actionLowerBounds[i]) action[i] = _actionLowerBounds[i];
    }
  }

  if (_policyDistribution == "Beta")
  {
    // Take only the modes without noise
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      action[i] = _actionLowerBounds[i] + 2.*_actionScales[i] * curPolicy.distributionParameters[i];
    }
  }

  return action;
}

float Continuous::calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy)
{
  float logpCurPolicy = 0.0f;
  float logpOldPolicy = 0.0f;

  // Storage for importance weight value
  float importanceWeight = 0.0f;

  if (_policyDistribution == "Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      logpCurPolicy += normalLogDensity(action[i], curMean, curSigma);
      logpOldPolicy += normalLogDensity(action[i], oldMean, oldSigma);
    }
  }

  if (_policyDistribution == "Squashed Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      // Importance weight of squashed normal is the importance weight of normal evaluated at unbounded action
      logpCurPolicy += normalLogDensity(oldPolicy.unboundedAction[i], curMean, curSigma);
      logpOldPolicy += normalLogDensity(oldPolicy.unboundedAction[i], oldMean, oldSigma);
    }
  }
 
  if (_policyDistribution == "Clipped Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      if( action[i] <= _actionLowerBounds[i] )
      {
        logpCurPolicy += normalLogCDF(_actionLowerBounds[i], curMu, curSigma);
        logpOldPolicy += normalLogCDF(_actionLowerBounds[i], oldMu, oldSigma);
      }
      else if (_actionUpperBounds[i] <= action[i] )
      {
        logpCurPolicy += normalLogCCDF(_actionUpperBounds[i], curMu, curSigma);
        logpOldPolicy += normalLogCCDF(_actionUpperBounds[i], oldMu, oldSigma);
      }
      else
      {
        logpCurPolicy += normalLogDensity(action[i], curMu, curSigma);
        logpOldPolicy += normalLogDensity(action[i], oldMu, oldSigma);
      }
    }
  }
  
  if (_policyDistribution == "Truncated Normal")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      // Normalization constants
      const float Cq = 1./(normalCDF(_actionUpperBounds[i], curMu, curSigma));
      const float Cp = 1./(normalCDF(_actionUpperBounds[i], oldMu, oldSigma));
        
      logpCurPolicy += std::log(Cq/curSigma)-(action[i]-curMu)*(action[i]-curMu)/(2.*curSigma*curSigma);
      logpOldPolicy += std::log(Cp/oldSigma)-(action[i]-oldMu)*(action[i]-oldMu)/(2.*oldSigma*oldSigma);
    }
  }


  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < action.size(); i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldVariance = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      logpCurPolicy += betaLogDensityAlt(action[i], curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
      logpOldPolicy += betaLogDensityAlt(action[i], oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
    }
  }

  // Calculating log importance weight
  float logImportanceWeight = logpCurPolicy - logpOldPolicy;

  // Normalizing extreme values to prevent loss of precision
  if (logImportanceWeight > +7.0f) logImportanceWeight = +7.0f;
  if (logImportanceWeight < -7.0f) logImportanceWeight = -7.0f;

  // Calculating actual importance weight by exp
  importanceWeight = std::exp(logImportanceWeight);

  return importanceWeight;
}

std::vector<float> Continuous::calculateImportanceWeightGradient(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy)
{
  // Storage for importance weight gradients
  std::vector<float> importanceWeightGradients(2 * _problem->_actionVectorSize, 0.0);

  if (_policyDistribution == "Normal")
  {
    float logpCurPolicy = 0.0f;
    float logpOldPolicy = 0.0f;

    // ParamsOne are the Means, ParamsTwo are the Sigmas
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      // Calculate importance weight
      logpCurPolicy += normalLogDensity(action[i], curMean, curSigma);
      logpOldPolicy += normalLogDensity(action[i], oldMean, oldSigma);

      // Deviation from expAction and current Mean
      float curActionDiff = (action[i] - curMean);

      // Inverse Variances
      float curInvVar = 1. / (curSigma * curSigma);

      // Gradient with respect to Mean
      importanceWeightGradients[i] = curActionDiff * curInvVar;

      // Gradient with respect to Sigma
      importanceWeightGradients[_problem->_actionVectorSize + i] = (curActionDiff * curActionDiff) * (curInvVar / curSigma) - 1.0f / curSigma;
    }

    float logImportanceWeight = logpCurPolicy - logpOldPolicy;
    float importanceWeight = std::exp(logImportanceWeight);

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
      importanceWeightGradients[i] *= importanceWeight;
  }

  if (_policyDistribution == "Squashed Normal")
  {
    float logpCurPolicy = 0.0f;
    float logpOldPolicy = 0.0f;

    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float scale = _actionScales[i]; // verify if half the domain width is correct (DW)
      const float shift = _actionShifts[i];
      const float unboundedAction = std::atanh((action[i] - shift) / scale);

      // Importance weight of squashed normal is the importance weight of normal evaluated at unbounded action
      logpCurPolicy += normalLogDensity(unboundedAction, curMu, curSigma);
      logpOldPolicy += normalLogDensity(unboundedAction, oldMu, oldSigma);

      // Deviation from expAction and current Mean
      const float curActionDiff = (unboundedAction - curMu);

      // Inverse Variances
      const float curInvVar = 1. / (curSigma * curSigma);

      // Gradient with respect to Mean
      importanceWeightGradients[i] = curActionDiff * curInvVar;

      // Gradient with respect to Sigma
      importanceWeightGradients[_problem->_actionVectorSize + i] = (curActionDiff * curActionDiff) * (curInvVar / curSigma) - 1.0f / curSigma;
    }

    const float logImportanceWeight = logpCurPolicy - logpOldPolicy;
    const float importanceWeight = std::exp(logImportanceWeight);

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
      importanceWeightGradients[i] *= importanceWeight;
  }
 
  if (_policyDistribution == "Clipped Normal")
  {
    float logpCurPolicy = 0.0f;
    float logpOldPolicy = 0.0f;

    // ParamsOne are the Means, ParamsTwo are the Sigmas
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curVar = curSigma*curSigma;
 
      if(action[i] <= _actionLowerBounds[i])
      {
        const float curNormalCdfLower = normalCDF(_actionLowerBounds[i], curMu, curSigma);

        // Grad wrt. curMu
        importanceWeightGradients[i] = -curSigma*std::exp(normalLogDensity(_actionLowerBounds[i], curMu, curSigma))/curNormalCdfLower;

        // Some constants
        const double k1 = (curMu*curMu-curVar)/curVar*std::exp(normalLogDensity(_actionLowerBounds[i], curMu, curSigma));
        const double k2 = curMu*curSigma*M_SQRT2/std::sqrt(M_PI)*std::exp(-(_actionLowerBounds[i]-curMu)*(_actionLowerBounds[i]-curMu)/(2.*curVar));
        const double k3 = std::sqrt(M_PI)/M_SQRT2*std::erff(_actionLowerBounds[i]/(M_SQRT2*curSigma))-_actionLowerBounds[i]*curVar*std::exp(-_actionLowerBounds[i]*_actionLowerBounds[i]/(2.*curVar));
 
        // Grad wrt. curSigma
        importanceWeightGradients[_problem->_actionVectorSize+i] = (k1+k2+k3)/curNormalCdfLower;
        printf("k: %f %f %f\n", k1, k2, k3);
        printf("NCDF: %f %f %f %f %f %f\n", curNormalCdfLower, _actionLowerBounds[i], curMu, curSigma, oldMu, oldSigma);
        printf("iwg: %f %f\n", importanceWeightGradients[i], importanceWeightGradients[i+1] );

        // Calculate importance weight
        logpCurPolicy += normalLogCDF(_actionLowerBounds[i], curMu, curSigma);
        logpOldPolicy += normalLogCDF(_actionLowerBounds[i], oldMu, oldSigma);
 
      }
      else if(_actionUpperBounds[i] <= action[i])
      {
        const float curNormalPdfUpper = std::exp(normalCDF(_actionUpperBounds[i], curMu, curSigma));
        const float curNormalCCdfUpper = normalCCDF(_actionUpperBounds[i], curMu, curSigma);
        
        // Grad wrt. curMu
        importanceWeightGradients[i] = curSigma*curNormalPdfUpper/curNormalCCdfUpper;

        // Some constants
        const float k1 = (curMu*curMu-curVar)/curVar*curNormalPdfUpper;
        const float k2 = curMu*curSigma*M_SQRT2/std::sqrt(M_PI)*std::exp(-(_actionUpperBounds[i]-curMu)*(_actionUpperBounds[i]-curMu)/(2.*curVar));
        const float k3 = std::sqrt(M_PI)/M_SQRT2*std::erff(_actionUpperBounds[i]/(M_SQRT2*curSigma))-_actionUpperBounds[i]*curVar*std::exp(-_actionUpperBounds[i]*_actionUpperBounds[i]/(2.*curVar));
        
        // Grad wrt. curSigma
        importanceWeightGradients[_problem->_actionVectorSize+i] = -(k1+k2+k3)/curNormalCCdfUpper;
        
        printf("k: %f %f %f\n", k1, k2, k3);
        printf("NCDF: %f %f %f %f %f %f\n", normalCDF(_actionUpperBounds[i], curMu, curSigma), _actionUpperBounds[i], curMu, curSigma, oldMu, oldSigma);
        printf("iwg: %f %f\n", importanceWeightGradients[i], importanceWeightGradients[i+1] );
        printf("lp: %f %f\n", normalLogCCDF(_actionUpperBounds[i], curMu,curSigma), normalCCDF(_actionUpperBounds[i], oldMu, oldSigma));

        // Calculate importance weight
        logpCurPolicy += normalLogCCDF(_actionUpperBounds[i], curMu, curSigma);
        logpOldPolicy += normalLogCCDF(_actionUpperBounds[i], oldMu, oldSigma);
      }
      else
      {
        // Grad wrt. curMu
        importanceWeightGradients[i] = (action[i]-curMu)/curSigma;
        
        // Grad wrt. curSigma
        importanceWeightGradients[_problem->_actionVectorSize+i] = ((action[i]-curMu)*(action[i]-curMu)-curVar)/(oldSigma*oldSigma*oldSigma);
        printf("iwg: %f %f\n", importanceWeightGradients[i], importanceWeightGradients[i+1] );

        // Calculate importance weight
        logpCurPolicy += normalLogDensity(action[i], curMu, curSigma);
        logpOldPolicy += normalLogDensity(action[i], oldMu, oldSigma);

      }
    }

    const float logImportanceWeight = logpCurPolicy - logpOldPolicy;
    const float importanceWeight = std::exp(logImportanceWeight);
    printf("iw: %f\n", importanceWeight);

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
    {
      importanceWeightGradients[i] *= importanceWeight;

      const float p1 = oldPolicy.distributionParameters[i];
      const float p2 = curPolicy.distributionParameters[i];
 
      printf("iwgrad[%zu]: %f (%f %f)\n", i, importanceWeightGradients[i], p1, p2);
    }

  }
 
 if (_policyDistribution == "Truncated Normal")
  {
    float logpCurPolicy = 0.0f;
    float logpOldPolicy = 0.0f;

    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curVar = curSigma*curSigma;

      // Action differences to mu
      const float curActionDif = action[i]-curMu;

      // Normalization constants
      const float Cq = 1./(normalCDF(_actionUpperBounds[i], curMu, curSigma));
      const float Cp = 1./(normalCDF(_actionUpperBounds[i], oldMu, oldSigma));
 
      // Constants involving exp expressions
      const float cexpa = std::exp(-(_actionLowerBounds[i]-curMu)*(_actionLowerBounds[i]-curMu)/(2.*curVar));
      const float cexpb = std::exp(-(_actionUpperBounds[i]-curMu)*(_actionUpperBounds[i]-curMu)/(2.*curVar));
 
      // Normalized gradients of constants
      const float dCqMu = Cq/(M_SQRT2*std::sqrt(M_PI)*curSigma)*(cexpb-cexpa);
      const float dCqSig = Cq/(M_SQRT2*std::sqrt(M_PI)*curSigma)*((_actionUpperBounds[i]-curMu)*cexpb-(_actionLowerBounds[i]-curMu)*cexpa);
      
      // Gradient with respect to Mean
      importanceWeightGradients[i] = (curActionDif/curVar+dCqMu);

      // Gradient with respect to Sigma
      importanceWeightGradients[_problem->_actionVectorSize + i] = (curActionDif * curActionDif) / (curSigma * curVar) - 1.0f / curSigma + dCqSig;

      // Calculate Importance Weight
      logpCurPolicy += std::log(Cq/curSigma)-(action[i]-curMu)*(action[i]-curMu)/(2.*curSigma*curSigma);
      logpOldPolicy += std::log(Cp/oldSigma)-(action[i]-oldMu)*(action[i]-oldMu)/(2.*oldSigma*oldSigma);

    }

    const float logImportanceWeight = logpCurPolicy - logpOldPolicy;
    const float importanceWeight = std::exp(logImportanceWeight);

    // Scale by importance weight to get gradient
    for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
      importanceWeightGradients[i] *= importanceWeight;
  }
 

  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldVariance = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      float alphaCur;
      float betaCur;
      std::tie(alphaCur, betaCur) = betaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      float alphaOld;
      float betaOld;
      std::tie(alphaOld, betaOld) = betaParamTransformAlt(oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // Log probability of action with old policy params
      const float logpOldPolicy = betaLogDensityAlt(action[i], oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);
      const float invpOldPolicy = std::exp(-logpOldPolicy);

      // Variable preparation
      const float Bab = gsl_sf_beta(alphaCur, betaCur);

      const float psiAb = gsl_sf_psi(alphaCur + betaCur);

      const float actionRange = _actionUpperBounds[i] - _actionLowerBounds[i];
      const float logscale = std::log(actionRange);
      const float powscale = std::pow(actionRange, -betaCur - alphaCur + 1.);
      const float factor = -1. * std::pow(action[i] - _actionLowerBounds[i], alphaCur - 1.) * powscale * std::pow(_actionUpperBounds[i] - action[i], betaCur - 1.) * invpOldPolicy / Bab;

      // Rho Grad wrt alpha and beta
      const float daBab = gsl_sf_psi(alphaCur) - psiAb;
      const float drhoda = ((logscale - std::log(action[i] - _actionLowerBounds[i])) + daBab) * factor;
      const float dbBab = gsl_sf_psi(betaCur) - psiAb;
      const float drhodb = (logscale - std::log(_actionUpperBounds[i] - action[i]) + dbBab) * factor;

      // Derivatives of alpha and beta wrt mu and varc
      float dadmu, dadvarc, dbdmu, dbdvarc;
      std::tie(dadmu, dadvarc, dbdmu, dbdvarc) = derivativesBetaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // Rho Grad wrt mu and varc
      importanceWeightGradients[i] = drhoda * dadmu + drhodb * dbdmu;
      importanceWeightGradients[_problem->_actionVectorSize + i] = drhoda * dadvarc + drhodb * dbdvarc;
    }
  }

  return importanceWeightGradients;
}

std::vector<float> Continuous::calculateKLDivergenceGradient(const policy_t &oldPolicy, const policy_t &curPolicy)
{
  // Storage for KL Divergence Gradients
  std::vector<float> KLDivergenceGradients(2.0 * _problem->_actionVectorSize, 0.0);

  if (_policyDistribution == "Normal" || _policyDistribution == "Squashed Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; ++i)
    {
      // Getting parameters from the new and old policies
      const float oldMean = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMean = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float curInvVar = 1. / (curSigma * curSigma);
      const float curInvSig3 = 1. / (curSigma * curSigma * curSigma);
      const float actionDiff = (curMean - oldMean);

      // KL-Gradient with respect to Mean
      KLDivergenceGradients[i] = actionDiff * curInvVar;

      // Contribution to Sigma from Trace
      float gradTr = - curInvSig3 * oldSigma * oldSigma;

      // Contribution to Sigma from Quadratic term
      float gradQuad = -(actionDiff * actionDiff) * curInvSig3;

      // Contribution to Sigma from Determinant
      float gradDet = 1.0f / curSigma;

      // KL-Gradient with respect to Sigma
      KLDivergenceGradients[_problem->_actionVectorSize + i] = gradTr + gradQuad + gradDet;
    }
  }

  if (_policyDistribution == "Clipped Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; ++i)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      const float curInvVar = 1./(curSigma * curSigma);
      const float curInvSig3 = 1./(curSigma * curSigma * curSigma);
      const float actionDiff = (curMu - oldMu);
      
      const float erfa = std::erff((_actionLowerBounds[i]-oldMu)/(M_SQRT2*oldSigma));
      const float erfb = std::erff((_actionUpperBounds[i]-oldMu)/(M_SQRT2*oldSigma));
      
      const float expa = std::exp(-(_actionLowerBounds[i]-oldMu)*(_actionLowerBounds[i]-oldMu)/(2.*oldSigma*oldSigma));
      const float expb = std::exp(-(_actionUpperBounds[i]-oldMu)*(_actionUpperBounds[i]-oldMu)/(2.*oldSigma*oldSigma));

      // KL-Gradient with respect to Mean
      KLDivergenceGradients[i] = -normalCDF(_actionLowerBounds[i], curMu, curSigma)*normalCDF(_actionLowerBounds[i], oldMu, oldSigma)*std::exp(normalLogDensity(_actionLowerBounds[i], curMu, curSigma));
      KLDivergenceGradients[i] -= (oldMu-curMu) * curInvVar * (erfb-erfa);
      KLDivergenceGradients[i] -= oldSigma/(M_SQRT2*std::sqrt(M_PI))*curInvVar*(expa-expb);
      KLDivergenceGradients[i] += (1.-normalLogCDF(_actionUpperBounds[i], curMu, curSigma))*normalCDF(_actionUpperBounds[i], oldMu, oldSigma)*std::exp(normalLogDensity(_actionUpperBounds[i], curMu, curSigma));

      // KL-Gradient with respect to Sigma
      KLDivergenceGradients[_problem->_actionVectorSize + i] = -(_actionLowerBounds[i]-curMu)/curSigma*normalCDF(_actionLowerBounds[i],curMu,curSigma)*std::exp(normalLogDensity(_actionLowerBounds[i], curMu, curSigma));
      KLDivergenceGradients[_problem->_actionVectorSize + i] += 0.5*(-actionDiff*actionDiff*curInvSig3+1./curSigma)*(erfb-erfa);
      KLDivergenceGradients[_problem->_actionVectorSize + i] -= oldSigma*M_SQRT2/std::sqrt(M_PI)*(oldMu-curMu)*curInvSig3*(expa-expb);
      KLDivergenceGradients[_problem->_actionVectorSize + i] -= 0.5*oldSigma*oldSigma*curInvSig3*(erfb-erfa);
      KLDivergenceGradients[_problem->_actionVectorSize + i] -= oldSigma*curInvSig3*((_actionLowerBounds[i]-oldMu)*expa-(_actionUpperBounds[i]-oldMu)*expb);
      KLDivergenceGradients[_problem->_actionVectorSize + i] += (_actionUpperBounds[i]-curMu)/curSigma*(1.-normalCDF(_actionUpperBounds[i], curMu, curSigma))*std::exp(normalLogDensity(_actionUpperBounds[i], curMu, curSigma));
    }
  }
 
  if (_policyDistribution == "Truncated Normal")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldSigma = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float oldVar = oldSigma*oldSigma;
      const float curMu = curPolicy.distributionParameters[i];
      const float curSigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curVar = curSigma*curSigma;
      const float curSigma3 = curSigma*curVar;

      // Action differences between current and old mu
      const float muDif = curMu-oldMu;;

      // Normalization constants
      const float Cq = 1./(normalCDF(_actionUpperBounds[i], curMu, curSigma));
      const float Cp = 1./(normalCDF(_actionUpperBounds[i], oldMu, oldSigma));
 
      // Constants involving exp expressions
      const float cexpa = std::exp(-(_actionLowerBounds[i]-curMu)*(_actionLowerBounds[i]-curMu)/(2.*curVar));
      const float cexpb = std::exp(-(_actionUpperBounds[i]-curMu)*(_actionUpperBounds[i]-curMu)/(2.*curVar));
 
      const float oexpa = std::exp(-(_actionLowerBounds[i]-oldMu)*(_actionLowerBounds[i]-oldMu)/(2.*oldVar));
      const float oexpb = std::exp(-(_actionUpperBounds[i]-oldMu)*(_actionUpperBounds[i]-oldMu)/(2.*oldVar));

      // Normalized gradients of constants
      const float dCqMu = Cq/(M_SQRT2*std::sqrt(M_PI)*curSigma)*(cexpb-cexpa);
      const float dCqSig = Cq/(M_SQRT2*std::sqrt(M_PI)*curSigma)*((_actionUpperBounds[i]-curMu)*cexpb-(_actionLowerBounds[i]-curMu)*cexpa);
      
      // KL grad wrt. mu
      KLDivergenceGradients[i] = muDif/curVar-dCqMu-Cq*curSigma/(M_SQRT2*std::sqrt(M_PI)*curVar)*(oexpa-oexpb);

      // KL grad wrt. sigma
      KLDivergenceGradients[_problem->_actionVectorSize + i] = -muDif*muDif/curSigma3 + 1./curSigma + dCqSig - 2.*Cp*oldSigma/(M_SQRT2*std::sqrt(M_PI))*muDif/curSigma3*(oexpa-oexpb) - oldVar/curSigma3 + Cp*oldSigma/(M_SQRT2*std::sqrt(M_PI)*curSigma3)*((_actionUpperBounds[i]-oldMu)*oexpb-(_actionLowerBounds[i]-oldMu)*oexpa);
    }

  }
 
  if (_policyDistribution == "Beta")
  {
    for (size_t i = 0; i < _problem->_actionVectorSize; ++i)
    {
      // Getting parameters from the new and old policies
      const float oldMu = oldPolicy.distributionParameters[i];
      const float oldVariance = oldPolicy.distributionParameters[_problem->_actionVectorSize + i];
      const float curMu = curPolicy.distributionParameters[i];
      const float curVariance = curPolicy.distributionParameters[_problem->_actionVectorSize + i];

      float alphaCur;
      float betaCur;
      std::tie(alphaCur, betaCur) = betaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      float alphaOld;
      float betaOld;
      std::tie(alphaOld, betaOld) = betaParamTransformAlt(oldMu, oldVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // Constants involving psi function
      const float psiAbCur = gsl_sf_psi(alphaCur + betaCur);
      const float psiAbOld = gsl_sf_psi(alphaOld + betaOld);

      const float actionRange = _actionUpperBounds[i] - _actionLowerBounds[i];

      // KL Grad wrt alpha
      const float dklda = (gsl_sf_psi(alphaCur) - psiAbCur - gsl_sf_psi(alphaOld) - psiAbOld) / actionRange;

      // KL Grad wrt beta
      const float dkldb = (gsl_sf_psi(betaCur) - psiAbCur - gsl_sf_psi(betaOld) - psiAbOld) / actionRange;

      // Derivatives of alpha and beta wrt mu and varc
      float dadmu, dadvarc, dbdmu, dbdvarc;
      std::tie(dadmu, dadvarc, dbdmu, dbdvarc) = derivativesBetaParamTransformAlt(curMu, curVariance, _actionLowerBounds[i], _actionUpperBounds[i]);

      // KL Grad wrt mu and varc
      KLDivergenceGradients[i] = dklda * dadmu + dkldb * dbdmu;
      KLDivergenceGradients[_problem->_actionVectorSize + i] = dklda * dadvarc + dkldb * dbdvarc;
    }
  }

  return KLDivergenceGradients;
}

void Continuous::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Normal Generator"))
 {
 _normalGenerator = dynamic_cast<korali::distribution::univariate::Normal*>(korali::Module::getModule(js["Normal Generator"], _k));
 _normalGenerator->applyVariableDefaults();
 _normalGenerator->applyModuleDefaults(js["Normal Generator"]);
 _normalGenerator->setConfiguration(js["Normal Generator"]);
   eraseValue(js, "Normal Generator");
 }

 if (isDefined(js, "Action Shifts"))
 {
 try { _actionShifts = js["Action Shifts"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Action Shifts']\n%s", e.what()); } 
   eraseValue(js, "Action Shifts");
 }

 if (isDefined(js, "Action Scales"))
 {
 try { _actionScales = js["Action Scales"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Action Scales']\n%s", e.what()); } 
   eraseValue(js, "Action Scales");
 }

 if (isDefined(js, "Policy", "Parameter Count"))
 {
 try { _policyParameterCount = js["Policy"]["Parameter Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Count']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Count");
 }

 if (isDefined(js, "Policy", "Parameter Transformation Masks"))
 {
 try { _policyParameterTransformationMasks = js["Policy"]["Parameter Transformation Masks"].get<std::vector<std::string>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Transformation Masks']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Transformation Masks");
 }

 if (isDefined(js, "Policy", "Parameter Scaling"))
 {
 try { _policyParameterScaling = js["Policy"]["Parameter Scaling"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Scaling']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Scaling");
 }

 if (isDefined(js, "Policy", "Parameter Shifting"))
 {
 try { _policyParameterShifting = js["Policy"]["Parameter Shifting"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Parameter Shifting']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Shifting");
 }

 if (isDefined(js, "Policy", "Distribution"))
 {
 try { _policyDistribution = js["Policy"]["Distribution"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ continuous ] \n + Key:    ['Policy']['Distribution']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_policyDistribution == "Normal") validOption = true; 
 if (_policyDistribution == "Squashed Normal") validOption = true; 
 if (_policyDistribution == "Clipped Normal") validOption = true; 
 if (_policyDistribution == "Truncated Normal") validOption = true; 
 if (_policyDistribution == "Beta") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Policy']['Distribution'] required by continuous.\n", _policyDistribution.c_str()); 
}
   eraseValue(js, "Policy", "Distribution");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Policy']['Distribution'] required by continuous.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Agent::setConfiguration(js);
 _type = "agent/continuous";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: continuous: \n%s\n", js.dump(2).c_str());
} 

void Continuous::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Policy"]["Distribution"] = _policyDistribution;
 if(_normalGenerator != NULL) _normalGenerator->getConfiguration(js["Normal Generator"]);
   js["Action Shifts"] = _actionShifts;
   js["Action Scales"] = _actionScales;
   js["Policy"]["Parameter Count"] = _policyParameterCount;
   js["Policy"]["Parameter Transformation Masks"] = _policyParameterTransformationMasks;
   js["Policy"]["Parameter Scaling"] = _policyParameterScaling;
   js["Policy"]["Parameter Shifting"] = _policyParameterShifting;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Agent::getConfiguration(js);
} 

void Continuous::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Normal Generator\": {\"Type\": \"Univariate/Normal\", \"Mean\": 0.0, \"Standard Deviation\": 1.0}, \"Policy\": {\"Distribution\": \"Normal\"}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Agent::applyModuleDefaults(js);
} 

void Continuous::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Agent::applyVariableDefaults();
} 

bool Continuous::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Agent::checkTermination();
 return hasFinished;
}



} //agent
} //solver
} //korali

