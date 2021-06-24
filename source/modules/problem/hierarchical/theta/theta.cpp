#include "modules/conduit/conduit.hpp"
#include "modules/problem/hierarchical/theta/theta.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
namespace hierarchical
{


  void
  Theta::initialize()
{
  // Setting experiment configurations to actual korali experiments
  _psiExperimentObject._js.getJson() = _psiExperiment;
  _thetaExperimentObject._js.getJson() = _thetaExperiment;

  // Running initialization to verify that the configuration is correct
  _psiExperimentObject.initialize();
  _thetaExperimentObject.initialize();

  _psiProblem = dynamic_cast<Psi *>(_psiExperimentObject._problem);
  if (_psiProblem == NULL) KORALI_LOG_ERROR("Psi experiment passed is not of type Hierarchical/Psi\n");

  if (_thetaExperiment["Is Finished"] == false)
    KORALI_LOG_ERROR("The Hierarchical Bayesian (Theta) requires that the theta problem has run completely, but this one has not.\n");

  // Now inheriting Sub problem's variables
  _k->_distributions = _thetaExperimentObject._distributions;
  _k->_variables = _thetaExperimentObject._variables;

  _thetaVariableCount = _thetaExperimentObject._variables.size();
  _psiVariableCount = _psiExperimentObject._variables.size();

  // Loading Psi problem results
  _psiProblemSampleCount = _psiExperiment["Solver"]["Chain Leaders LogLikelihoods"].size();
  _psiProblemSampleLogLikelihoods = _psiExperiment["Solver"]["Sample LogLikelihood Database"].get<std::vector<double>>();
  _psiProblemSampleLogPriors = _psiExperiment["Solver"]["Sample LogPrior Database"].get<std::vector<double>>();
  _psiProblemSampleCoordinates = _psiExperiment["Solver"]["Sample Database"].get<std::vector<std::vector<double>>>();

  for (size_t i = 0; i < _psiProblemSampleLogPriors.size(); i++)
  {
    double expPrior = exp(_psiProblemSampleLogPriors[i]);
    if (std::isfinite(expPrior) == false)
      KORALI_LOG_ERROR("Non finite (%lf) prior has been detected at sample %zu in Psi problem.\n", expPrior, i);
  }

  // Loading Theta problem results
  _thetaProblemSampleCount = _thetaExperiment["Solver"]["Chain Leaders LogLikelihoods"].size();
  _thetaProblemSampleLogLikelihoods = _thetaExperiment["Solver"]["Sample LogLikelihood Database"].get<std::vector<double>>();
  _thetaProblemSampleLogPriors = _thetaExperiment["Solver"]["Sample LogPrior Database"].get<std::vector<double>>();
  _thetaProblemSampleCoordinates = _thetaExperiment["Solver"]["Sample Database"].get<std::vector<std::vector<double>>>();

  for (size_t i = 0; i < _thetaProblemSampleLogPriors.size(); i++)
  {
    double expPrior = exp(_thetaProblemSampleLogPriors[i]);
    if (std::isfinite(expPrior) == false)
      KORALI_LOG_ERROR("Non finite (%lf) prior has been detected at sample %zu in Theta problem.\n", expPrior, i);
  }

  std::vector<double> logValues;
  logValues.resize(_thetaProblemSampleCount);

  _psiProblem = dynamic_cast<Psi *>(_psiProblem);

  for (size_t i = 0; i < _psiProblemSampleCount; i++)
  {
    Sample psiSample;
    psiSample["Parameters"] = _psiProblemSampleCoordinates[i];

    _psiProblem->updateConditionalPriors(psiSample);

    for (size_t j = 0; j < _thetaProblemSampleCount; j++)
    {
      double logConditionalPrior = 0;
      for (size_t k = 0; k < _thetaVariableCount; k++)
        logConditionalPrior += _psiExperimentObject._distributions[_psiProblem->_conditionalPriorIndexes[k]]->getLogDensity(_thetaProblemSampleCoordinates[j][k]);

      logValues[j] = logConditionalPrior - _thetaProblemSampleLogPriors[j];
    }

    double localSum = -log(_thetaProblemSampleCount) + logSumExp(logValues);

    _precomputedLogDenominator.push_back(localSum);
  }

  Hierarchical::initialize();
}

void Theta::evaluateLogLikelihood(Sample &sample)
{
  std::vector<double> logValues;
  logValues.resize(_psiProblemSampleCount);

  for (size_t i = 0; i < _psiProblemSampleCount; i++)
  {
    Sample psiSample;
    psiSample["Parameters"] = _psiProblemSampleCoordinates[i];

    _psiProblem->updateConditionalPriors(psiSample);

    double logConditionalPrior = 0.;
    for (size_t k = 0; k < _thetaVariableCount; k++)
      logConditionalPrior += _psiExperimentObject._distributions[_psiProblem->_conditionalPriorIndexes[k]]->getLogDensity(sample["Parameters"][k]);

    logValues[i] = logConditionalPrior - _precomputedLogDenominator[i];
  }

  sample["logLikelihood"] = -log(_psiProblemSampleCount) + logSumExp(logValues);
}

void Theta::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Theta Experiment"))
 {
 _thetaExperiment = js["Theta Experiment"].get<knlohmann::json>();

   eraseValue(js, "Theta Experiment");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Theta Experiment'] required by theta.\n"); 

 if (isDefined(js, "Psi Experiment"))
 {
 _psiExperiment = js["Psi Experiment"].get<knlohmann::json>();

   eraseValue(js, "Psi Experiment");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Psi Experiment'] required by theta.\n"); 

 Hierarchical::setConfiguration(js);
 _type = "hierarchical/theta";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: theta: \n%s\n", js.dump(2).c_str());
} 

void Theta::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Theta Experiment"] = _thetaExperiment;
   js["Psi Experiment"] = _psiExperiment;
 Hierarchical::getConfiguration(js);
} 

void Theta::applyModuleDefaults(knlohmann::json& js) 
{

 Hierarchical::applyModuleDefaults(js);
} 

void Theta::applyVariableDefaults() 
{

 Hierarchical::applyVariableDefaults();
} 



  } //hierarchical
} //problem
} //korali

