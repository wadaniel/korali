#include "korali.h"

void Korali::Problem::Hierarchical::getConfiguration()
{
 _k->_js["Problem"]["Type"] = "Hierarchical Bayesian";

 if (_operationType == SamplePsi)   _k->_js["Problem"]["Model"] = "Sample Psi";
 if (_operationType == SampleTheta) _k->_js["Problem"]["Model"] = "Sample Theta";

 for (size_t i = 0; i < _k->N; i++)  _k->_variables[i]->getDistribution(_k->_js["Variables"][i]["Bayesian"]["Prior Distribution"]);
}

void Korali::Problem::Hierarchical::setConfiguration()
{
  bool foundSubProblemType = false;
  std::string operationTypeString = consume(_k->_js, { "Problem", "Model" }, KORALI_STRING, "Undefined");
  if (operationTypeString == "Sample Psi")   { _operationType = SamplePsi;   foundSubProblemType = true; }
  if (operationTypeString == "Sample Theta") { _operationType = SampleTheta; foundSubProblemType = true; }
  if (foundSubProblemType == false) { koraliError("Incorrect or no sub-problem Type selected for Hierarchical Bayesian: %s.\n", operationTypeString.c_str()); exit(-1); }

  if (isArray(_k->_js, { "Variables" } ))
  for (size_t i = 0; i < _k->N; i++)
  {
    bool foundPriorDistribution = isDefined(_k->_js["Variables"][i], { "Prior Distribution" });
    if (foundPriorDistribution == false) { koraliError("No Prior Distribution information provided for variable: %s.\n", _k->_variables[i]->_name.c_str()); exit(-1); }

    _k->_js["Variables"][i]["Prior Distribution"]["Seed"] = _k->_seed++;
    _k->_variables[i]->setDistribution(_k->_js["Variables"][i]["Prior Distribution"]);
  }
}

void Korali::Problem::Hierarchical::initialize()
{
 if (_k->_constraints.size() > 0) koraliError("Bayesian Problems do not allow constraint definitions.\n");
 if (_k->_modelDefined == true) koraliError("Hierarchical Bayesian does not require a computational model, but one was provided.\n");
 if (_k->_likelihoodDefined == true) koraliError("Hierarchical Bayesian does not require a likelihood function, but one was provided.\n");
 if (_k->_subProblems.size() < 2) koraliError("Hierarchical Bayesian problem requires defining at least two executed sub-problems.\n");

 for (size_t i = 0; i < _k->_subProblems.size(); i++)
   if (_k->_subProblems[i]["Is Finished"] == false) koraliError("The Hierarchical Bayesian requires that all problems have run completely, but Problem %lu has not.\n", i);

 if (_operationType == SamplePsi)   _k->setModel(Korali::Problem::Hierarchical::samplePsi);
 if (_operationType == SampleTheta) _k->setModel(Korali::Problem::Hierarchical::sampleTheta);
}

void Korali::Problem::Hierarchical::finalize()
{

}

void Korali::Problem::Hierarchical::packVariables(double* sample, Korali::Model& data)
{
 for (size_t i = 0; i < _k->N; i++) data._computationalVariables.push_back(sample[i]);
}

double Korali::Problem::Hierarchical::evaluateFitness(Korali::Model& data)
{
 double fitness = 0.0;

 fitness = data._results[0];

 return fitness;
}

double Korali::Problem::Hierarchical::evaluateLogPrior(double* sample)
{
 double logPrior = 0.0;
 for (size_t i = 0; i < _k->N; i++) logPrior += _k->_variables[i]->getLogDensity(sample[i]);
 return logPrior;
}

void Korali::Problem::Hierarchical::samplePsi(Korali::Model& data)
{
 data.addResult(0.0);
}

void Korali::Problem::Hierarchical::sampleTheta(Korali::Model& data)
{
 data.addResult(0.0);
}
