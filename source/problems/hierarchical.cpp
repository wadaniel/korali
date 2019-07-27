#include "korali.hpp"

void Korali::Problem::Hierarchical::getConfiguration()
{
 _k->_js["Problem"]["Type"] = "Hierarchical Bayesian";

 if (_operationType == SamplePsi)   _k->_js["Problem"]["Model"] = "Sample Psi";
 if (_operationType == SampleTheta) _k->_js["Problem"]["Model"] = "Sample Theta";

 for (size_t i = 0; i < _subProblems.size(); i++)
  _k->_js["Problem"]["Sub-Problems"][i] = _subProblems[i];

 for (size_t i = 0; i < _conditionalPriors.size(); i++)
 {
  _conditionalPriors[i]._variable->getDistribution(_k->_js["Problem"]["Sub-Problems"][i]);
  for (size_t j = 0; j < _conditionalPriors[i]._properties.size(); j++)
  {
   _k->_js["Problem"]["Sub-Problems"][i][_conditionalPriors[i]._properties[j].first] =
     _k->_variables[_conditionalPriors[i]._properties[j].second]->_name;
  }
 }
}

void Korali::Problem::Hierarchical::setConfiguration()
{
  bool foundSubProblemType = false;
  std::string operationTypeString = consume(_k->_js, { "Problem", "Model" }, KORALI_STRING, "Undefined");
  if (operationTypeString == "Sample Psi")   { _operationType = SamplePsi;   foundSubProblemType = true; }
  if (operationTypeString == "Sample Theta") { _operationType = SampleTheta; foundSubProblemType = true; }
  if (foundSubProblemType == false) koraliError("Incorrect or no sub-problem Type selected for Hierarchical Bayesian: %s.\n", operationTypeString.c_str());

  if (isArray(_k->_js["Problem"], { "Sub-Problems" } ))
   for (size_t i = 0; i < _k->_js["Problem"]["Sub-Problems"].size(); i++)
   {
    std::string str = _k->_js["Problem"]["Sub-Problems"][i];
    auto js = nlohmann::json::parse(str);
    _subProblems.push_back(js);
   }
  _k->_js["Problem"].erase("Sub-Problems");

  if (isArray(_k->_js["Problem"], { "Conditional Priors" } ))
  for (size_t i = 0; i < _k->_js["Problem"]["Conditional Priors"].size(); i++)
  {
   conditionalPrior prior;
   prior._variable = new Korali::Variable();
   _k->_js["Problem"]["Conditional Priors"][i]["Seed"] = _k->_seed++;
   prior._variable->setDistribution(_k->_js["Problem"]["Conditional Priors"][i]);
   _conditionalPriors.push_back(prior);
  }

}

void Korali::Problem::Hierarchical::initialize()
{
 if (_conditionalPriors.size() == 0) koraliError("Hierarchical Bayesian problems require at least one conditional prior\n");

 for(size_t i = 0; i < _k->N; i++) if(_k->_variables[i]->_distributionType == "No Distribution")
	koraliError("Hierarchical Bayesian problems requires prior distribution for all variables. (Missing for %s).\n", _k->_variables[i]->_name.c_str());

 if (_k->_constraints.size() > 0) koraliError("Hierarchical Bayesian problems do not allow constraint definitions.\n");
 if (_k->_modelDefined == true) koraliError("Hierarchical Bayesian does not require a computational model, but one was provided.\n");
 if (_subProblems.size() < 2) koraliError("Hierarchical Bayesian problem requires defining at least two executed sub-problems.\n");

 for (size_t i = 0; i < _subProblems.size(); i++)
   if (_subProblems[i]["Is Finished"] == false) koraliError("The Hierarchical Bayesian requires that all problems have run completely, but Problem %lu has not.\n", i);

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
