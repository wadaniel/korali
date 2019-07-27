#include "korali.hpp"

void Korali::Problem::Hierarchical::getConfiguration()
{
 _k->_js["Problem"]["Type"] = "Hierarchical Bayesian";

 if (_operationType == SamplePsi)   _k->_js["Problem"]["Model"] = "Sample Psi";
 if (_operationType == SampleTheta) _k->_js["Problem"]["Model"] = "Sample Theta";

 for (size_t i = 0; i < _conditionalPriors.size(); i++)
 {
  _conditionalPriors[i]->_variable->getDistribution(_k->_js["Problem"]["Conditional Priors"][i]);
  for (size_t j = 0; j < _conditionalPriors[i]->_properties.size(); j++)
  {
   _k->_js["Problem"]["Conditional Priors"][i][_conditionalPriors[i]->_properties[j].first] =
     _k->_variables[_conditionalPriors[i]->_properties[j].second]->_name;
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
   auto prior = new conditionalPrior();
   prior->_variable = new Korali::Variable();
   _k->_js["Problem"]["Conditional Priors"][i]["Seed"] = _k->_seed++;
   prior->_variable->setDistribution(_k->_js["Problem"]["Conditional Priors"][i]);

   // Processing references to hyperparameters
   for (auto& property : _k->_js["Problem"]["Conditional Priors"][i].items())
   {
    prior->_variable->setProperty(property.key(), 1.0); // Testing property exists.
    if (_k->_js["Problem"]["Conditional Priors"][i][property.key()].is_string())
    {
     bool foundHyperparameter = false;
     for (size_t v = 0; v < _k->_variables.size(); v++)
      if (_k->_variables[v]->_name == property.value())
      {
       prior->_properties.push_back(std::make_pair(property.key(), v));
       foundHyperparameter = true;
      }
     if (foundSubProblemType == false) koraliError("Conditional Prior %d references a hyperparameter (%s) which was not defined as problem variable.\n", i, property.key().c_str());
    }
   }
   _conditionalPriors.push_back(prior);
  }

  _k->_js["Problem"].erase("Conditional Priors");
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
 // Now re-configuring conditional priors given hyperparameters
 for (size_t i = 0; i < _conditionalPriors.size(); i++)
 {
  for (size_t j = 0; j < _conditionalPriors[i]->_properties.size(); j++)
  {
   std::string propertyName = _conditionalPriors[i]->_properties[j].first;
   size_t variableId = _conditionalPriors[i]->_properties[j].second;
   double propertyValue = data.getVariable(variableId);
   _conditionalPriors[i]->_variable->setProperty(propertyName, propertyValue);
  }
 }

 double fitness = 1.0;

 for (size_t i = 0; i < _conditionalPriors.size(); i++)
 {
  // George: I just put gibberish here. I'll let you to do the rest.
  double x = _conditionalPriors[i]->_variable->getRandomNumber();
  fitness *= _conditionalPriors[i]->_variable->getLogDensity(x);
 }

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
