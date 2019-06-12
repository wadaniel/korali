#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Problem::Base::Base(nlohmann::json& js)
{
 setConfiguration(js);
}

Korali::Problem::Base::~Base()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Problem::Base::getConfiguration()
{
 auto js = nlohmann::json();

 for (size_t i = 0; i < N; i++) js["Variables"][i] = _variables[i]->getConfiguration();

 return js;
}

void Korali::Problem::Base::setConfiguration(nlohmann::json& js)
{
 // Configure Variables
  std::vector<Korali::Variable::Base*> tmp;

  if (isArray(js, { "Variables" } ))
  for (size_t i = 0; i < js["Variables"].size(); i++)
  {
   auto dString = consume(js["Variables"][i], { "Distribution", "Type" }, KORALI_STRING, "Default");
   bool foundDistribution = false;
   if (dString == "Default")     { tmp.push_back(new Korali::Variable::Base(js["Variables"][i], _k->_seed++));        foundDistribution = true; }
   if (dString == "Uniform")     { tmp.push_back(new Korali::Variable::Uniform(js["Variables"][i], _k->_seed++));     foundDistribution = true; }
   if (dString == "Gaussian")    { tmp.push_back(new Korali::Variable::Gaussian(js["Variables"][i], _k->_seed++));    foundDistribution = true; }
   if (dString == "Gamma")       { tmp.push_back(new Korali::Variable::Gamma(js["Variables"][i], _k->_seed++));       foundDistribution = true; }
   if (dString == "Exponential") { tmp.push_back(new Korali::Variable::Exponential(js["Variables"][i], _k->_seed++)); foundDistribution = true; }
   if (foundDistribution == false) { fprintf(stderr, "[Korali] Error: Incorrect or missing distribution for parameter %lu.\n", i); exit(-1); }
  }

  if (tmp.size() == 0) { fprintf(stderr, "[Korali] Error: Incorrect or undefined parameters.\n"); exit(-1); }

  _statisticalVariableCount = 0;
  _computationalVariableCount = 0;
  for (size_t i = 0; i < tmp.size(); i++) if (tmp[i]->_type == KORALI_COMPUTATIONAL) { _variables.push_back(tmp[i]); _computationalVariableCount++; }
  for (size_t i = 0; i < tmp.size(); i++) if (tmp[i]->_type == KORALI_STATISTICAL)   { _variables.push_back(tmp[i]); _statisticalVariableCount++; };
  N = _variables.size();
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

