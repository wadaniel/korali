#include "korali.hpp"

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Problem::Constrained::getConfiguration()
{
 _k->_js["Problem"]["Type"] = "Constrained Optimization";

 if (_objective == maximizeFitness)  _k->_js["Problem"]["Objective"] = "Maximize";
 if (_objective == minimizeFitness)  _k->_js["Problem"]["Objective"] = "Minimize";

}

void Korali::Problem::Constrained::setConfiguration()
{

 bool foundObjective = false;
 std::string objectiveString = consume(_k->_js, { "Problem", "Objective" }, KORALI_STRING, "Undefined");
 if (objectiveString == "Maximize") { _objective = maximizeFitness; foundObjective = true; }
 if (objectiveString == "Minimize") { _objective = minimizeFitness; foundObjective = true; }
 if (foundObjective == false) { koraliError("Missing or incorrect optimization objective (maximize/minimize): %s.\n", objectiveString.c_str()); exit(-1); }

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Problem::Constrained::initialize()
{
 if (_k->_constraints.size() < 1) koraliError("Constrained Optimization problems require defining at least one constraint.\n");
 if (_k->_modelDefined == false) koraliError("Constrained Optimization problems require defining a computational model.\n");
}

void Korali::Problem::Constrained::finalize()
{

}

void Korali::Problem::Constrained::packVariables(double* sample, Korali::Model& data)
{
 for (size_t i = 0; i < _k->N; i++) data._computationalVariables.push_back(sample[i]);
}

double Korali::Problem::Constrained::evaluateFitness(Korali::Model& data)
{

 if (data._results.size() != 1)
  koraliError("Constrained Optimization problems require exactly a 1-element result array. Provided: %lu.\n", data._results.size());

 double fitnessSign = _objective == maximizeFitness ? 1.0 : -1.0;

 return fitnessSign*data._results[0];
}

double Korali::Problem::Constrained::evaluateLogPrior(double* sample)
{
  return 0.0;
}
