#include "modules/problem/reaction/reaction.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Reaction::initialize()
{
  if (_k->_variables.size() == 0) KORALI_LOG_ERROR("Reaction problems require at least one variable.\n");
}

void Reaction::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Simulation Length"))
 {
 try { _simulationLength = js["Simulation Length"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reaction ] \n + Key:    ['Simulation Length']\n%s", e.what()); } 
   eraseValue(js, "Simulation Length");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Simulation Length'] required by reaction.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Reactant Number"))
 {
 try { _k->_variables[i]->_initialReactantNumber = _k->_js["Variables"][i]["Initial Reactant Number"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reaction ] \n + Key:    ['Initial Reactant Number']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Reactant Number");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Reactant Number'] required by reaction.\n"); 

 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("SSM"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: reaction\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "reaction";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: reaction: \n%s\n", js.dump(2).c_str());
} 

void Reaction::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Simulation Length"] = _simulationLength;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Initial Reactant Number"] = _k->_variables[i]->_initialReactantNumber;
 } 
 Problem::getConfiguration(js);
} 

void Reaction::applyModuleDefaults(knlohmann::json& js) 
{

 Problem::applyModuleDefaults(js);
} 

void Reaction::applyVariableDefaults() 
{

 std::string defaultString = "{\"Initial Reactant Number\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Problem::applyVariableDefaults();
} 

;

} //problem
} //korali
;
