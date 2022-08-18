#include "modules/solver/SSM/SSM.hpp"

namespace korali
{
namespace solver
{
;

void simulateTrajectory() { return; }

void SSM::runGeneration()
{
  // TODO
  if (_k->_currentGeneration == 1) setInitialConfiguration();
}

void SSM::printGenerationBefore() 
{ 
    // TODO
}

void SSM::printGenerationAfter()
{
    // TODO
}

void SSM::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Exponential Generator"))
 {
 _exponentialGenerator = dynamic_cast<korali::distribution::univariate::Exponential*>(korali::Module::getModule(js["Exponential Generator"], _k));
 _exponentialGenerator->applyVariableDefaults();
 _exponentialGenerator->applyModuleDefaults(js["Exponential Generator"]);
 _exponentialGenerator->setConfiguration(js["Exponential Generator"]);
   eraseValue(js, "Exponential Generator");
 }

 if (isDefined(js, "Num Bins"))
 {
 try { _numBins = js["Num Bins"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Num Bins']\n%s", e.what()); } 
   eraseValue(js, "Num Bins");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Num Bins'] required by SSM.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Num Simulations"))
 {
 try { _maxNumSimulations = js["Termination Criteria"]["Max Num Simulations"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Termination Criteria']['Max Num Simulations']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Num Simulations");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Num Simulations'] required by SSM.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Solver::setConfiguration(js);
 _type = "SSM";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: SSM: \n%s\n", js.dump(2).c_str());
} 

void SSM::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Num Bins"] = _numBins;
   js["Termination Criteria"]["Max Num Simulations"] = _maxNumSimulations;
 if(_exponentialGenerator != NULL) _exponentialGenerator->getConfiguration(js["Exponential Generator"]);
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void SSM::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Num Bins\": 100, \"Termination Criteria\": {\"Max Num Simulations\": 1}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void SSM::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Solver::applyVariableDefaults();
} 

bool SSM::checkTermination()
{
 bool hasFinished = false;

 if (_maxNumSimulations < _k->_currentGeneration)
 {
  _terminationCriteria.push_back("SSM['Max Num Simulations'] = " + std::to_string(_maxNumSimulations) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

;

} //solver
} //korali
;
