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

 if (isDefined(js, "Num Simulations"))
 {
 try { _numSimulations = js["Num Simulations"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Num Simulations']\n%s", e.what()); } 
   eraseValue(js, "Num Simulations");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Num Simulations'] required by SSM.\n"); 

 if (isDefined(js, "Num Bins"))
 {
 try { _numBins = js["Num Bins"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Num Bins']\n%s", e.what()); } 
   eraseValue(js, "Num Bins");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Num Bins'] required by SSM.\n"); 

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
   js["Num Simulations"] = _numSimulations;
   js["Num Bins"] = _numBins;
 if(_exponentialGenerator != NULL) _exponentialGenerator->getConfiguration(js["Exponential Generator"]);
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void SSM::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Num Simulatons\": 1, \"Num Bins\": 100, \"Termination Criteria\": {}}";
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

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

;

} //solver
} //korali
;
