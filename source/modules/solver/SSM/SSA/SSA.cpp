#include "modules/solver/SSM/SSA/SSA.hpp"

namespace korali
{
namespace solver
{
namespace ssm
{
;

void SSA::setInitialConfiguration()
{
    //TODO
    _variableCount = _k->_variables.size();
}

void SSA::simulateTrajectory()
{
    //TODO
}

void SSA::finalize()
{
    //TODO
}

void SSA::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 SSM::setConfiguration(js);
 _type = "SSM/SSA";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: SSA: \n%s\n", js.dump(2).c_str());
} 

void SSA::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 SSM::getConfiguration(js);
} 

void SSA::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 SSM::applyModuleDefaults(js);
} 

void SSA::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 SSM::applyVariableDefaults();
} 

bool SSA::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || SSM::checkTermination();
 return hasFinished;
}

;

} //ssm
} //solver
} //korali
;
