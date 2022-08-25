#include "engine.hpp"
#include "modules/solver/integrator/montecarlo/MonteCarlo.hpp"

namespace korali
{
namespace solver
{
namespace integrator
{
;

void MonteCarlo::setInitialConfiguration()
{
  _variableCount = _k->_variables.size();

  _integral = 0.;
  _sampleCount = 1;
  for (size_t i = 0; i < _variableCount; i++)
    _sampleCount *= _k->_variables[i]->_numberOfGridpoints;
  _maxModelEvaluations = std::min(_maxModelEvaluations, _sampleCount);

  if (_k->_variables[0]->_samplePoints.size() > 0)
  { // quadrature
    _indicesHelper.resize(_variableCount);
    _indicesHelper[0] = _k->_variables[0]->_samplePoints.size();
    _indicesHelper[1] = _k->_variables[0]->_samplePoints.size();
    for (size_t i = 2; i < _indicesHelper.size(); i++)
    {
      _indicesHelper[i] = _indicesHelper[i - 1] * _k->_variables[i - 1]->_samplePoints.size();
    }
  }
}

void MonteCarlo::integrate()
{
    // TODO
}

void MonteCarlo::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Number of Samples"))
 {
 try { _numberofSamples = js["Number of Samples"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ montecarlo ] \n + Key:    ['Number of Samples']\n%s", e.what()); } 
   eraseValue(js, "Number of Samples");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Number of Samples'] required by montecarlo.\n"); 

 Integrator::setConfiguration(js);
 _type = "integrator/montecarlo";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: montecarlo: \n%s\n", js.dump(2).c_str());
} 

void MonteCarlo::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Number of Samples"] = _numberofSamples;
 Integrator::getConfiguration(js);
} 

void MonteCarlo::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Executions Per Generation\": 500000000}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Integrator::applyModuleDefaults(js);
} 

void MonteCarlo::applyVariableDefaults() 
{

 Integrator::applyVariableDefaults();
} 

;

} //integrator
} //solver
} //korali
;
