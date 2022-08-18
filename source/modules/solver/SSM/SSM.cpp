#include "modules/solver/SSM/SSM.hpp"

namespace korali
{
namespace solver
{
;


void SSM::initialize()
{
    _variableCount = _k->_variables.size();
    _problem = dynamic_cast<problem::Reaction *>(_k->_problem);
    _binCounter = std::vector<std::vector<int>>(_maxNumSimulations,std::vector<int>(_diagnosticsNumBins, 0));
    _binnedTrajectories = std::vector<std::vector<std::vector<int>>>(_variableCount, std::vector<std::vector<int>>(_maxNumSimulations,std::vector<int>(_diagnosticsNumBins, 0)));
}

void SSM::reset(std::vector<int> numReactants, double time)
{
    _time = time;
    _numReactants = std::move(numReactants);
}

void SSM::updateBins()
{
    size_t binIndex = _time * _simulationLength / (double) _diagnosticsNumBins;
    for(size_t k = 0; k < _variableCount; k++)
    {
        _binCounter[_k->_currentGeneration-1][binIndex] += 1;
        _binnedTrajectories[k][_k->_currentGeneration-1][binIndex] += _numReactants[k];
    }
}


void SSM::runGeneration()
{
  if (_k->_currentGeneration == 1) setInitialConfiguration();

  reset(_problem->_initialReactantNumbers);
  updateBins();

  while (_time < _simulationLength)
  {
    //for (auto& d : diagnostics_)
        //d->collect(i, solver_->getTime(), solver_->getState());
    advance();
    updateBins();
  }

}

void SSM::printGenerationBefore() 
{ 
    // TODO
}

void SSM::printGenerationAfter()
{
    // TODO
}

void SSM::finalize()
{
    //TODO
    std::vector<std::vector<double>> resultsMeanTrajectory(_variableCount, std::vector<double>(_diagnosticsNumBins, 0.));
    for(size_t k = 0; k < _variableCount; k++)
    {
        for(size_t idx = 0; idx < _diagnosticsNumBins; ++idx)
        {
            for(size_t sim = 0; sim < _maxNumSimulations; ++sim) if (_binCounter[sim][idx] > 0)
                resultsMeanTrajectory[k][idx] += _binnedTrajectories[k][sim][idx] / _binCounter[sim][idx];
            resultsMeanTrajectory[k][idx] /= _maxNumSimulations;
        }
    }

}


void SSM::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Time"))
 {
 try { _time = js["Time"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Time']\n%s", e.what()); } 
   eraseValue(js, "Time");
 }

 if (isDefined(js, "Num Reactants"))
 {
 try { _numReactants = js["Num Reactants"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Num Reactants']\n%s", e.what()); } 
   eraseValue(js, "Num Reactants");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Bin Counter"))
 {
 try { _binCounter = js["Bin Counter"].get<std::vector<std::vector<int>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Bin Counter']\n%s", e.what()); } 
   eraseValue(js, "Bin Counter");
 }

 if (isDefined(js, "Binned Trajectories"))
 {
 try { _binnedTrajectories = js["Binned Trajectories"].get<std::vector<std::vector<std::vector<int>>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Binned Trajectories']\n%s", e.what()); } 
   eraseValue(js, "Binned Trajectories");
 }

 if (isDefined(js, "Simulation Length"))
 {
 try { _simulationLength = js["Simulation Length"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Simulation Length']\n%s", e.what()); } 
   eraseValue(js, "Simulation Length");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Simulation Length'] required by SSM.\n"); 

 if (isDefined(js, "Diagnostics", "Num Bins"))
 {
 try { _diagnosticsNumBins = js["Diagnostics"]["Num Bins"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ SSM ] \n + Key:    ['Diagnostics']['Num Bins']\n%s", e.what()); } 
   eraseValue(js, "Diagnostics", "Num Bins");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Diagnostics']['Num Bins'] required by SSM.\n"); 

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
   js["Simulation Length"] = _simulationLength;
   js["Diagnostics"]["Num Bins"] = _diagnosticsNumBins;
   js["Termination Criteria"]["Max Num Simulations"] = _maxNumSimulations;
   js["Time"] = _time;
   js["Num Reactants"] = _numReactants;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Bin Counter"] = _binCounter;
   js["Binned Trajectories"] = _binnedTrajectories;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void SSM::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Diagnostics\": {\"Num Bins\": 100}, \"Termination Criteria\": {\"Max Num Simulations\": 1}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}}";
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
