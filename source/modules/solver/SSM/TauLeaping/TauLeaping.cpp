#include "modules/solver/SSM/TauLeaping/TauLeaping.hpp"

namespace korali
{
namespace solver
{
namespace ssm
{
;

void TauLeaping::advance()
{
  _propensities.resize(_problem->_reactions.size());

  double a0 = 0.0;

  // Calculate propensities
  for (size_t k = 0; k < _problem->_reactions.size(); ++k)
  {
    double a = _problem->computePropensity(k, _numReactants);

    a0 += a;
    _propensities[k] = a;
  }

  // Mark critical reactions
  bool allReactionsAreCritical = true;
  _isCriticalReaction.resize(_problem->_reactions.size());

  for (size_t k = 0; k < _problem->_reactions.size(); ++k)
  {
    const double a = _propensities[k];
    const double L = _problem->calculateMaximumAllowedFirings(k, _numReactants);

    const bool isCritical = !((a > 0) && (L <= _nc));
    _isCriticalReaction[k] = isCritical;

    allReactionsAreCritical = allReactionsAreCritical && isCritical;
  }

  // Estimate maximum tau
  const double tauP = allReactionsAreCritical ? std::numeric_limits<double>::infinity() : estimateLargestTau();


  // Accept or reject step
  if (tauP <  _acceptanceFactor / a0)
  {
        // reject, execute SSA.
        reset(_numReactants, _time);

        for (int i = 0; i < _numStepsSSA; ++i)
        {
            //TODO
            //_ssa.advance();
            //if (_ssa.getTime() >= tend_)
            //    break;
        }

        // TODO
        //_time = _ssa.getTime();
        //const auto newState = ssa_.getState();
        //std::copy(newState.begin(), newState.end(), numSpecies_.begin());
    }
    else
    {
        // TODO
    }
}

double TauLeaping::estimateLargestTau() const
{
    // TODO
    return 0.;
}

void TauLeaping::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Nc"))
 {
 try { _nc = js["Nc"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Nc']\n%s", e.what()); } 
   eraseValue(js, "Nc");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Nc'] required by TauLeaping.\n"); 

 if (isDefined(js, "Eps"))
 {
 try { _eps = js["Eps"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Eps']\n%s", e.what()); } 
   eraseValue(js, "Eps");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Eps'] required by TauLeaping.\n"); 

 if (isDefined(js, "Acceptance Factor"))
 {
 try { _acceptanceFactor = js["Acceptance Factor"].get<double>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Acceptance Factor']\n%s", e.what()); } 
   eraseValue(js, "Acceptance Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Acceptance Factor'] required by TauLeaping.\n"); 

 if (isDefined(js, "Num Steps SSA"))
 {
 try { _numStepsSSA = js["Num Steps SSA"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ TauLeaping ] \n + Key:    ['Num Steps SSA']\n%s", e.what()); } 
   eraseValue(js, "Num Steps SSA");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Num Steps SSA'] required by TauLeaping.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 SSM::setConfiguration(js);
 _type = "SSM/TauLeaping";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: TauLeaping: \n%s\n", js.dump(2).c_str());
} 

void TauLeaping::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Nc"] = _nc;
   js["Eps"] = _eps;
   js["Acceptance Factor"] = _acceptanceFactor;
   js["Num Steps SSA"] = _numStepsSSA;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 SSM::getConfiguration(js);
} 

void TauLeaping::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 SSM::applyModuleDefaults(js);
} 

void TauLeaping::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 SSM::applyVariableDefaults();
} 

bool TauLeaping::checkTermination()
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
