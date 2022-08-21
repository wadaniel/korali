#include "modules/solver/SSM/TauLeaping/TauLeaping.hpp"

namespace korali
{
namespace solver
{
namespace ssm
{
;


void TauLeaping::ssaAdvance()
{
  _cumPropensities.resize(_numReactions);

  double a0 = 0.0;

  // Calculate propensities
  for (size_t k = 0; k < _numReactions; ++k)
  {
    const double a = _problem->computePropensity(k, _numReactants);

    a0 += a;
    _cumPropensities[k] = a0;
  }

  // Sample time step from exponential distribution
  const double r1 = _uniformGenerator->getRandomNumber();

  double tau = -std::log(r1) / a0;

  // Advance time
  _time += tau;

  if (_time > _simulationLength)
    _time = _simulationLength;

  // Exit if no reactions fire
  if (a0 == 0)
    return;

  const double r2 = _cumPropensities.back() * _uniformGenerator->getRandomNumber();

  // Sample a reaction
  size_t selection = 0;
  while (r2 > _cumPropensities[selection])
    selection++;

  // Update the reactants according to chosen reaction
  _problem->applyChanges(selection, _numReactants);
}

void TauLeaping::advance()
{
  _propensities.resize(_numReactions);

  double a0 = 0.0;

  // Calculate propensities
  for (size_t k = 0; k < _numReactions; ++k)
  {
    double a = _problem->computePropensity(k, _numReactants);

    a0 += a;
    _propensities[k] = a;
  }

  // Mark critical reactions
  bool allReactionsAreCritical = true;
  _isCriticalReaction.resize(_numReactions);

  for (size_t k = 0; k < _numReactions; ++k)
  {
    const double a = _propensities[k];
    const double L = _problem->calculateMaximumAllowedFirings(k, _numReactants);

    const bool isCritical = !((a > 0) && (L <= _nc));
    _isCriticalReaction[k] = isCritical;

    allReactionsAreCritical = allReactionsAreCritical && isCritical;
  }

  // Estimate maximum tau
  double tauP = allReactionsAreCritical ? std::numeric_limits<double>::infinity() : estimateLargestTau();


  // Accept or reject step
  if (tauP <  _acceptanceFactor / a0)
  {
        // reject, execute SSA steps
        for (int i = 0; i < _numStepsSSA; ++i)
        {
            ssaAdvance();
            if (_time >= _simulationLength)
                break;
        }
    }
    else
    {
        // accept, perform tau leap
         
        // calibrate taupp
        double a0c = 0;
        for (size_t k = 0; k < _numReactions; ++k)
        {
            if (_isCriticalReaction[k])
                a0c += _propensities[k];
        }

        const double tauPP = - std::log(_uniformGenerator->getRandomNumber()) / a0c;

        double tau;
        bool anySpeciesNegative = false;
    
        do {
            
            tau = tauP < tauPP ? tauP : tauPP;
            if ( _time + tau > _simulationLength)
                tau = _simulationLength - _time;

            _numFirings.resize(_numReactions, 0);

            for (size_t i = 0; i < _numReactions; ++i)
            {
                if (_isCriticalReaction[i])
                {
                    _numFirings[i] = 0;
                }
                else
                {
                    _poissonGenerator->_mean = _propensities[i] * tau;
                    _numFirings[i] = _poissonGenerator->getRandomNumber();
                }
            }

            if (tauPP <= tauP)
            {
                _cumPropensities.resize(_numReactions);
                double cumulative = 0;
                for (size_t i = 0; i < _numReactions; ++i)
                {
                    if (_isCriticalReaction[i])
                        cumulative += _propensities[i];
                    _cumPropensities[i] = cumulative;
                }

                const double u = a0c * _uniformGenerator->getRandomNumber();
                size_t jc = 0;
                while (jc < _numReactions && (!_isCriticalReaction[jc] || u > _cumPropensities[jc]))
                {
                    ++jc;
                }

                _numFirings[jc] = 1;
            }

            _candidateNumReactants = _numReactants;

            for (size_t i = 0; i < _numReactions; ++i)
            {
                const int ki = _numFirings[i];
                if (ki > 0)
                    _problem->applyChanges(i, _candidateNumReactants, ki);
            }

            anySpeciesNegative = false;
            for (auto candidate : _candidateNumReactants)
            {
                if (candidate < 0)
                {
                    anySpeciesNegative = true;
                    tauP /= 2.;
                    break;
                }
            }
        } while (anySpeciesNegative);

        _time += tau;
        std::swap(_numReactants, _candidateNumReactants);
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

 if (isDefined(js, "Poisson Generator"))
 {
 _poissonGenerator = dynamic_cast<korali::distribution::univariate::Poisson*>(korali::Module::getModule(js["Poisson Generator"], _k));
 _poissonGenerator->applyVariableDefaults();
 _poissonGenerator->applyModuleDefaults(js["Poisson Generator"]);
 _poissonGenerator->setConfiguration(js["Poisson Generator"]);
   eraseValue(js, "Poisson Generator");
 }

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
 if(_poissonGenerator != NULL) _poissonGenerator->getConfiguration(js["Poisson Generator"]);
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
