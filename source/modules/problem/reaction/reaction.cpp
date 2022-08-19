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

  for (size_t idx = 0; idx < _k->_variables.size(); ++idx)
  {
    _reactantNameToIndexMap[_k->_variables[idx]->_name] = idx;
    _initialReactantNumbers.push_back(_k->_variables[idx]->_initialReactantNumber);
  }

  // Parsing user-defined reactions
  for (size_t i = 0; i < _reactions.size(); i++)
  {
    double rate = _reactions[i]["Rate"].get<double>();
    std::string eq = _reactions[i]["Equation"];

    auto reaction = parseReactionString(eq);
    std::vector<int> reactantIds, productIds;
    for (auto &name : reaction.reactantNames)
      reactantIds.push_back(_reactantNameToIndexMap[name]);
    for (auto &name : reaction.productNames)
      productIds.push_back(_reactantNameToIndexMap[name]);

    _reactionVector.emplace_back(rate,
                                 std::move(reactantIds),
                                 std::move(reaction.reactantSCs),
                                 std::move(productIds),
                                 std::move(reaction.productSCs),
                                 std::move(reaction.isReactantReservoir));
  }
}

double Reaction::computePropensity(size_t reactionIndex, std::vector<int> &reactantNumbers) const
{
  const auto &reaction = _reactionVector[reactionIndex];

  double propensity = reaction.rate;

  for (size_t s = 0; s < reaction.reactantIds.size(); ++s)
  {
    const int nu = reaction.reactantStoichiometries[s];
    const int x = reactantNumbers[reaction.reactantIds[s]];

    int numerator = x;
    int denominator = nu;

    for (int k = 1; k < nu; ++k)
    {
      numerator *= x - k;
      denominator *= k;
    }

    propensity *= (double)numerator / denominator;
  }

  return propensity;
}

void Reaction::applyChanges(size_t reactionIndex, std::vector<int> &reactantNumbers, int numFirings) const
{
  const auto &reaction = _reactionVector[reactionIndex];

  for (size_t s = 0; s < reaction.reactantIds.size(); ++s)
  {
    if (!reaction.isReactantReservoir[s])
      reactantNumbers[reaction.reactantIds[s]] -= numFirings * reaction.reactantStoichiometries[s];
  }

  for (size_t s = 0; s < reaction.productIds.size(); ++s)
  {
    reactantNumbers[reaction.productIds[s]] += numFirings * reaction.productStoichiometries[s];
  }

  int total = 0;
  for (size_t s = 0; s < reactantNumbers.size(); ++s)
  {
    total += reactantNumbers[s];
  }
}

void Reaction::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Reactions"))
 {
 _reactions = js["Reactions"].get<knlohmann::json>();

   eraseValue(js, "Reactions");
 }

 if (isDefined(js, "Reactant Name To Index Map"))
 {
 try { _reactantNameToIndexMap = js["Reactant Name To Index Map"].get<std::map<std::string, int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reaction ] \n + Key:    ['Reactant Name To Index Map']\n%s", e.what()); } 
   eraseValue(js, "Reactant Name To Index Map");
 }

 if (isDefined(js, "Initial Reactant Numbers"))
 {
 try { _initialReactantNumbers = js["Initial Reactant Numbers"].get<std::vector<int>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reaction ] \n + Key:    ['Initial Reactant Numbers']\n%s", e.what()); } 
   eraseValue(js, "Initial Reactant Numbers");
 }

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Reactant Number"))
 {
 try { _k->_variables[i]->_initialReactantNumber = _k->_js["Variables"][i]["Initial Reactant Number"].get<int>();
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
   js["Reactions"] = _reactions;
   js["Reactant Name To Index Map"] = _reactantNameToIndexMap;
   js["Initial Reactant Numbers"] = _initialReactantNumbers;
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
