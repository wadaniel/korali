#ifndef _KORALI_MODEL_BASE_
#define _KORALI_MODEL_BASE_

#include <stdlib.h>
#include "json/json.hpp"

namespace Korali::Model
{

class Base {
 public:

 Base();

 void addResult(double x) {_self->_results.push_back(x); }
 size_t getParameterCount() { return _self->_parameters.size(); }
 std::vector<double>& getParameters() { return _self->_parameters; }
 std::vector<double>& getResults() { return _self->_results; }

 double getParameter(size_t i)
 {
  if (i > _self->_parameters.size())
  {
     fprintf(stderr, "[Korali] Error: Trying to access parameter %lu, when only %lu are provided.\n", i, _self->_parameters.size());
     exit(-1);
   }
  return _self->_parameters[i];
 }

 Base* _self;
 std::vector<double> _parameters;
 std::vector<double> _results;

 // Evaluation Method
 virtual void evaluate(std::vector<double>& parameters, std::vector<double>& results) = 0;

 // Team Creation and Distribution
 size_t _nRanks;
 virtual void initialize(size_t nRanks) { _nRanks = nRanks; };
 virtual size_t getTeamCount() = 0;
 virtual size_t getRanksPerTeam() = 0;

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

class Model;

} // namespace Korali

#endif // _KORALI_MODEL_BASE_
