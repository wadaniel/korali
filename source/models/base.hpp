#ifndef _KORALI_MODEL_BASE_HPP_
#define _KORALI_MODEL_BASE_HPP_

#include "auxiliars/logger.hpp"

namespace Korali { namespace Model {

class Base {

 private:

 public:

 Base* _self;
 size_t _sampleId;
 std::vector<double> _variables;

 Base() { _self = this; }

 std::vector<double>& getVariables() { return _self->_variables; }
 double getVariable(size_t i)
 {
  if (i >= _self->_variables.size()) Korali::logError("Requesting variable with index %lu, but problem has only %lu variables.\n", i, _self->_variables.size());
  return _self->_variables[i];
 }

 size_t& getSampleId() { return _self->_sampleId; }

};

} } // namespace Korali::Solver

#endif // _KORALI_MODEL_BASE_HPP_
