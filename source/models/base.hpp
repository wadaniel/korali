#ifndef _KORALI_MODEL_BASE_HPP_
#define _KORALI_MODEL_BASE_HPP_

#include "auxiliar.hpp"

#ifdef _KORALI_USE_MPI
#include "mpi.h"
#endif

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
  if (i >= _self->_variables.size()) koraliError("Requesting variable with index %lu, but problem has only %lu variables.\n", i, _self->_variables.size());
  return _self->_variables[i];
 }

 size_t& getSampleId() { return _self->_sampleId; }

 #ifdef _KORALI_USE_MPI
 MPI_Comm _comm;
 MPI_Comm getComm() { return _self->_comm; }
 long int getCommPointer() { return (long int)(&_self->_comm); }
 #endif
};

} } // namespace Korali::Solver

#endif // _KORALI_MODEL_BASE_HPP_
