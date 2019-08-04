#ifndef _KORALI_MODEL_BASE_HPP_
#define _KORALI_MODEL_BASE_HPP_

#ifdef _KORALI_USE_MPI
#include "mpi.h"
#endif

namespace Korali { namespace Model {

class Base {

 private:

 Base* _self;
 size_t _sampleId;
 std::vector<double> _variables;

 public:

 Base() { _self = this; }

 std::vector<double>& getVariables() { return _self->variables; }
 size_t& getSampleId() { return _self->_sampleId; }

 #ifdef _KORALI_USE_MPI
 MPI_Comm _comm;
 MPI_Comm getComm() { return _self->_comm; }
 long int getCommPointer() { return (long int)(&_self->_comm); }
 #endif
};

} } // namespace Korali::Solver

#endif // _KORALI_MODEL_BASE_HPP_
