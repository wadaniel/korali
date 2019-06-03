#ifndef _KORALI_CONDUIT_BASE_H_
#define _KORALI_CONDUIT_BASE_H_

#include <stdlib.h>
#include "json/json.hpp"

namespace Korali { namespace Conduit {

class Base {
 public:

 virtual void run() = 0;
 virtual void evaluateSample(double* sampleArray, size_t sampleId) = 0;
 virtual void checkProgress() = 0;
 virtual bool isRoot() = 0;

 size_t _currentSample;

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

class Conduit;

} } // namespace Korali::Conduit

namespace Korali
{
 class ModelData {
  public:

  void addResult(double x) {_self->_results.push_back(x); }
  size_t getVariableCount() { return _self->_computationalVariables.size(); }
  std::vector<double>& getVariables() { return _self->_computationalVariables; }
  std::vector<double>& getResults() { return _self->_results; }
  size_t& getHashId() { return _self->_hashId; }

  #ifdef _KORALI_USE_MPI
  MPI_Comm _comm;
  MPI_Comm getComm() { return _self->_comm; }
  long int getCommPointer() { return (long int)(&_self->_comm); }
  #endif

  double getVariable(size_t i)
  {
   if (i >= _self->_computationalVariables.size())
   {
      fprintf(stderr, "[Korali] Error: Trying to access parameter %lu, when only %lu are provided.\n", i, _self->_computationalVariables.size());
      exit(-1);
    }
   return _self->_computationalVariables[i];
  }

  ModelData* _self;
  size_t _hashId;
  std::vector<double> _computationalVariables;
  std::vector<double> _statisticalVariables;
  std::vector<double> _results;

  // Constructor / Destructor
  ModelData() { _self = this; }
 };
}

#endif // _KORALI_CONDUIT_BASE_H_

