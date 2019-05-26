#ifndef _KORALI_CONDUIT_BASE_H_
#define _KORALI_CONDUIT_BASE_H_

#include <stdlib.h>
#include "auxiliar/json/json.hpp"

namespace Korali::Conduit
{
class Base {
 public:

 virtual void run() = 0;
 virtual void evaluateSample(double* sampleArray, size_t sampleId) = 0;
 virtual void checkProgress() = 0;
 virtual bool isRoot() = 0;

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

class Conduit;

} // namespace Korali

namespace Korali
{
 class ModelData {
  public:

  void addResult(double x) {_self->_results.push_back(x); }
  size_t getParameterCount() { return _self->_parameters.size(); }
  std::vector<double>& getParameters() { return _self->_parameters; }
  std::vector<double>& getResults() { return _self->_results; }

  #ifdef _KORALI_USE_MPI
  MPI_Comm _comm;
  MPI_Comm getComm() { return _self->_comm; }
  long int getCommPointer() { return (long int)(&_self->_comm); }
  #endif

  double getParameter(size_t i)
  {
   if (i >= _self->_parameters.size())
   {
      fprintf(stderr, "[Korali] Error: Trying to access parameter %lu, when only %lu are provided.\n", i, _self->_parameters.size());
      exit(-1);
    }
   return _self->_parameters[i];
  }

  ModelData* _self;
  std::vector<double> _parameters;
  std::vector<double> _results;

  // Constructor / Destructor
  ModelData() { _self = this; }
 };
}

#endif // _KORALI_CONDUIT_BASE_H_

