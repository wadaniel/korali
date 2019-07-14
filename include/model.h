#ifndef _KORALI_MODEL_H_
#define _KORALI_MODEL_H_

namespace Korali
{
 class Model {
  public:

  void addResult(double x) {_self->_results.push_back(x); }
  size_t getVariableCount() { return _self->_computationalVariables.size(); }
  std::vector<double>& getVariables() { return _self->_computationalVariables; }
  std::vector<double>& getResults() { return _self->_results; }
  size_t& getSampleId() { return _self->_sampleId; }

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

  Model* _self;
  size_t _sampleId;
  std::vector<double> _computationalVariables;
  std::vector<double> _statisticalVariables;
  std::vector<double> _results;

  // Constructor / Destructor
  Model() { _self = this; }
 };
}

#endif // _KORALI_MODEL_H
