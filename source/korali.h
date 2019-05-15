#ifndef _KORALI_H_
#define _KORALI_H_

#include "stdlib.h"
#include <vector>
#include <functional>

#include "problems/direct/direct.h"
#include "problems/bayesian/bayesian.h"

#include "variables/exponential/exponential.h"
#include "variables/gamma/gamma.h"
#include "variables/gaussian/gaussian.h"
#include "variables/uniform/uniform.h"
#include "variables/cauchy/cauchy.h"
#include "variables/laplace/laplace.h"

#include "solvers/cmaes/cmaes.h"
#include "solvers/tmcmc/tmcmc.h"

#include "conduits/sequential/sequential.h"
#include "conduits/multithread/multithread.h"
#include "conduits/upcxx/upcxx.h"

#include "models/sequential/sequential.h"

#include "koralijson/koralijson.h"

#ifdef _KORALI_USE_MPI
 #include "mpi.h"
#endif

#ifdef _KORALI_USE_PYTHON
 #undef _POSIX_C_SOURCE
 #undef _XOPEN_SOURCE
#endif

enum verbosity { KORALI_SILENT = 0, KORALI_MINIMAL = 1, KORALI_NORMAL = 2, KORALI_DETAILED = 3 };

namespace Korali
{

class modelData
{
 public:

 modelData() { _self = this; }

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

 modelData* _self;
 std::vector<double> _parameters;
 std::vector<double> _results;

 #ifdef _KORALI_USE_MPI
 MPI_Comm _comm;
 #endif
};

class Engine {

 public:

 nlohmann::json  _js;
 nlohmann::json& operator[](std::string key) { return _js[key]; }

 std::function<void(modelData&)> _model;
 Korali::Conduit::Base* _conduit;
 Korali::Problem::Base* _problem;
 Korali::Solver::Base*  _solver;

 // Model Functions and constructors
 Engine();
 ~Engine();

 void run(std::function<void(modelData&)> model);

 // Python Configuration Binding Methods
 KoraliJsonWrapper _wr;
 KoraliJsonWrapper& getItem(const std::string& key)           { _wr._js = &(_js[key]); return _wr;}
 KoraliJsonWrapper& getItem(const unsigned long int& key)     { _wr._js = &(_js[key]); return _wr;}
 void setItem(const std::string& key, const std::string& val) { _js[key] = val; }
 void setItem(const std::string& key, const double& val)      { _js[key] = val; }
 void setItem(const std::string& key, const int& val)         { _js[key] = val; }
 void setItem(const std::string& key, const bool& val)        { _js[key] = val; }

 void loadConfig(std::string fileName);
 void loadState(std::string fileName);
 void saveState(std::string fileName);
 void saveState();

 size_t _seed;
 int _verbosity;
 size_t _outputFrequency;

 size_t _currentFileId;

 // Serialization Methods
 nlohmann::json getConfiguration();
 void setConfiguration(nlohmann::json js);
};

extern Engine* _k;

} // namespace Korali

#endif // _KORALI_H_
