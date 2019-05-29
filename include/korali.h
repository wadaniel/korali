#ifndef _KORALI_H_
#define _KORALI_H_

#include "stdlib.h"
#include <vector>
#include <functional>

#ifdef _KORALI_USE_MPI
#include "mpi.h"
#endif

#include "problems/direct.h"
#include "problems/bayesian.h"

#include "variables/exponential.h"
#include "variables/gamma.h"
#include "variables/gaussian.h"
#include "variables/uniform.h"
#include "variables/cauchy.h"
#include "variables/laplace.h"

#include "solvers/cmaes.h"
#include "solvers/tmcmc.h"

#include "conduits/single.h"
#include "conduits/mpi.h"

#include "json/koralijson.h"

#ifdef _KORALI_USE_PYTHON
 #undef _POSIX_C_SOURCE
 #undef _XOPEN_SOURCE
#endif

enum verbosity { KORALI_SILENT = 0, KORALI_MINIMAL = 1, KORALI_NORMAL = 2, KORALI_DETAILED = 3 };

namespace Korali
{

class Engine {

 public:

 nlohmann::json  _js;
 nlohmann::json& operator[](std::string key) { return _js[key]; }

 std::function<void(Korali::ModelData&)> _model;
 Korali::Conduit::Base* _conduit;
 Korali::Problem::Base* _problem;
 Korali::Solver::Base*  _solver;

 // Model Functions and constructors
 Engine();
 ~Engine();

 void run(std::function<void(Korali::ModelData&)> model);

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
