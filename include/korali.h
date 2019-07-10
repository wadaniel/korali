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

#include "solvers/cmaes.h"
#include "solvers/dea.h"
#include "solvers/mcmc.h"
#include "solvers/tmcmc.h"

#include "conduits/semi-intrusive.h"
#include "conduits/distributed.h"
#include "conduits/nonintrusive.h"

#include "json/koralijson.h"
#include "variable.h"

#ifdef _KORALI_USE_PYTHON
 #undef _POSIX_C_SOURCE
 #undef _XOPEN_SOURCE
#endif

enum verbosity { KORALI_UNDEFINED = -1, KORALI_SILENT = 0, KORALI_MINIMAL = 1, KORALI_NORMAL = 2, KORALI_DETAILED = 3 };

namespace Korali
{

using fcon = std::function<double(std::vector<double>&)>;

class Engine {

 public:

 nlohmann::json  _js;
 nlohmann::json& operator[](std::string key) { return _js[key]; }

 bool _modelDefined;
 bool _likelihoodDefined;
 std::function<void(ModelData&)> _model;
 std::vector<fcon> _fconstraints;

 size_t N; // Variable Count size_t N; // Variable Count
 size_t currentGeneration;
 size_t maxGenerations;
 size_t consoleOutputFrequency;
 size_t fileOutputFrequency; //TODO: should be resultOutputFrequency (aligned with resultfolder) (DW)
 size_t functionEvaluationCount;
 size_t maxFunctionEvaluations;
 bool _isFinished;

 std::vector<Korali::Variable*> _variables;
 Korali::Conduit::Base* _conduit;
 Korali::Problem::Base* _problem;
 Korali::Solver::Base*  _solver;

 // Model Functions and constructors
 Engine();
 ~Engine();

 void run();
 void setModel(std::function<void(Korali::ModelData&)> model);
 void setLikelihood(std::function<void(Korali::ModelData&)> model);
 void addConstraint(fcon fconstraint);

 // Python Configuration Binding Methods
 KoraliJsonWrapper _wr;
 KoraliJsonWrapper& getItem(const std::string& key)           { _wr._js = &(_js[key]); return _wr;}
 KoraliJsonWrapper& getItem(const unsigned long int& key)     { _wr._js = &(_js[key]); return _wr;}
 void setItem(const std::string& key, const std::string& val) { _js[key] = val; }
 void setItem(const std::string& key, const double& val)      { _js[key] = val; }
 void setItem(const std::string& key, const int& val)         { if(_js[key].is_boolean()) _js[key] = val == true; else _js[key] = val; }
 void setItem(const std::string& key, const bool& val)        { _js[key] = val; }

 void loadConfig(std::string fileName);
 void loadState(std::string fileName);
 void saveState(std::string fileName);
 void saveState(int fileId);

 bool checkTermination();

 size_t _seed;
 int _verbosity;
 std::string _result_dir;

 // Serialization Methods
 void getConfiguration();
 void setConfiguration();
};

extern Engine* _k;

} // namespace Korali

#endif // _KORALI_H_
