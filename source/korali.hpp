#ifndef _KORALI_HPP_
#define _KORALI_HPP_

#include <vector>
#include <functional>

#ifdef _KORALI_USE_MPI
#include "mpi.h"
#endif

#include "problems/optimization/optimization.hpp"
#include "problems/sampling/sampling.hpp"
#include "problems/bayesian/bayesian.hpp"
#include "problems/hierarchical/hierarchical.hpp"

#include "solvers/cmaes/cmaes.hpp"
#include "solvers/dea/dea.hpp"
#include "solvers/mcmc/mcmc.hpp"
#include "solvers/tmcmc/tmcmc.hpp"

#include "conduits/mpi.hpp"
#include "conduits/simple.hpp"
#include "conduits/external.hpp"

#include "variable.hpp"
#include "auxiliar.hpp"
#include "model.hpp"

#ifdef _KORALI_USE_PYTHON
 #undef _POSIX_C_SOURCE
 #undef _XOPEN_SOURCE
#endif

namespace Korali
{

class Engine {

 public:

 nlohmann::json  _js;
 nlohmann::json& operator[](std::string key) { return _js[key]; }

 bool _modelDefined;
 std::function<void(Korali::Model&)> _model;
 std::vector<std::function<void(Korali::Model&)>> _constraints;

 size_t N; // Variable Count size_t N; // Variable Count
 size_t _currentGeneration;
 size_t _consoleOutputFrequency;
 size_t _resultsOutputFrequency;
 size_t _functionEvaluationCount;
 size_t _runId;
 std::string _runTimestamp;
 bool _hasComputedGeneration;

 std::string _solverType;
 std::string _conduitType;
 std::string _problemType;

 bool _isFinished;

 std::vector<Korali::Variable*> _variables;
 std::shared_ptr<Korali::Conduit::Base> _conduit;
 std::shared_ptr<Korali::Problem::Base> _problem;
 std::shared_ptr<Korali::Solver::Base>  _solver;

 // Model Functions and constructors
 Engine();
 ~Engine();

 // Start functions
 void start(bool isDryRun);
 void run() { start(false); }
 void dry() { start(true);  }

 void setModel(std::function<void(Korali::Model&)> model);
 void addConstraint(std::function<void(Korali::Model&)> constraint);

 // Python Configuration Binding Methods
 KoraliJsonWrapper _wr;
 KoraliJsonWrapper& getItem(const std::string& key)           { _wr._js = &(_js[key]); return _wr;}
 KoraliJsonWrapper& getItem(const unsigned long int& key)     { _wr._js = &(_js[key]); return _wr;}
 void setItem(const std::string& key, const std::string& val) { _js[key] = val; }
 void setItem(const std::string& key, const double& val)      { _js[key] = val; }
 void setItem(const std::string& key, const int& val)         { if(_js[key].is_boolean()) _js[key] = val == true; else _js[key] = val; }
 void setItem(const std::string& key, const bool& val)        { _js[key] = val; }

 void loadState(std::string fileName);
 void saveState(std::string fileName);
 void saveState(int fileId);
 static std::string getResults(std::string fileName);

 size_t _seed;
 std::string _result_dir;

 // Serialization Methods
 void getConfiguration();
 void setConfiguration();
};

extern Engine* _k;

} // namespace Korali

#endif // _KORALI_HPP_
