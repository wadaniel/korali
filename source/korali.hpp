#ifndef _KORALI_HPP_
#define _KORALI_HPP_

#include <vector>
#include <functional>

#include "distributions/cauchy/cauchy.hpp"
#include "distributions/exponential/exponential.hpp"
#include "distributions/gamma/gamma.hpp"
#include "distributions/gaussian/gaussian.hpp"
#include "distributions/geometric/geometric.hpp"
#include "distributions/multivariate/multivariate.hpp"
#include "distributions/laplace/laplace.hpp"
#include "distributions/uniform/uniform.hpp"
#include "distributions/lognormal/lognormal.hpp"

#include "problems/optimization/optimization.hpp"
#include "problems/sampling/sampling.hpp"
#include "problems/bayesian/bayesian.hpp"
#include "problems/hierarchical/hierarchical.hpp"

#include "solvers/cmaes/cmaes.hpp"
#include "solvers/dea/dea.hpp"
#include "solvers/mcmc/mcmc.hpp"
#include "solvers/tmcmc/tmcmc.hpp"

#include "conduits/mpi/mpi.hpp"
#include "conduits/simple/simple.hpp"
#include "conduits/external/external.hpp"

#include "models/constraint/constraint.hpp"
#include "models/direct/direct.hpp"
#include "models/likelihood/likelihood.hpp"
#include "models/reference/reference.hpp"

#include "variable/variable.hpp"

#include "auxiliars/json.hpp"
#include "auxiliars/koralijson.hpp"
#include "auxiliars/logger.hpp"
#include "auxiliars/python.hpp"

namespace Korali
{

class Engine {

 public:

 nlohmann::json  _js;
 nlohmann::json& operator[](std::string key) { return _js[key]; }

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
 std::string _modelType;

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

 std::function<void(Korali::Model::Direct&)> _directModel;
 std::function<void(Korali::Model::Likelihood&)> _likelihoodModel;
 std::function<void(Korali::Model::Reference&)> _referenceModel;
 std::vector<std::function<void(Korali::Model::Constraint&)>> _constraints;

 void setDirectModel(std::function<void(Korali::Model::Direct&)> model);
 void setLikelihoodModel(std::function<void(Korali::Model::Likelihood&)> model);
 void setReferenceModel(std::function<void(Korali::Model::Reference&)> model);
 void addConstraint(std::function<void(Korali::Model::Constraint&)> constraint);

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

 size_t _seed;
 std::string _result_dir;

 // Serialization Methods
 void getConfiguration();
 void setConfiguration();
};

extern Engine* _k;

} // namespace Korali

#endif // _KORALI_HPP_
