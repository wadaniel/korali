#include "korali.h"
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdio>

Korali::Engine* Korali::_k;

/************************************************************************/
/*                  Python Binding Declarations                         */
/************************************************************************/

#ifdef _KORALI_USE_PYTHON

#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

PYBIND11_MODULE(libkorali, m) {
 pybind11::class_<Korali::ModelData>(m, "ModelData")
  .def("getVariable",      &Korali::ModelData::getVariable, pybind11::return_value_policy::reference)
  .def("getVariableCount", &Korali::ModelData::getVariableCount, pybind11::return_value_policy::reference)
  .def("getVariables",     &Korali::ModelData::getVariables, pybind11::return_value_policy::reference)
  .def("getResults",       &Korali::ModelData::getResults, pybind11::return_value_policy::reference)
 .def("getHashId",         &Korali::ModelData::getHashId, pybind11::return_value_policy::reference)
  #ifdef _KORALI_USE_MPI
  .def("getCommPointer",   &Korali::ModelData::getCommPointer)
  #endif
  .def("addResult",        &Korali::ModelData::addResult, pybind11::return_value_policy::reference);

 pybind11::class_<Korali::Engine>(m, "Engine")
 .def(pybind11::init<>())
 .def("__getitem__", pybind11::overload_cast<const std::string&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
 .def("__getitem__", pybind11::overload_cast<const unsigned long int&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const std::string&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const double&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const int&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const bool&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("run", &Korali::Engine::run)
 .def("setModel",      &Korali::Engine::setModel, pybind11::return_value_policy::reference)
 .def("setLikelihood", &Korali::Engine::setLikelihood, pybind11::return_value_policy::reference)
 .def("addConstraint", &Korali::Engine::addConstraint, pybind11::return_value_policy::reference)
 .def("loadState",     &Korali::Engine::loadState, pybind11::return_value_policy::reference)
 .def("loadConfig",    &Korali::Engine::loadConfig, pybind11::return_value_policy::reference);

 pybind11::class_<Korali::KoraliJsonWrapper>(m, "__KoraliJsonWrapper")
 .def(pybind11::init<>())
 .def("getValue", &Korali::KoraliJsonWrapper::getValue)
 .def("__getitem__", pybind11::overload_cast<const std::string&>(&Korali::KoraliJsonWrapper::getItem), pybind11::return_value_policy::reference)
 .def("__getitem__", pybind11::overload_cast<const unsigned long int&>(&Korali::KoraliJsonWrapper::getItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const std::string&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const double&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const int&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const bool&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const std::vector<double>&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const int&, const std::string&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const int&, const double&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const int&, const int&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const int&, const bool&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference);
}

#endif

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Engine::Engine()
{
 _modelDefined = false;
 _likelihoodDefined = false;

 _problem = nullptr;
 _conduit = nullptr;
 _solver  = nullptr;
}

Korali::Engine::~Engine()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Engine::getConfiguration()
{
 auto js = nlohmann::json();

 js["Seed"]      = _seed;
 
 if (_verbosity == KORALI_SILENT)   js["Verbosity"] = "Silent";
 if (_verbosity == KORALI_MINIMAL)  js["Verbosity"] = "Minimal";
 if (_verbosity == KORALI_NORMAL)   js["Verbosity"] = "Normal";
 if (_verbosity == KORALI_DETAILED) js["Verbosity"] = "Detailed";
 
 js["Result Directory"] = _result_dir;

 if (_problem != nullptr) _problem->getConfiguration(js);
 if (_conduit != nullptr) _conduit->getConfiguration(js);

 return js;
}

void Korali::Engine::setConfiguration()
{
 // Configure Korali Engine
 _variables.clear();

 // Initializing Seed and GSL Random Environment
 _seed = 0;
 FILE *fid = fopen("/dev/random", "rb");
 if (fid != nullptr)
 {
  fread(&_seed, 1, sizeof(size_t), fid);
  fclose(fid);
 }
 _seed = consume(_js, { "Seed" }, KORALI_NUMBER, std::to_string(_seed));
 gsl_rng_env_setup();

  _verbosity = KORALI_UNDEFINED;
 std::string vLevel = consume(_js, { "Verbosity" }, KORALI_STRING, "Normal");
 if (vLevel == "Silent")   _verbosity = KORALI_SILENT;
 if (vLevel == "Minimal")  _verbosity = KORALI_MINIMAL;
 if (vLevel == "Normal")   _verbosity = KORALI_NORMAL;
 if (vLevel == "Detailed") _verbosity = KORALI_DETAILED;
 if (_verbosity == KORALI_UNDEFINED) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Vebosity Level '%s'.", vLevel.c_str()); exit(-1); }

 _result_dir = consume(_js, { "Result Directory" }, KORALI_STRING, "_korali_result");

 // Configure Problem

 std::string pName = consume(_js, { "Problem" }, KORALI_STRING);
 if (pName == "Direct Evaluation")   { _problem = new Korali::Problem::Direct(_js); }
 if (pName == "Bayesian") { _problem = new Korali::Problem::Bayesian(_js); }
 if (_problem == nullptr) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Problem '%s'.", pName.c_str()); exit(-1); }

 N = _variables.size();
 if (N == 0) { fprintf(stderr, "[Korali] Error: No variables have been defined.\n"); exit(-1); }

 // Configure Conduit

 int rankCount = 1;

 std::string conduitType =  consume(_js, { "Conduit" }, KORALI_STRING, "Semi-Intrusive");

 if (conduitType == "Semi-Intrusive") _conduit = new Korali::Conduit::SemiIntrusive(_js["Conduit"]);
 #ifdef _KORALI_USE_MPI
 if (conduitType == "Distributed") _conduit = new Korali::Conduit::Distributed(_js["Conduit"]);
 #else
 if (conduitType == "Distributed") { fprintf(stderr, "[Korali] Error: Distributed Conduit selected, but Korali has not been compiled with MPI or UPC++ support.\n"); exit(-1); }
 #endif
 if (conduitType == "Nonintrusive") _conduit = new Korali::Conduit::Nonintrusive(_js["Conduit"]);

 if (_conduit == nullptr) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Conduit '%s'.\n", conduitType.c_str()); exit(-1); }

 // Configure Solver

 std::string solverName = consume(_js, { "Solver" }, KORALI_STRING);
 if (solverName == "CMA-ES")  _solver = new Korali::Solver::CMAES();
 if (solverName == "DE")      _solver = new Korali::Solver::DE();
 if (solverName == "MCMC")    _solver = new Korali::Solver::MCMC();
 if (solverName == "TMCMC")   _solver = new Korali::Solver::TMCMC();
 if (_solver == nullptr) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Solver '%s'.", solverName.c_str()); exit(-1); }

 // Setting module configuration
// _problem->setConfiguration();
// _conduit->setConfiguration();
 _solver->setConfiguration();

 if (isEmpty(_js) == false)
 {
  fprintf(stderr, "[Korali] Error: Unrecognized Settings for Korali:\n");
  fprintf(stderr, "%s\n", _js.dump(2).c_str());
  exit(-1);
 }

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Engine::setModel(std::function<void(Korali::ModelData&)> model)
{
 _model = model;
 _modelDefined = true;
}

void Korali::Engine::setLikelihood(std::function<void(Korali::ModelData&)> likelihood)
{
 _model = likelihood;
 _likelihoodDefined = true;
}

void Korali::Engine::run()
{
 _k = this;

 setConfiguration();

 // Creating Results directory
 mkdir(_result_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

 std::chrono::time_point<std::chrono::system_clock> startTime, endTime;
 std::chrono::time_point<std::chrono::system_clock> t0, t1, t2, t3;

 // Running Engine
 _conduit->initialize();
 _solver->initialize();

 startTime = std::chrono::system_clock::now();

 currentGeneration = 0;

 while(!_solver->checkTermination())
 {
   t0 = std::chrono::system_clock::now();

   _solver->runGeneration();
   currentGeneration++;

   t1 = std::chrono::system_clock::now();

   _solver->printGeneration();
 }

 endTime = std::chrono::system_clock::now();

 _solver->finalize();
 _conduit->finalize();

 if (_conduit->isRoot()) if(_verbosity >= KORALI_MINIMAL) printf("[Korali] Results saved to folder: '%s'\n", _result_dir.c_str());
}

void Korali::Engine::addConstraint(fcon fconstraint)
{
 _fconstraints.push_back(fconstraint);
}


void Korali::Engine::saveState(std::string fileName)
{
 if (!_conduit->isRoot()) return;

 _js = getConfiguration();
 saveJsonToFile(fileName.c_str(), _js);
}

void Korali::Engine::saveState(int fileId)
{
 if (!_conduit->isRoot()) return;

 char fileName[256];

 sprintf(fileName, "./%s/s%05d.json", _result_dir.c_str(), fileId++);

 saveState(fileName);
}

void Korali::Engine::loadState(std::string fileName)
{
 _js = loadJsonFromFile(fileName.c_str());
}

void Korali::Engine::loadConfig(std::string fileName)
{
 _js = loadJsonFromFile(fileName.c_str());
 if (isDefined(_js, {"State"})) _js.erase("State");
}
