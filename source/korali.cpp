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
//
 if (_verbosity == KORALI_SILENT)   js["Verbosity"] = "Silent";
 if (_verbosity == KORALI_MINIMAL)  js["Verbosity"] = "Minimal";
 if (_verbosity == KORALI_NORMAL)   js["Verbosity"] = "Normal";
 if (_verbosity == KORALI_DETAILED) js["Verbosity"] = "Detailed";

 _problem->getConfiguration(js);
 _solver->getConfiguration(js);
 _conduit->getConfiguration(js);

 return js;
}

void Korali::Engine::setConfiguration(nlohmann::json js)
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
 _seed = consume(js, { "Seed" }, KORALI_NUMBER, std::to_string(_seed));
 gsl_rng_env_setup();

  _verbosity = KORALI_UNDEFINED;
 std::string vLevel = consume(js, { "Verbosity" }, KORALI_STRING, "Normal");
 if (vLevel == "Silent")   _verbosity = KORALI_SILENT;
 if (vLevel == "Minimal")  _verbosity = KORALI_MINIMAL;
 if (vLevel == "Normal")   _verbosity = KORALI_NORMAL;
 if (vLevel == "Detailed") _verbosity = KORALI_DETAILED;
 if (_verbosity == KORALI_UNDEFINED) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Vebosity Level '%s'.", vLevel.c_str()); exit(-1); }

 // Configure Problem

 _problem = nullptr;
 std::string pName = consume(js, { "Problem" }, KORALI_STRING);
 if (pName == "Direct Evaluation")   { _problem = new Korali::Problem::Direct(js); }
 if (pName == "Bayesian") { _problem = new Korali::Problem::Bayesian(js); }
 if (_problem == nullptr) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Problem '%s'.", pName.c_str()); exit(-1); }

 N = _variables.size();
 if (N == 0) { fprintf(stderr, "[Korali] Error: No variables have been defined.\n"); exit(-1); }

 // Configure Conduit

 int rankCount = 1;

 _conduit = nullptr;
 std::string conduitType =  consume(js, { "Conduit" }, KORALI_STRING, "Semi-Intrusive");

 if (conduitType == "Semi-Intrusive") _conduit = new Korali::Conduit::SemiIntrusive(js["Conduit"]);
 #ifdef _KORALI_USE_MPI
 if (conduitType == "Distributed") _conduit = new Korali::Conduit::Distributed(js["Conduit"]);
 #else
 if (conduitType == "Distributed") { fprintf(stderr, "[Korali] Error: Distributed Conduit selected, but Korali has not been compiled with MPI or UPC++ support.\n"); exit(-1); }
 #endif
 if (conduitType == "Nonintrusive") _conduit = new Korali::Conduit::Nonintrusive(js["Conduit"]);

 if (_conduit == nullptr) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Conduit '%s'.\n", conduitType.c_str()); exit(-1); }

 // Configure Solver

 _solver = nullptr;
 std::string solverName = consume(js, { "Solver" }, KORALI_STRING);
 if (solverName == "CMA-ES")  _solver = new Korali::Solver::CMAES(js, solverName);
 if (solverName == "CCMA-ES") _solver = new Korali::Solver::CMAES(js, solverName);
 if (solverName == "DE")      _solver = new Korali::Solver::DE(js, solverName);
 if (solverName == "MCMC")    _solver = new Korali::Solver::MCMC(js, solverName);
 if (solverName == "DRAM")    _solver = new Korali::Solver::MCMC(js, solverName);
 if (solverName == "TMCMC")   _solver = new Korali::Solver::TMCMC(js);
 if (_solver == nullptr) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Solver '%s'.", solverName.c_str()); exit(-1); }

 if (isEmpty(js) == false)
 {
  fprintf(stderr, "[Korali] Error: Unrecognized Settings for Korali:\n");
  fprintf(stderr, "%s\n", js.dump(2).c_str());
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

 setConfiguration(_js);

 // Creating Results directory
 mkdir("_korali_result", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

 // Running Engine
 _conduit->run();

 if (_conduit->isRoot()) if(_verbosity >= KORALI_MINIMAL) printf("[Korali] Results saved to folder: '_korali_result'\n");
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

 sprintf(fileName, "./_korali_result/s%05d.json", fileId++);

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
