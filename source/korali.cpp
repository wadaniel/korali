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
 .def("getSampleId",         &Korali::ModelData::getSampleId, pybind11::return_value_policy::reference)
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
 .def("getBoolean", &Korali::KoraliJsonWrapper::getBoolean)
 .def("getString", &Korali::KoraliJsonWrapper::getString)
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

Korali::Engine::Engine() : _solver(nullptr), _problem(nullptr), _conduit(nullptr)
{
 _modelDefined      = false;
 _likelihoodDefined = false;
}

Korali::Engine::~Engine()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Engine::getConfiguration()
{
 auto js = nlohmann::json();
 _js = js;

 _js["Seed"]      = _seed;
 
 if (_verbosity == KORALI_SILENT)   _js["Verbosity"] = "Silent";
 if (_verbosity == KORALI_MINIMAL)  _js["Verbosity"] = "Minimal";
 if (_verbosity == KORALI_NORMAL)   _js["Verbosity"] = "Normal";
 if (_verbosity == KORALI_DETAILED) _js["Verbosity"] = "Detailed";
 
 _js["Result Directory"] = _result_dir;
 _js["Termination Criteria"]["Max Generations"] = maxGenerations;
 _js["Termination Criteria"]["Max Function Evaluations"] = maxFunctionEvaluations;
 _js["Current Generation"] = currentGeneration;
 _js["Console Output Frequency"] = consoleOutputFrequency;
 _js["File Output Frequency"] = fileOutputFrequency;
 _js["Function Evaluation Count"] = functionEvaluationCount;
 _js["Is Finished"] = _isFinished;
 for (int i = 0; i < _variables.size(); i++) _js["Variables"][i]["Name"] = _variables[i]->_name;

 if (_problem != nullptr) _problem->getConfiguration();
 if (_conduit != nullptr) _conduit->getConfiguration();
 if (_conduit != nullptr) _solver->getConfiguration();
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

 maxGenerations = consume(_js, { "Termination Criteria", "Max Generations" }, KORALI_NUMBER, "5000000");
 currentGeneration = consume(_js, { "Current Generation" }, KORALI_NUMBER, "0");
 consoleOutputFrequency = consume(_js, { "Console Output Frequency" }, KORALI_NUMBER, "1");
 fileOutputFrequency = consume(_js, { "File Output Frequency" }, KORALI_NUMBER, "1");
 functionEvaluationCount = consume(_js, { "Function Evaluation Count" }, KORALI_NUMBER, "0");
 maxFunctionEvaluations = consume(_js, { "Termination Criteria", "Max Function Evaluations" }, KORALI_NUMBER, "50000000");
 _isFinished = consume(_js, { "Is Finished" }, KORALI_BOOLEAN, "false");

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
 if (pName == "Direct Evaluation")  _problem = std::make_shared<Korali::Problem::Direct>(); 
 if (pName == "Bayesian")           _problem = std::make_shared<Korali::Problem::Bayesian>();
 if (_problem == nullptr) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Problem '%s'.", pName.c_str()); exit(-1); }

 // Create Variables

 if (isArray(_js, { "Variables" } ))
 for (size_t i = 0; i < _js["Variables"].size(); i++)
 {
  _k->_variables.push_back(new Korali::Variable());
  _k->_variables[i]->setConfiguration(_js["Variables"][i]);
 }

 N = _variables.size();
 if (N == 0) { fprintf(stderr, "[Korali] Error: No variables have been defined.\n"); exit(-1); }

 // Configure Conduit

 int rankCount = 1;

 std::string conduitType =  consume(_js, { "Conduit", "Type" }, KORALI_STRING, "Linked");

 if (conduitType == "Linked") _conduit = std::make_shared<Korali::Conduit::Linked>();
 if (conduitType == "External") _conduit = std::make_shared<Korali::Conduit::External>();

 if (_conduit == nullptr) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Conduit '%s'.\n", conduitType.c_str()); exit(-1); }

 // Configure Solver

 std::string solverName = consume(_js, { "Solver" }, KORALI_STRING);
 if (solverName == "CMAES")  _solver = std::make_shared<Korali::Solver::CMAES>();
 if (solverName == "DEA")    _solver = std::make_shared<Korali::Solver::DEA>();
 if (solverName == "MCMC")   _solver = std::make_shared<Korali::Solver::MCMC>();
 if (solverName == "TMCMC")  _solver = std::make_shared<Korali::Solver::TMCMC>();
 if (_solver == nullptr) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Solver '%s'.", solverName.c_str()); exit(-1); }

 // Setting module configuration
 _problem->setConfiguration();
 _conduit->setConfiguration();
 _solver->setConfiguration();
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

 if (isEmpty(_js) == false)
 {
  fprintf(stderr, "[Korali] Error: Unrecognized Settings for Korali:\n");
  fprintf(stderr, "%s\n", _js.dump(2).c_str());
  exit(-1);
 }

 // Creating Results directory
 mkdir(_result_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

 std::chrono::time_point<std::chrono::system_clock> startTime, endTime;

 // Running Engine
 _conduit->initialize();
 _problem->initialize();
 if (currentGeneration == 0) _solver->initialize();

 if (_conduit->isRoot())
 {
  saveState(currentGeneration);

  _solver->printGeneration();

  auto startTime = std::chrono::system_clock::now();

  while(checkTermination() == false)
  {
   auto t0 = std::chrono::system_clock::now();

   _solver->runGeneration();
   currentGeneration++;

   auto t1 = std::chrono::system_clock::now();

   if(_verbosity >= KORALI_DETAILED) printf("[Korali] Generation Time: %.3fs\n", std::chrono::duration<double>(t1-t0).count());

   if (currentGeneration % consoleOutputFrequency == 0) _solver->printGeneration();
   if (currentGeneration % fileOutputFrequency    == 0) saveState(currentGeneration);

  }

  auto endTime = std::chrono::system_clock::now();

  saveState(currentGeneration);

  _solver->finalize();
  _problem->finalize();
  _conduit->finalize();

  for (size_t i = 0; i < N; i++) delete _variables[i];
  _variables.clear();

  if(_verbosity >= KORALI_MINIMAL) printf("[Korali] Total Generations: %lu\n", currentGeneration);
  if(_verbosity >= KORALI_MINIMAL) printf("[Korali] Total Function Evaluations: %lu\n", functionEvaluationCount);
  if(_verbosity >= KORALI_MINIMAL) printf("[Korali] Elapsed Time: %.3fs\n", std::chrono::duration<double>(endTime-startTime).count());
  if(_verbosity >= KORALI_MINIMAL) printf("[Korali] Results saved to folder: '%s'\n", _result_dir.c_str());
 }
}

void Korali::Engine::addConstraint(fcon fconstraint)
{
 _fconstraints.push_back(fconstraint);
}


void Korali::Engine::saveState(std::string fileName)
{
 if (!_conduit->isRoot()) return;

 getConfiguration();
 saveJsonToFile(fileName.c_str(), _js);
}

void Korali::Engine::saveState(int fileId)
{
 if (!_conduit->isRoot()) return;

 char fileName[256];

 sprintf(fileName, "./%s/s%05d.json", _result_dir.c_str(), fileId);

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

bool Korali::Engine::checkTermination()
{
   _isFinished = _solver->checkTermination();
   
   if (currentGeneration >= maxGenerations) 
   { if(_verbosity >= KORALI_MINIMAL) printf("[Korali] Maximum generation count reached (%lu).\n", maxGenerations); _isFinished = true; }
   
   if (functionEvaluationCount >= maxFunctionEvaluations) 
   { if(_verbosity >= KORALI_MINIMAL) printf("[Korali] Maximum function evaluation count reached (%lu).\n", maxFunctionEvaluations); _isFinished = true; }

   return _isFinished;
}
