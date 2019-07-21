#include "korali.hpp"
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdio>

Korali::Engine* Korali::_k;
Korali::verbosity _korali_verbosity;

/************************************************************************/
/*                  Python Binding Declarations                         */
/************************************************************************/

#ifdef _KORALI_USE_PYTHON

#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

PYBIND11_MODULE(libkorali, m) {
 pybind11::class_<Korali::Model>(m, "Model")
  .def("getVariable",      &Korali::Model::getVariable, pybind11::return_value_policy::reference)
  .def("getVariableCount", &Korali::Model::getVariableCount, pybind11::return_value_policy::reference)
  .def("getVariables",     &Korali::Model::getVariables, pybind11::return_value_policy::reference)
  .def("getResults",       &Korali::Model::getResults, pybind11::return_value_policy::reference)
 .def("getSampleId",         &Korali::Model::getSampleId, pybind11::return_value_policy::reference)
  #ifdef _KORALI_USE_MPI
  .def("getCommPointer",   &Korali::Model::getCommPointer)
  #endif
  .def("addResult",        &Korali::Model::addResult, pybind11::return_value_policy::reference);

 pybind11::class_<Korali::Engine>(m, "Engine")
 .def(pybind11::init<>())
 .def("__getitem__", pybind11::overload_cast<const std::string&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
 .def("__getitem__", pybind11::overload_cast<const unsigned long int&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const std::string&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const double&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const int&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const bool&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("run", &Korali::Engine::run)
 .def("dry", &Korali::Engine::dry)
 .def("setModel",      &Korali::Engine::setModel, pybind11::return_value_policy::reference)
 .def("setLikelihood", &Korali::Engine::setLikelihood, pybind11::return_value_policy::reference)
 .def("addConstraint", &Korali::Engine::addConstraint, pybind11::return_value_policy::reference)
 .def("addSubProblem", &Korali::Engine::addSubProblem, pybind11::return_value_policy::reference)
 .def("loadState",     &Korali::Engine::loadState, pybind11::return_value_policy::reference)
 .def("loadConfig",    &Korali::Engine::loadConfig, pybind11::return_value_policy::reference);

 pybind11::class_<Korali::KoraliJsonWrapper>(m, "__KoraliJsonWrapper")
 .def(pybind11::init<>())
 .def("getValue", &Korali::KoraliJsonWrapper::getValue)
 .def("getBoolean", &Korali::KoraliJsonWrapper::getBoolean)
 .def("getString", &Korali::KoraliJsonWrapper::getString)
 .def("getArray", &Korali::KoraliJsonWrapper::getArray)
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
 _runId             = 0;
 _modelDefined      = false;
 _likelihoodDefined = false;
 consoleOutputFrequency = 1;
 resultsOutputFrequency = 1;
 _isFinished = false;
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

 _js["General"]["Random Seed"]   = _seed;
 _js["General"]["Run ID"] = _runId;

 time_t rawtime; time (&rawtime);
 std::string curTime(ctime(&rawtime));
 _runTimestamp = curTime.substr(0, curTime.size()-1);
 _js["General"]["Timestamp"] = _runTimestamp;
 
 if (_korali_verbosity == KORALI_SILENT)   _js["General"]["Console Output"]["Verbosity"] = "Silent";
 if (_korali_verbosity == KORALI_MINIMAL)  _js["General"]["Console Output"]["Verbosity"] = "Minimal";
 if (_korali_verbosity == KORALI_NORMAL)   _js["General"]["Console Output"]["Verbosity"] = "Normal";
 if (_korali_verbosity == KORALI_DETAILED) _js["General"]["Console Output"]["Verbosity"] = "Detailed";
 
 _js["General"]["Max Generations"] = maxGenerations;
 _js["General"]["Max Function Evaluations"] = maxFunctionEvaluations;
 _js["General"]["Current Generation"] = currentGeneration;
 _js["General"]["Console Output"]["Frequency"] = consoleOutputFrequency;
 _js["General"]["Results Output"]["Frequency"] = resultsOutputFrequency;
 _js["General"]["Results Output"]["Path"] = _result_dir;
 _js["General"]["Function Evaluation Count"] = functionEvaluationCount;
 _js["General"]["Is Finished"] = _isFinished;
 for (int i = 0; i < _variables.size(); i++) _js["Variables"][i]["Name"] = _variables[i]->_name;

 if (_problem != nullptr) _problem->getConfiguration();
 if (_conduit != nullptr) _conduit->getConfiguration();
 if (_conduit != nullptr) _solver->getConfiguration();
}

void Korali::Engine::setConfiguration()
{
 auto js = _js;

 // Configure Conduit
 std::string conduitType =  consume(_js, { "Conduit", "Type" }, KORALI_STRING, "Simple");
 if (conduitType == "Simple") _conduit = std::make_shared<Korali::Conduit::Simple>();
 if (conduitType == "External") _conduit = std::make_shared<Korali::Conduit::External>();
 if (conduitType == "MPI") _conduit = std::make_shared<Korali::Conduit::MPI>();
 if (_conduit == nullptr)
 {
  _conduit = std::make_shared<Korali::Conduit::Simple>();
  koraliError("Incorrect or undefined Conduit '%s'.\n", conduitType.c_str());
 }

 // Configure Korali Engine
 _variables.clear();

 // Setting Run Timestamp
 _runTimestamp = "";
 time_t rawtime; time (&rawtime);
 std::string curTime(ctime(&rawtime));
 _runTimestamp = consume(_js, { "General", "Timestamp" }, KORALI_STRING, curTime.substr(0, curTime.size()-1));

 // Setting Run Hash Id
 std::hash<std::string> hasher;
 _runId = hasher(_runTimestamp);
 _runId = consume(_js, { "General", "Run ID" }, KORALI_NUMBER, std::to_string(_runId));

 // Initializing Seed and GSL Random Environment
 _seed  = consume(_js, { "General", "Random Seed" }, KORALI_NUMBER, std::to_string(_runId));
 gsl_rng_env_setup();

 // Configuring Korali Settings
 _korali_verbosity = KORALI_UNDEFINED;
 std::string vLevel = consume(_js, { "General", "Console Output", "Verbosity" }, KORALI_STRING, "Normal");
 if (vLevel == "Silent")   _korali_verbosity = KORALI_SILENT;
 if (vLevel == "Minimal")  _korali_verbosity = KORALI_MINIMAL;
 if (vLevel == "Normal")   _korali_verbosity = KORALI_NORMAL;
 if (vLevel == "Detailed") _korali_verbosity = KORALI_DETAILED;
 if (_korali_verbosity == KORALI_UNDEFINED) koraliError("Incorrect or undefined Verbosity Level '%s'\n.", vLevel.c_str());

 _result_dir = consume(_js, { "General", "Results Output", "Path" }, KORALI_STRING, "_korali_result");

 // Configure Problem
 std::string pName = consume(_js, { "Problem", "Type" }, KORALI_STRING);
 if (pName == "Optimization")  _problem = std::make_shared<Korali::Problem::Optimization>();
 if (pName == "Constrained Optimization")  _problem = std::make_shared<Korali::Problem::Constrained>();
 if (pName == "Sampling") _problem = std::make_shared<Korali::Problem::Sampling>();
 if (pName == "Bayesian Inference") _problem = std::make_shared<Korali::Problem::Bayesian>();
 if (pName == "Hierarchical Bayesian") _problem = std::make_shared<Korali::Problem::Hierarchical>();
 if (_problem == nullptr) koraliError("Incorrect or undefined Problem '%s'.\n", pName.c_str());

 // Create Variables

 if (isArray(_js, { "Variables" } ))
 for (size_t i = 0; i < _js["Variables"].size(); i++)
 {
  _k->_variables.push_back(new Korali::Variable());
  _k->_js["Variables"][i]["Prior Distribution"]["Seed"] = _k->_seed++;
  _k->_variables[i]->setConfiguration(_js["Variables"][i]);
 }

 N = _variables.size();
 if (N == 0) koraliError("No variables have been defined.\n");

 // Configure Solver
 std::string solverName = consume(_js, { "Solver", "Type" }, KORALI_STRING);
 if (solverName == "CMAES")  _solver = std::make_shared<Korali::Solver::CMAES>();
 if (solverName == "CCMAES") _solver = std::make_shared<Korali::Solver::CCMAES>();
 if (solverName == "DEA")    _solver = std::make_shared<Korali::Solver::DEA>();
 if (solverName == "MCMC")   _solver = std::make_shared<Korali::Solver::MCMC>();
 if (solverName == "TMCMC")  _solver = std::make_shared<Korali::Solver::TMCMC>();
 if (_solver == nullptr) koraliError("Incorrect or undefined Solver '%s'.", solverName.c_str());

 // Setting module configuration
 _problem->setConfiguration();
 _conduit->setConfiguration();
 _solver->setConfiguration();

 // Korali-specific configuration

 maxGenerations = consume(_js, { "General", "Max Generations" }, KORALI_NUMBER, "5000000");
 currentGeneration = consume(_js, { "General", "Current Generation" }, KORALI_NUMBER, "0");
 consoleOutputFrequency = consume(_js, { "General", "Console Output", "Frequency" }, KORALI_NUMBER, std::to_string(consoleOutputFrequency));
 resultsOutputFrequency = consume(_js, { "General", "Results Output", "Frequency" }, KORALI_NUMBER, std::to_string(consoleOutputFrequency));
 functionEvaluationCount = consume(_js, { "General", "Function Evaluation Count" }, KORALI_NUMBER, "0");
 maxFunctionEvaluations = consume(_js, { "General", "Max Function Evaluations" }, KORALI_NUMBER, "50000000");
 _isFinished = consume(_js, { "General", "Is Finished" }, KORALI_BOOLEAN, "false");

 if (isEmpty(_js) == false) koraliError("Unrecognized Settings for Korali:\n %s \n", _js.dump(2).c_str());
 _js = js;
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Engine::setModel(std::function<void(Korali::Model&)> model)
{
 _model = model;
 _modelDefined = true;
}

void Korali::Engine::setLikelihood(std::function<void(Korali::Model&)> likelihood)
{
 _model = likelihood;
 _likelihoodDefined = true;
}

void Korali::Engine::start(bool isDryRun)
{
 _k = this; 

 setConfiguration();

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

  if (isDryRun == false)
  while(checkTermination() == false)
  {
   auto t0 = std::chrono::system_clock::now();

   _solver->runGeneration();
   currentGeneration++;

   auto t1 = std::chrono::system_clock::now();

   koraliLog(KORALI_DETAILED, "Generation Time: %.3fs\n", std::chrono::duration<double>(t1-t0).count());

   if (currentGeneration % consoleOutputFrequency == 0) _solver->printGeneration();
   if (currentGeneration % resultsOutputFrequency == 0) saveState(currentGeneration);
  }

  auto endTime = std::chrono::system_clock::now();

  saveState(currentGeneration);

  _solver->finalize();
  _problem->finalize();
  _conduit->finalize();

  koraliLog(KORALI_MINIMAL, "Total Generations: %lu\n", currentGeneration);
  koraliLog(KORALI_MINIMAL, "Total Function Evaluations: %lu\n", functionEvaluationCount);
  koraliLog(KORALI_MINIMAL, "Elapsed Time: %.3fs\n", std::chrono::duration<double>(endTime-startTime).count());
  koraliLog(KORALI_MINIMAL, "Results saved to folder: '%s'\n", _result_dir.c_str());
 }
}

void Korali::Engine::addConstraint(fcon fconstraint)
{
 _constraints.push_back(fconstraint);
}

void Korali::Engine::addSubProblem(Korali::Engine& problem)
{
 problem.getConfiguration();
 _subProblems.push_back(problem._js);
}

void Korali::Engine::saveState(std::string fileName)
{
 getConfiguration();
 if (!_conduit->isRoot()) return;

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
   { koraliLog(KORALI_MINIMAL, "Maximum generation count reached (%lu).\n", maxGenerations); _isFinished = true; }
   
   if (functionEvaluationCount >= maxFunctionEvaluations) 
   { koraliLog(KORALI_MINIMAL, "Maximum function evaluation count reached (%lu).\n", maxFunctionEvaluations); _isFinished = true; }

   return _isFinished;
}
