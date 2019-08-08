#include "korali.hpp"
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdio>

Korali::Engine* Korali::_k;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Engine::Engine()
{
 _runId        = 0;
 _consoleOutputFrequency = 1;
 _resultsOutputFrequency = 1;
 _isFinished = false;
 _solver = nullptr;
 _problem = nullptr;
 _conduit = nullptr;
 _modelType = "Unassigned";
 _hasComputedGeneration = false;
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
 _js["General"]["Has Computed Generation"] = _hasComputedGeneration;

 time_t rawtime; time (&rawtime);
 std::string curTime(ctime(&rawtime));
 _runTimestamp = curTime.substr(0, curTime.size()-1);
 _js["General"]["Timestamp"] = _runTimestamp;
 
 if (_korali_verbosity == KORALI_SILENT)   _js["General"]["Console Output"]["Verbosity"] = "Silent";
 if (_korali_verbosity == KORALI_MINIMAL)  _js["General"]["Console Output"]["Verbosity"] = "Minimal";
 if (_korali_verbosity == KORALI_NORMAL)   _js["General"]["Console Output"]["Verbosity"] = "Normal";
 if (_korali_verbosity == KORALI_DETAILED) _js["General"]["Console Output"]["Verbosity"] = "Detailed";
 
 _js["General"]["Current Generation"] = _currentGeneration;
 _js["General"]["Console Output"]["Frequency"] = _consoleOutputFrequency;
 _js["General"]["Results Output"]["Frequency"] = _resultsOutputFrequency;
 _js["General"]["Results Output"]["Path"] = _result_dir;
 _js["General"]["Function Evaluation Count"] = _functionEvaluationCount;
 _js["General"]["Is Finished"] = _isFinished;

 _js["Solver"]["Type"] = _solverType;
 _js["Conduit"]["Type"] = _conduitType;
 _js["Problem"]["Type"] = _problemType;

 if (_problem != nullptr) _problem->getConfiguration(_js["Problem"]);
 if (_conduit != nullptr) _conduit->getConfiguration(_js["Conduit"]);
 if (_conduit != nullptr) _solver->getConfiguration(_js["Solver"]);

 for (int i = 0; i < _variables.size(); i++) _variables[i]->getConfiguration(_js["Variables"][i]);
}

void Korali::Engine::setConfiguration()
{
 auto js = _js;

 // Obtaining module selections
 _solverType = consume(_js, { "Solver", "Type" }, KORALI_STRING);
 _problemType = consume(_js, { "Problem", "Type" }, KORALI_STRING);
 _conduitType =  consume(_js, { "Conduit", "Type" }, KORALI_STRING, "Simple");

 // Configure Conduit
 if (_conduitType == "Simple") _conduit = std::make_shared<Korali::Conduit::Simple>();
 if (_conduitType == "External") _conduit = std::make_shared<Korali::Conduit::External>();
 if (_conduitType == "MPI") _conduit = std::make_shared<Korali::Conduit::MPI>();
 if (_conduit == nullptr) {  fprintf(stderr, "Incorrect or undefined Conduit '%s'.\n", _conduitType.c_str()); exit(-1); }

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
 if (_korali_verbosity == KORALI_UNDEFINED) Korali::logError("Incorrect or undefined Verbosity Level '%s'\n.", vLevel.c_str());

 _result_dir = consume(_js, { "General", "Results Output", "Path" }, KORALI_STRING, "_korali_result");

 // Configure Problem
 if (_problemType == "Optimization")  _problem = std::make_shared<Korali::Problem::Optimization>();
 if (_problemType == "Sampling") _problem = std::make_shared<Korali::Problem::Sampling>();
 if (_problemType == "Bayesian Inference") _problem = std::make_shared<Korali::Problem::Bayesian>();
 if (_problemType == "Hierarchical Bayesian") _problem = std::make_shared<Korali::Problem::Hierarchical>();
 if (_problem == nullptr) Korali::logError("Incorrect or undefined Problem '%s'.\n", _problemType.c_str());

 // Configure Solver

 if (_solverType == "CMAES")  _solver = std::make_shared<Korali::Solver::CMAES>();
 if (_solverType == "DEA")    _solver = std::make_shared<Korali::Solver::DEA>();
 if (_solverType == "MCMC")   _solver = std::make_shared<Korali::Solver::MCMC>();
 if (_solverType == "TMCMC")  _solver = std::make_shared<Korali::Solver::TMCMC>();
 if (_solver == nullptr) Korali::logError("Incorrect or undefined Solver '%s'.\n", _solverType.c_str());

 // Create Variables
 if (isArray(_js, { "Variables" } ))
  for (size_t i = 0; i < _js["Variables"].size(); i++) _variables.push_back(new Korali::Variable());

 N = _variables.size();
 if (N == 0) Korali::logError("No variables have been defined.\n");

 // Setting configuration for the modules
 for (size_t i = 0; i < N; i++) _variables[i]->setConfiguration(_js["Variables"][i]);
 _problem->setConfiguration(_js["Problem"]);
 _conduit->setConfiguration(_js["Conduit"]);
 _solver->setConfiguration(_js["Solver"]);

 // Korali-specific configuration
 _hasComputedGeneration = consume(_js, { "General", "Has Computed Generation" }, KORALI_BOOLEAN, "false");
 _currentGeneration = consume(_js, { "General", "Current Generation" }, KORALI_NUMBER, "0");
 _consoleOutputFrequency = consume(_js, { "General", "Console Output", "Frequency" }, KORALI_NUMBER, std::to_string(_consoleOutputFrequency));
 _resultsOutputFrequency = consume(_js, { "General", "Results Output", "Frequency" }, KORALI_NUMBER, std::to_string(_consoleOutputFrequency));
 _functionEvaluationCount = consume(_js, { "General", "Function Evaluation Count" }, KORALI_NUMBER, "0");
 _isFinished = consume(_js, { "General", "Is Finished" }, KORALI_BOOLEAN, "false");

 if (isEmpty(_js) == false) Korali::logError("Unrecognized Settings for Korali:\n %s \n", _js.dump(2).c_str());
 _js = js;
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Engine::setDirectModel(std::function<void(Korali::Model::Direct&)> model)
{
 _modelType = "Direct";
 _directModel = model;
}

void Korali::Engine::setLikelihoodModel(std::function<void(Korali::Model::Likelihood&)> model)
{
 _modelType = "Likelihood";
 _likelihoodModel = model;
}

void Korali::Engine::setReferenceModel(std::function<void(Korali::Model::Reference&)> model)
{
 _modelType = "Reference";
 _referenceModel = model;
}

void Korali::Engine::addConstraint(std::function<void(Korali::Model::Constraint&)> constraint)
{
 _constraints.push_back(constraint);
}

void Korali::Engine::start(bool isDryRun)
{
 if(_isFinished) { Korali::logWarning(KORALI_MINIMAL, "Cannot restart engine, a previous run has already been executed.\n"); return; }
 _k = this; 

 setConfiguration();

 // Creating Results directory
 mkdir(_result_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

 std::chrono::time_point<std::chrono::system_clock> startTime, endTime;

 // Running Engine
 _problem->initialize();
 if (_currentGeneration == 0) _solver->initialize();
 _conduit->initialize();

 if (_conduit->isRoot())
 {
  saveState(_currentGeneration);

  auto startTime = std::chrono::system_clock::now();

  if (isDryRun == false)
  while(_solver->checkTermination() == false)
  {
   auto t0 = std::chrono::system_clock::now();

   _currentGeneration++;
   _solver->runGeneration();
   _hasComputedGeneration = true;

   auto t1 = std::chrono::system_clock::now();

   if (_currentGeneration % _consoleOutputFrequency == 0)
   {
    Korali::logInfo(KORALI_MINIMAL,  "--------------------------------------------------------------------\n");
    Korali::logInfo(KORALI_MINIMAL,  "Generation: #%zu\n", _currentGeneration);
    Korali::logInfo(KORALI_DETAILED, "Generation Time: %.3fs\n", std::chrono::duration<double>(t1-t0).count());
    _solver->printGeneration();
   }

   if (_currentGeneration % _resultsOutputFrequency == 0) saveState(_currentGeneration);
  }

  auto endTime = std::chrono::system_clock::now();

  saveState(_currentGeneration);
  saveState("final.json");

  Korali::logInfo(KORALI_MINIMAL, "--------------------------------------------------------------------\n");
  Korali::logInfo(KORALI_MINIMAL, "%s finished correctly.\n", _solverType.c_str(), _currentGeneration);

  _solver->finalize();
  _problem->finalize();
  _conduit->finalize();

  Korali::logInfo(KORALI_MINIMAL, "--------------------------------------------------------------------\n");
  Korali::logInfo(KORALI_MINIMAL, "Total Generations: %lu\n", _currentGeneration);
  Korali::logInfo(KORALI_MINIMAL, "Total Function Evaluations: %lu\n", _functionEvaluationCount);
  Korali::logInfo(KORALI_MINIMAL, "Elapsed Time: %.3fs\n", std::chrono::duration<double>(endTime-startTime).count());
  Korali::logInfo(KORALI_MINIMAL, "Results saved to folder: '%s'\n", _result_dir.c_str());
 }
}

void Korali::Engine::saveState(std::string fileName)
{
 getConfiguration();
 if (!_conduit->isRoot()) return;

 fileName = "./" + _result_dir + "/" + fileName;
 saveJsonToFile(fileName.c_str(), _js);
}

void Korali::Engine::saveState(int fileId)
{
 if (!_conduit->isRoot()) return;

 char fileName[256];

 sprintf(fileName, "s%05d.json", fileId);

 saveState(fileName);
}

void Korali::Engine::loadState(std::string fileName)
{
 _js = loadJsonFromFile(fileName.c_str());
}
