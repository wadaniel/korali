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
 _modelDefined = false;
 _consoleOutputFrequency = 1;
 _resultsOutputFrequency = 1;
 _isFinished = false;
 _solver = nullptr;
 _problem = nullptr;
 _conduit = nullptr;
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

 if (_problem != nullptr) _problem->getConfiguration();
 if (_conduit != nullptr) _conduit->getConfiguration();
 if (_conduit != nullptr) _solver->getConfiguration();

 for (int i = 0; i < _variables.size(); i++) _variables[i]->getConfiguration(_js["Variables"][i]);
}

void Korali::Engine::setConfiguration()
{
 auto js = _js;

 // Configure Conduit
 _conduitType =  consume(_js, { "Conduit", "Type" }, KORALI_STRING, "Simple");
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
 if (_korali_verbosity == KORALI_UNDEFINED) koraliError("Incorrect or undefined Verbosity Level '%s'\n.", vLevel.c_str());

 _result_dir = consume(_js, { "General", "Results Output", "Path" }, KORALI_STRING, "_korali_result");

 // Configure Problem
 _problemType = consume(_js, { "Problem", "Type" }, KORALI_STRING);
 if (_problemType == "Optimization")  _problem = std::make_shared<Korali::Problem::Optimization>();
 if (_problemType == "Sampling") _problem = std::make_shared<Korali::Problem::Sampling>();
 if (_problemType == "Bayesian Inference") _problem = std::make_shared<Korali::Problem::Bayesian>();
 if (_problemType == "Hierarchical Bayesian") _problem = std::make_shared<Korali::Problem::Hierarchical>();
 if (_problem == nullptr) koraliError("Incorrect or undefined Problem '%s'.\n", _problemType.c_str());

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
 _solverType = consume(_js, { "Solver", "Type" }, KORALI_STRING);
 if (_solverType == "CMAES") _solver = std::make_shared<Korali::Solver::CMAES>();
 if (_solverType == "DEA")    _solver = std::make_shared<Korali::Solver::DEA>();
 if (_solverType == "MCMC")   _solver = std::make_shared<Korali::Solver::MCMC>();
 if (_solverType == "TMCMC")  _solver = std::make_shared<Korali::Solver::TMCMC>();
 if (_solver == nullptr) koraliError("Incorrect or undefined Solver '%s'.\n", _solverType.c_str());

 // Setting module configuration
 _problem->setConfiguration();
 _conduit->setConfiguration();
 _solver->setConfiguration();

 // Korali-specific configuration

 _hasComputedGeneration = consume(_js, { "General", "Has Computed Generation" }, KORALI_BOOLEAN, "false");
 _currentGeneration = consume(_js, { "General", "Current Generation" }, KORALI_NUMBER, "0");
 _consoleOutputFrequency = consume(_js, { "General", "Console Output", "Frequency" }, KORALI_NUMBER, std::to_string(_consoleOutputFrequency));
 _resultsOutputFrequency = consume(_js, { "General", "Results Output", "Frequency" }, KORALI_NUMBER, std::to_string(_consoleOutputFrequency));
 _functionEvaluationCount = consume(_js, { "General", "Function Evaluation Count" }, KORALI_NUMBER, "0");
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

void Korali::Engine::addConstraint(std::function<void(Korali::Model&)> constraint)
{
 _constraints.push_back(constraint);
}

void Korali::Engine::start(bool isDryRun)
{
 if(_isFinished) { koraliWarning(KORALI_MINIMAL, "Cannot restart engine, a previous run has already been executed."); return; }
 _k = this; 

 setConfiguration();

 // Creating Results directory
 mkdir(_result_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

 std::chrono::time_point<std::chrono::system_clock> startTime, endTime;

 // Running Engine
 _conduit->initialize();
 _problem->initialize();
 if (_currentGeneration == 0) _solver->initialize();


 if (_conduit->isRoot())
 {
  saveState(_currentGeneration);

  _solver->printGeneration();

  auto startTime = std::chrono::system_clock::now();

  if (isDryRun == false)
  while(_solver->checkTermination() == false)
  {
   auto t0 = std::chrono::system_clock::now();

   _currentGeneration++;
   _solver->runGeneration();
   _hasComputedGeneration = true;

   auto t1 = std::chrono::system_clock::now();

   koraliLog(KORALI_DETAILED, "Generation Time: %.3fs\n", std::chrono::duration<double>(t1-t0).count());

   if (_currentGeneration % _consoleOutputFrequency == 0) _solver->printGeneration();
   if (_currentGeneration % _resultsOutputFrequency == 0) saveState(_currentGeneration);
  }

  auto endTime = std::chrono::system_clock::now();

  saveState(_currentGeneration);
  saveState("final.json");

  _solver->finalize();
  _problem->finalize();
  _conduit->finalize();

  koraliLog(KORALI_MINIMAL, "Total Generations: %lu\n", _currentGeneration);
  koraliLog(KORALI_MINIMAL, "Total Function Evaluations: %lu\n", _functionEvaluationCount);
  koraliLog(KORALI_MINIMAL, "Elapsed Time: %.3fs\n", std::chrono::duration<double>(endTime-startTime).count());
  koraliLog(KORALI_MINIMAL, "Results saved to folder: '%s'\n", _result_dir.c_str());
 }
}

void Korali::Engine::saveState(std::string fileName)
{
 getConfiguration();
 if (!_conduit->isRoot()) return;

 fileName = "./" + _result_dir + "/" + fileName;
 saveJsonToFile(fileName.c_str(), _js);
}

std::string Korali::Engine::getResults(std::string fileName)
{
 auto js = loadJsonFromFile(fileName.c_str());
 return js.dump();
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
