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
 consoleOutputFrequency = 1;
 resultsOutputFrequency = 1;
 _isFinished = false;
 _solver = nullptr;
 _problem = nullptr;
 _conduit = nullptr;
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

 if (_problem != nullptr) _problem->getConfiguration();
 if (_conduit != nullptr) _conduit->getConfiguration();
 if (_conduit != nullptr) _solver->getConfiguration();

 for (int i = 0; i < _variables.size(); i++) _variables[i]->getConfiguration(_js["Variables"][i]);
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
 if (solverName == "CMAES") _solver = std::make_shared<Korali::Solver::CMAES>();
 if (solverName == "DEA")    _solver = std::make_shared<Korali::Solver::DEA>();
 if (solverName == "MCMC")   _solver = std::make_shared<Korali::Solver::MCMC>();
 if (solverName == "TMCMC")  _solver = std::make_shared<Korali::Solver::TMCMC>();
 if (_solver == nullptr) koraliError("Incorrect or undefined Solver '%s'.\n", solverName.c_str());

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

   currentGeneration++;
   _solver->runGeneration();

   auto t1 = std::chrono::system_clock::now();

   koraliLog(KORALI_DETAILED, "Generation Time: %.3fs\n", std::chrono::duration<double>(t1-t0).count());

   if (currentGeneration % consoleOutputFrequency == 0) _solver->printGeneration();
   if (currentGeneration % resultsOutputFrequency == 0) saveState(currentGeneration);
  }

  auto endTime = std::chrono::system_clock::now();

  saveState(currentGeneration);
  saveState("final.json");

  _solver->finalize();
  _problem->finalize();
  _conduit->finalize();

  koraliLog(KORALI_MINIMAL, "Total Generations: %lu\n", currentGeneration);
  koraliLog(KORALI_MINIMAL, "Total Function Evaluations: %lu\n", functionEvaluationCount);
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

bool Korali::Engine::checkTermination()
{
   _isFinished = _solver->checkTermination();
   
   if (currentGeneration >= maxGenerations) 
   { koraliLog(KORALI_MINIMAL, "Maximum generation count reached (%lu).\n", maxGenerations); _isFinished = true; }
   
   if (functionEvaluationCount >= maxFunctionEvaluations) 
   { koraliLog(KORALI_MINIMAL, "Maximum function evaluation count reached (%lu).\n", maxFunctionEvaluations); _isFinished = true; }

   return _isFinished;
}
