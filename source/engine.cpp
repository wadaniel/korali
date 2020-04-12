#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/conduit/conduit.hpp"
#include "modules/solver/solver.hpp"
#include "modules/problem/problem.hpp"
#include "modules/conduit/distributed/distributed.hpp"
#include "auxiliar/koraliJson.hpp"

std::vector<std::function<void(korali::Sample&)>*> korali::_functionVector;

korali::Engine::Engine()
{
 _cumulativeTime = 0.0;
 _thread = co_active();

 // Turn Off GSL Error Handler
 gsl_set_error_handler_off();
}

void korali::Engine::initialize()
{
 // Instantiating Engine logger.
 _logger = new korali::Logger();

 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Profiling']['Detail']")) _js["Profiling"]["Detail"] = "None";
 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Profiling']['Path']")) _js["Profiling"]["Path"] = "./profiling.json";
 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Profiling']['Frequency']")) _js["Profiling"]["Frequency"] = 60.0;
 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Conduit']['Type']")) _js["Conduit"]["Type"] = "Sequential";
 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Dry Run']")) _js["Dry Run"] = false;

 _isDryRun = _js["Dry Run"];
 _profilingPath = _js["Profiling"]["Path"];
 _profilingDetail = _js["Profiling"]["Detail"];
 _profilingFrequency = _js["Profiling"]["Frequency"];

 for (size_t i = 0; i < _experimentVector.size(); i++)
 {
  _experimentVector[i]->_experimentId = i;
  _experimentVector[i]->_engine = this;
  _experimentVector[i]->initialize();
  _experimentVector[i]->_isFinished = false;
  std::string fileName = "./log.txt";
  //std::string fileName = "./" + _experimentVector[i]->_resultsOutputPath + "/log.txt";
  if (_experimentVector.size() > 1)  _experimentVector[i]->_logFile = fopen(fileName.c_str(), "a");
  if (_experimentVector.size() == 1) _experimentVector[i]->_logFile = stdout;
 }

 // Check configuration correctness
 auto js = _js.getJson();
 if (korali::JsonInterface::isDefined(js, "['Conduit']")) korali::JsonInterface::eraseValue(js, "['Conduit']");
 if (korali::JsonInterface::isDefined(js, "['Dry Run']")) korali::JsonInterface::eraseValue(js, "['Dry Run']");
 if (korali::JsonInterface::isDefined(js, "['Conduit']['Type']")) korali::JsonInterface::eraseValue(js, "['Conduit']['Type']");
 if (korali::JsonInterface::isDefined(js, "['Profiling']['Detail']")) korali::JsonInterface::eraseValue(js, "['Profiling']['Detail']");
 if (korali::JsonInterface::isDefined(js, "['Profiling']['Path']")) korali::JsonInterface::eraseValue(js, "['Profiling']['Path']");
 if (korali::JsonInterface::isDefined(js, "['Profiling']['Frequency']")) korali::JsonInterface::eraseValue(js, "['Profiling']['Frequency']");
 if (korali::JsonInterface::isEmpty(js) == false) _logger->logError("Unrecognized settings for Korali's Engine: \n%s\n", js.dump(2).c_str());
}

void korali::Engine::run()
{
 // Checking if its a dry run and return if it is
 if (_isDryRun) return;

 // Only initialize conduit if the Engine being ran is the first one in the process
 if (_conduit == NULL)
 {
  // Configuring conduit
  auto conduit = dynamic_cast<korali::Conduit*>(getModule(_js["Conduit"]));

  // Initializing conduit server
  conduit->initServer();

  // Assigning pointer after starting workers, so they can initialize their own conduit
  _conduit = conduit;

  // Recovering Conduit configuration in case of restart
  _conduit->getConfiguration(_js.getJson()["Conduit"]);
 }

 if (_conduit->isRoot())
 {
  // Stacking current Engine
  _conduit->stackEngine(this);

  // Setting base time for profiling.
  _startTime = std::chrono::high_resolution_clock::now();
  _profilingLastSave = std::chrono::high_resolution_clock::now();

  if (_experimentVector.size() > 1) for (size_t i = 0; i < _experimentVector.size(); i++) _logger->logInfo(_experimentVector[0]->_consoleOutputVerbosity, "Starting Experiment %lu...\n", i);

  while(true)
  {
   bool executed = false;
   for (size_t i = 0; i < _experimentVector.size(); i++)
   if (_experimentVector[i]->_isFinished == false)
   {
    _currentExperiment = _experimentVector[i];
    co_switch(_experimentVector[i]->_thread);
    executed = true;
    saveProfilingInfo(false);
    if (_experimentVector.size() > 1) if (_experimentVector[i]->_isFinished == true) _logger->logInfo(_experimentVector[0]->_consoleOutputVerbosity, "Experiment %lu has finished.\n", i);
   }
   if (executed == false) break;
  }

  _endTime = std::chrono::high_resolution_clock::now();

  if (_experimentVector.size() > 1) _logger->logInfo(_experimentVector[0]->_consoleOutputVerbosity, "All jobs have finished correctly.\n");
  if (_experimentVector.size() > 1) _logger->logInfo(_experimentVector[0]->_consoleOutputVerbosity, "Elapsed Time: %.3fs\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-_startTime).count());

  saveProfilingInfo(true);
  _cumulativeTime += std::chrono::duration<double>(_endTime-_startTime).count();

  // Finalizing experiments
  for (size_t i = 0; i < _experimentVector.size(); i++) _experimentVector[i]->finalize();

  // Removing the current engine to the conduit's engine stack
  _conduit->popEngine();
 }

 // Finalizing Conduit if last engine in the stack
 if (_conduit->_engineStack.size() == 0)
 {
  _conduit->finalize();
  _conduit = NULL;
 }
}

void korali::Engine::saveProfilingInfo(bool forceSave)
{
 if (_profilingDetail == "Full")
 {
  auto currTime = std::chrono::high_resolution_clock::now();
  double timeSinceLast = std::chrono::duration<double>(currTime-_profilingLastSave).count();
  if ((timeSinceLast > _profilingFrequency) || forceSave)
  {
    double elapsedTime = std::chrono::duration<double>(currTime-_startTime).count();
    __profiler["Experiment Count"] = _experimentVector.size();
    __profiler["Elapsed Time"] = elapsedTime + _cumulativeTime;
    korali::JsonInterface::saveJsonToFile(_profilingPath.c_str(), __profiler);
    _profilingLastSave = std::chrono::high_resolution_clock::now();
  }
 }
}

void korali::Engine::run(korali::Experiment& experiment)
{
 _experimentVector.clear();
 _experimentVector.push_back(experiment._k);
 initialize();
 run();
}

void korali::Engine::run(std::vector<korali::Experiment>& experiments)
{
 _experimentVector.clear();
 for (size_t i = 0; i < experiments.size(); i++) _experimentVector.push_back(experiments[i]._k);
 initialize();
 run();
}

#ifdef _KORALI_USE_MPI
long int korali::Engine::getMPICommPointer() { return (long int)(&__KoraliTeamComm); }
#endif

knlohmann::json& korali::Engine::operator[](const std::string& key) { return _js[key]; }
knlohmann::json& korali::Engine::operator[](const unsigned long int& key) { return _js[key]; }
pybind11::object korali::Engine::getItem(pybind11::object key) { return _js.getItem(key); }
void korali::Engine::setItem(pybind11::object key, pybind11::object val) { _js.setItem(key, val); }

void korali::Engine::serialize(knlohmann::json& js)
{
  for (size_t i = 0; i < _experimentVector.size(); i++)
  {
   _experimentVector[i]->getConfiguration(_experimentVector[i]->_js.getJson());
   js["Experiment Vector"][i] = _experimentVector[i]->_js.getJson();
  }
}

korali::Engine* korali::Engine::deserialize(knlohmann::json& js)
{
 auto k = new korali::Engine;

 for (size_t i = 0; i < js["Experiment Vector"].size(); i++)
 {
  auto e = new korali::Experiment;
  e->_js.getJson() = js["Experiment Vector"][i];
  k->_experimentVector.push_back(e);
 }

 k->initialize();

 return k;
}

PYBIND11_MODULE(libkorali, m)
{
 #ifdef _KORALI_USE_MPI
 m.def("getMPICommPointer", &korali::Engine::getMPICommPointer, pybind11::return_value_policy::reference);
 #endif

 pybind11::class_<korali::Engine>(m, "Engine")
  .def(pybind11::init<>())
  .def("run", pybind11::overload_cast<korali::Experiment&>(&korali::Engine::run))
  .def("run", pybind11::overload_cast<std::vector<korali::Experiment>&>(&korali::Engine::run))
  .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::Engine::getItem), pybind11::return_value_policy::reference)
  .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::Engine::setItem), pybind11::return_value_policy::reference);

 pybind11::class_<korali::KoraliJson>(m, "koraliJson")
  .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::KoraliJson::getItem), pybind11::return_value_policy::reference)
  .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::KoraliJson::setItem), pybind11::return_value_policy::reference);

 pybind11::class_<korali::Sample>(m, "Sample")
  .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::Sample::getItem), pybind11::return_value_policy::reference)
  .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::Sample::setItem), pybind11::return_value_policy::reference);

 pybind11::class_<korali::Experiment>(m, "Experiment")
   .def(pybind11::init<>())
   .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::Experiment::getItem), pybind11::return_value_policy::reference)
   .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::Experiment::setItem), pybind11::return_value_policy::reference)
   .def("loadState",   pybind11::overload_cast<std::string>(&korali::Experiment::loadState))
   .def("loadState",   pybind11::overload_cast<>(&korali::Experiment::loadState));
}

