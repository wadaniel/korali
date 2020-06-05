#include "engine.hpp"
#include "auxiliar/fs.hpp"
#include "auxiliar/koraliJson.hpp"
#include "modules/conduit/conduit.hpp"
#include "modules/conduit/distributed/distributed.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/solver.hpp"
#include <sys/stat.h>
#include <sys/types.h>

std::vector<std::function<void(korali::Sample &)> *> korali::_functionVector;
std::stack<korali::Engine *> korali::_engineStack;
bool korali::isPythonActive = 0;

korali::Engine::Engine()
{
  _cumulativeTime = 0.0;
  _thread = co_active();

  // Turn Off GSL Error Handler
  gsl_set_error_handler_off();
}

void korali::Engine::initialize()
{
  // Setting Engine configuration defaults
  if (!isDefined(_js.getJson(), "Profiling", "Detail")) _js["Profiling"]["Detail"] = "None";
  if (!isDefined(_js.getJson(), "Profiling", "Path")) _js["Profiling"]["Path"] = "./profiling.json";
  if (!isDefined(_js.getJson(), "Profiling", "Frequency")) _js["Profiling"]["Frequency"] = 60.0;
  if (!isDefined(_js.getJson(), "Conduit", "Type")) _js["Conduit"]["Type"] = "Sequential";
  if (!isDefined(_js.getJson(), "Dry Run")) _js["Dry Run"] = false;

  // Loading configuration values
  _isDryRun = _js["Dry Run"];
  _profilingPath = _js["Profiling"]["Path"];
  _profilingDetail = _js["Profiling"]["Detail"];
  _profilingFrequency = _js["Profiling"]["Frequency"];

  // Initializing experiment's configuration
  for (size_t i = 0; i < _experimentVector.size(); i++)
  {
    _experimentVector[i]->_experimentId = i;
    _experimentVector[i]->_engine = this;
    _experimentVector[i]->initialize();
    _experimentVector[i]->_isFinished = false;
  }

  // Check configuration correctness
  auto js = _js.getJson();
  try
  {
    if (isDefined(js, "Verbosity")) eraseValue(js, "Verbosity");
    if (isDefined(js, "Conduit")) eraseValue(js, "Conduit");
    if (isDefined(js, "Dry Run")) eraseValue(js, "Dry Run");
    if (isDefined(js, "Conduit", "Type")) eraseValue(js, "Conduit", "Type");
    if (isDefined(js, "Profiling", "Detail")) eraseValue(js, "Profiling", "Detail");
    if (isDefined(js, "Profiling", "Path")) eraseValue(js, "Profiling", "Path");
    if (isDefined(js, "Profiling", "Frequency")) eraseValue(js, "Profiling", "Frequency");
  }
  catch (const std::exception &e)
  {
    KORALI_LOG_ERROR("[Korali] Error parsing Korali Engine's parameters. Reason:\n%s", e.what());
  }

  if (isEmpty(js) == false) KORALI_LOG_ERROR("Unrecognized settings for Korali's Engine: \n%s\n", js.dump(2).c_str());
}

void korali::Engine::run()
{
  // Checking if its a dry run and return if it is
  if (_isDryRun) return;

  // Only initialize conduit if the Engine being ran is the first one in the process
  if (_conduit == NULL)
  {
    try
    {
      // Configuring conduit
      auto conduit = dynamic_cast<korali::Conduit *>(getModule(_js["Conduit"], _k));

      // Initializing conduit server
      conduit->initServer();

      // Assigning pointer after starting workers, so they can initialize their own conduit
      _conduit = conduit;

      // Recovering Conduit configuration in case of restart
      _conduit->getConfiguration(_js.getJson()["Conduit"]);
    }
    catch (const std::exception &e)
    {
      KORALI_LOG_ERROR("Could not initialize execution conduit. Reason:\n%s", e.what());
    }
  }

  if (_conduit->isRoot())
  {
    // Initializing experiment's result folder / log files (or stdout)
    for (size_t i = 0; i < _experimentVector.size(); i++)
    {
      _experimentVector[i]->_logFile = stdout;

      // If experiments are saving results to file, create folder
      if (_experimentVector[i]->_fileOutputEnabled)
      {
        if (!korali::dirExists(_experimentVector[i]->_fileOutputPath)) korali::mkdir(_experimentVector[i]->_fileOutputPath);

        // Setting log files to be saved on to the log folder
        std::string fileName = "./" + _experimentVector[i]->_fileOutputPath + "/log.txt";
        if (_experimentVector.size() > 1) _experimentVector[i]->_logFile = fopen(fileName.c_str(), "a");

        if (_experimentVector[i]->_logFile == NULL)
          KORALI_LOG_ERROR("[Korali] Could not create log file (%s) for experiment %lu.\n", fileName.c_str(), i);
      }

      _experimentVector[i]->_logger = new Logger(_experimentVector[i]->_consoleOutputVerbosity, _experimentVector[i]->_logFile);
    }

    // (Engine-Side) Adding engine to the stack to support Korali-in-Korali execution
    _engineStack.push(this);

    // (Workers-Side) Adding engine to the stack to support Korali-in-Korali execution
    _conduit->stackEngine(this);

    // Setting base time for profiling.
    _startTime = std::chrono::high_resolution_clock::now();
    _profilingLastSave = std::chrono::high_resolution_clock::now();

    if (_experimentVector.size() > 1)
      for (size_t i = 0; i < _experimentVector.size(); i++) _experimentVector[i]->_logger->logInfo("Minimal", "Starting Experiment %lu...\n", i);

    while (true)
    {
      // Checking for break signals coming from Python
      bool executed = false;
      for (size_t i = 0; i < _experimentVector.size(); i++)
        if (_experimentVector[i]->_isFinished == false)
        {
          _currentExperiment = _experimentVector[i];
          co_switch(_experimentVector[i]->_thread);
          executed = true;
          saveProfilingInfo(false);
          if (_experimentVector.size() > 1)
            if (_experimentVector[i]->_isFinished == true) _experimentVector[i]->_logger->logInfo("Minimal", "Experiment %lu has finished.\n", i);
        }
      if (executed == false) break;
    }

    _endTime = std::chrono::high_resolution_clock::now();

    if (_experimentVector.size() > 1) _experimentVector[0]->_logger->logInfo("Minimal", "All jobs have finished correctly.\n");
    if (_experimentVector.size() > 1) _experimentVector[0]->_logger->logInfo("Minimal", "Elapsed Time: %.3fs\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _startTime).count());

    saveProfilingInfo(true);
    _cumulativeTime += std::chrono::duration<double>(_endTime - _startTime).count();

    // Finalizing experiments
    for (size_t i = 0; i < _experimentVector.size(); i++)
    {
      _experimentVector[i]->finalize();
      if (_experimentVector.size() > 1) fclose(_experimentVector[i]->_logFile);
    }

    // (Engine-Side) Removing the current engine to the conduit's engine stack
    _engineStack.pop();

    // (Workers-Side) Removing the current engine to the conduit's engine stack
    _conduit->popEngine();
  }

  // Finalizing Conduit if last engine in the stack
  if (_engineStack.size() == 0)
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
    double timeSinceLast = std::chrono::duration<double>(currTime - _profilingLastSave).count();
    if ((timeSinceLast > _profilingFrequency) || forceSave)
    {
      double elapsedTime = std::chrono::duration<double>(currTime - _startTime).count();
      __profiler["Experiment Count"] = _experimentVector.size();
      __profiler["Elapsed Time"] = elapsedTime + _cumulativeTime;
      saveJsonToFile(_profilingPath.c_str(), __profiler);
      _profilingLastSave = std::chrono::high_resolution_clock::now();
    }
  }
}

void korali::Engine::run(korali::Experiment &experiment)
{
  experiment._js["Current Generation"] = 0;
  resume(experiment);
}

void korali::Engine::run(std::vector<korali::Experiment> &experiments)
{
  for (size_t i = 0; i < experiments.size(); i++) experiments[i]._js["Current Generation"] = 0;
  resume(experiments);
}

void korali::Engine::resume(korali::Experiment &experiment)
{
  _experimentVector.clear();
  _experimentVector.push_back(experiment._k);
  initialize();
  run();
}

void korali::Engine::resume(std::vector<korali::Experiment> &experiments)
{
  _experimentVector.clear();
  for (size_t i = 0; i < experiments.size(); i++) _experimentVector.push_back(experiments[i]._k);
  initialize();
  run();
}

void korali::Engine::initialize(korali::Experiment &experiment)
{
  _experimentVector.clear();
  _experimentVector.push_back(experiment._k);
  initialize();
}

void korali::Engine::serialize(knlohmann::json &js)
{
  for (size_t i = 0; i < _experimentVector.size(); i++)
  {
    _experimentVector[i]->getConfiguration(_experimentVector[i]->_js.getJson());
    js["Experiment Vector"][i] = _experimentVector[i]->_js.getJson();
  }
}

korali::Engine *korali::Engine::deserialize(knlohmann::json &js)
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

#ifdef _KORALI_USE_MPI
long int korali::Engine::getMPICommPointer()
{
  return (long int)(&__KoraliTeamComm);
}
#endif

knlohmann::json &korali::Engine::operator[](const std::string &key)
{
  return _js[key];
}
knlohmann::json &korali::Engine::operator[](const unsigned long int &key) { return _js[key]; }
pybind11::object korali::Engine::getItem(pybind11::object key) { return _js.getItem(key); }
void korali::Engine::setItem(pybind11::object key, pybind11::object val) { _js.setItem(key, val); }

PYBIND11_MODULE(libkorali, m)
{

#ifdef _KORALI_USE_MPI
   m.def("getMPICommPointer", &korali::Engine::getMPICommPointer, pybind11::return_value_policy::reference);
#endif

  pybind11::class_<korali::Engine>(m, "Engine")
    .def(pybind11::init<>())
    .def("run", [](korali::Engine &k, korali::Experiment &e) {
      korali::isPythonActive = true;
      k.run(e);
    })
    .def("run", [](korali::Engine &k, std::vector<korali::Experiment> &e) {
      korali::isPythonActive = true;
      k.run(e);
    })
    .def("resume", [](korali::Engine &k, korali::Experiment &e) {
      korali::isPythonActive = true;
      k.resume(e);
    })
    .def("resume", [](korali::Engine &k, std::vector<korali::Experiment> &e) {
      korali::isPythonActive = true;
      k.resume(e);
    })
    .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::Engine::getItem), pybind11::return_value_policy::reference)
    .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::Engine::setItem), pybind11::return_value_policy::reference);

  pybind11::class_<korali::KoraliJson>(m, "koraliJson")
    .def("get", &korali::KoraliJson::get)
    .def("set", &korali::KoraliJson::set)
    .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::KoraliJson::getItem), pybind11::return_value_policy::reference)
    .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::KoraliJson::setItem), pybind11::return_value_policy::reference);

  pybind11::class_<korali::Sample>(m, "Sample")
    .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::Sample::getItem), pybind11::return_value_policy::reference)
    .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::Sample::setItem), pybind11::return_value_policy::reference)
    .def("update", &korali::Sample::update);

  pybind11::class_<korali::Experiment>(m, "Experiment")
    .def(pybind11::init<>())
    .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::Experiment::getItem), pybind11::return_value_policy::reference)
    .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::Experiment::setItem), pybind11::return_value_policy::reference)
    .def("loadState", &korali::Experiment::loadState)
    .def("test", &korali::Experiment::test);
}
