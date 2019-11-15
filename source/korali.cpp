#include "korali.hpp"
#include "auxiliar/koraliJson.hpp"

korali::Engine::Engine()
{
 _isFirstRun = true;
 _mainThread = co_active();
}

void korali::Engine::run()
{
 _startTime = std::chrono::high_resolution_clock::now();

 // Setting output file to stdout, by default.
 korali::setConsoleOutputFile(stdout);
 korali::setVerbosityLevel("Minimal");

 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Profiling']['Detail']")) _js["Profiling"]["Detail"] = "None";
 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Profiling']['Path']")) _js["Profiling"]["Path"] = "./profiling.json";
 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Conduit']['Type']")) _js["Conduit"]["Type"] = "Sequential";
 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Dry Run']")) _js["Dry Run"] = false;

 std::string profilingPath = _js["Profiling"]["Path"];
 std::string profilingDetail = _js["Profiling"]["Detail"];

 for (size_t i = 0; i < _experimentVector.size(); i++)
 {
  _experimentVector[i]->_experimentId = i;
  std::string fileName = "./" + _experimentVector[i]->_resultsPath + "/log.txt";
  if (_experimentVector.size() > 1)  _experimentVector[i]->_logFile = fopen(fileName.c_str(), "a");
  if (_experimentVector.size() == 1) _experimentVector[i]->_logFile = stdout;

  _currentExperiment = _experimentVector[i];
  if (_experimentVector.size() > 1) korali::logInfo("Minimal", "Starting Experiment %lu...\n", i);
  _experimentVector[i]->initialize();
 }

 if (_isFirstRun == true)
 {
  _conduit = dynamic_cast<korali::Conduit*>(korali::Module::getModule(_js["Conduit"]));
  _isFirstRun = false;
 }

 _conduit->initialize();

 // If this is a worker process (not root), there's nothing else to do
 if (_conduit->isRoot())
 {
  // If this is a dry run and configuration succeeded, print sucess and return
  bool isDryRun = _js["Dry Run"];
  if (isDryRun)
  {
   korali::logInfo("Minimal",  "--------------------------------------------------------------------\n");
   korali::logInfo("Minimal",  "Dry Run Successful.\n");
   korali::logInfo("Minimal",  "--------------------------------------------------------------------\n");
   return;
  }

  if (korali::JsonInterface::isDefined(_js.getJson(), "['Dry Run']")) korali::JsonInterface::eraseValue(_js.getJson(), "['Dry Run']");
  if (korali::JsonInterface::isDefined(_js.getJson(), "['Conduit']['Type']")) korali::JsonInterface::eraseValue(_js.getJson(), "['Conduit']['Type']");
  if (korali::JsonInterface::isDefined(_js.getJson(), "['Profiling']['Detail']")) korali::JsonInterface::eraseValue(_js.getJson(), "['Profiling']['Detail']");
  if (korali::JsonInterface::isDefined(_js.getJson(), "['Profiling']['Path']")) korali::JsonInterface::eraseValue(_js.getJson(), "['Profiling']['Path']");
  if (korali::JsonInterface::isEmpty(_js.getJson()) == false) korali::logError("Unrecognized settings for Korali's Engine: \n%s\n", _js.getJson().dump(2).c_str());

  // Setting start time.
  _startTime = std::chrono::high_resolution_clock::now();

  while(true)
  {
   bool executed = false;

   for (size_t i = 0; i < _experimentVector.size(); i++) if (_experimentVector[i]->_isFinished == false)
   {
    korali::setVerbosityLevel(_experimentVector[i]->_consoleVerbosity);
    korali::setConsoleOutputFile(_experimentVector[i]->_logFile);
    _currentExperiment = _experimentVector[i];
    co_switch(_experimentVector[i]->_thread);
    executed = true;

    korali::setConsoleOutputFile(stdout);
    if (_experimentVector.size() > 1) if (_experimentVector[i]->_isFinished == true) korali::logInfo("Minimal", "Experiment %lu has finished.\n", i);
   }

   if (executed == false) break;
  }

  if (_experimentVector.size() > 1) korali::logInfo("Minimal", "All jobs have finished correctly.\n");
  if (_experimentVector.size() > 1) korali::logInfo("Normal", "Elapsed Time: %.3fs\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-_startTime).count());

  if (profilingDetail == "Full")
  {
   __profiler["Experiment Count"] = _experimentVector.size();
   __profiler["Elapsed Time"] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-_startTime).count();
   korali::JsonInterface::saveJsonToFile(profilingPath.c_str(), __profiler);
  }
 }

 _conduit->finalize();
}

void korali::Engine::run(korali::Experiment& experiment)
{
 _experimentVector.clear();
 _experimentVector.push_back(&experiment);
 run();
}

void korali::Engine::run(std::vector<korali::Experiment>& experiments)
{
 _experimentVector.clear();
 for (size_t i = 0; i < experiments.size(); i++) _experimentVector.push_back(experiments[i]._k);
 run();
}


#ifdef _KORALI_USE_MPI
long int korali::Engine::getMPICommPointer() { return (long int)(&__KoraliTeamComm); }
#endif

nlohmann::json& korali::Engine::operator[](const std::string& key) { return _js[key]; }
nlohmann::json& korali::Engine::operator[](const unsigned long int& key) { return _js[key]; }
pybind11::object korali::Engine::getItem(pybind11::object key) { return _js.getItem(key); }
void korali::Engine::setItem(pybind11::object key, pybind11::object val) { _js.setItem(key, val); }

PYBIND11_MODULE(libkorali, m)
{
 pybind11::class_<korali::Engine>(m, "Engine")
  .def(pybind11::init<>())
   #ifdef _KORALI_USE_MPI
   .def("getMPIComm", &korali::Engine::getMPICommPointer)
   #endif
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

