#include "korali.hpp"
#include "auxiliar/koraliJson.hpp"

korali::Korali::Korali()
{
 _isFirstRun = false;
 _mainThread = co_active();
}

void korali::Korali::run()
{
 _startTime = std::chrono::high_resolution_clock::now();

 // Setting output file to stdout, by default.
 korali::setConsoleOutputFile(stdout);
 korali::setVerbosityLevel("Minimal");

 _engineCount = _engineVector.size();

 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Conduit']['Type']")) _js["Conduit"]["Type"] = "Simple";
 if (! korali::JsonInterface::isDefined(_js.getJson(), "['Dry Run']")) _js["Dry Run"] = false;

 for (size_t i = 0; i < _engineCount; i++)
 {
  _engineVector[i]->_engineId = i;
  std::string fileName = "./" + _engineVector[i]->_resultPath + "/log.txt";
  if (_engineCount > 1)  _engineVector[i]->_logFile = fopen(fileName.c_str(), "a");
  if (_engineCount == 1) _engineVector[i]->_logFile = stdout;

  _currentEngine = _engineVector[i];
  if (_engineCount > 1) korali::logInfo("Minimal", "Starting Experiment %lu...\n", i);
  _engineVector[i]->initialize();
 }

 if (_isFirstRun == false)
 {
  _conduit = dynamic_cast<korali::Conduit*>(korali::Module::getModule(_js["Conduit"]));
  _isFirstRun = true;
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
  if (korali::JsonInterface::isDefined(_js.getJson(), "['Conduit']")) korali::JsonInterface::eraseValue(_js.getJson(), "['Conduit']");
  if (korali::JsonInterface::isEmpty(_js.getJson()) == false) korali::logError("Unrecognized settings for the Korali Engine: \n%s\n", _js.getJson().dump(2).c_str());

  // Setting start time.
  _startTime = std::chrono::high_resolution_clock::now();

  while(true)
  {
   bool executed = false;

   for (size_t i = 0; i < _engineCount; i++) if (_engineVector[i]->_isFinished == false)
   {
    korali::setVerbosityLevel(_engineVector[i]->_verbosity);
    korali::setConsoleOutputFile(_engineVector[i]->_logFile);
    _currentEngine = _engineVector[i];
    co_switch(_engineVector[i]->_thread);
    executed = true;

    korali::setConsoleOutputFile(stdout);
    if (_engineCount > 1) if (_engineVector[i]->_isFinished == true) korali::logInfo("Minimal", "Experiment %lu has finished.\n", i);
   }

   if (executed == false) break;
  }

  if (_engineCount > 1) korali::logInfo("Minimal", "All jobs have finished correctly.\n");
  if (_engineCount > 1) korali::logInfo("Normal", "Elapsed Time: %.3fs\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-_startTime).count());

  __profiler["Experiment Count"] = _engineCount;
  __profiler["Elapsed Time"] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-_startTime).count();
  std::string fileName = "./profiling.json";
  korali::JsonInterface::saveJsonToFile(fileName.c_str(), __profiler);
 }

 _conduit->finalize();
}

void korali::Korali::run(korali::Engine& engine)
{
 _engineVector.clear();
 _engineVector.push_back(&engine);
 run();
}

void korali::Korali::run(std::vector<korali::Engine>& engines)
{
 _engineVector.clear();
 for (size_t i = 0; i < engines.size(); i++) _engineVector.push_back(&engines[i]);
 run();
}


#ifdef _KORALI_USE_MPI
long int korali::Korali::getMPICommPointer() { return (long int)(&__KoraliTeamComm); }
#endif

nlohmann::json& korali::Korali::operator[](const std::string& key) { return _js[key]; }
nlohmann::json& korali::Korali::operator[](const unsigned long int& key) { return _js[key]; }
pybind11::object korali::Korali::getItem(pybind11::object key) { return _js.getItem(key); }
void korali::Korali::setItem(pybind11::object key, pybind11::object val) { _js.setItem(key, val); }

PYBIND11_MODULE(libkorali, m)
{
 pybind11::class_<korali::Korali>(m, "Korali")
  .def(pybind11::init<>())
  .def("run", pybind11::overload_cast<korali::Engine&>(&korali::Korali::run))
  .def("run", pybind11::overload_cast<std::vector<korali::Engine>&>(&korali::Korali::run))
  .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::Korali::getItem), pybind11::return_value_policy::reference)
  .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::Korali::setItem), pybind11::return_value_policy::reference);

 pybind11::class_<korali::KoraliJson>(m, "koraliJson")
  .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::KoraliJson::getItem), pybind11::return_value_policy::reference)
  .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::KoraliJson::setItem), pybind11::return_value_policy::reference);

 pybind11::class_<korali::Sample>(m, "Sample")
  .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::Sample::getItem), pybind11::return_value_policy::reference)
  .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::Sample::setItem), pybind11::return_value_policy::reference);

 pybind11::class_<korali::Engine>(m, "Engine")
   .def(pybind11::init<>())
    #ifdef _KORALI_USE_MPI
   .def("getMPIComm", &korali::Korali::getMPICommPointer)
    #endif
   .def("__getitem__", pybind11::overload_cast<pybind11::object>(&korali::Engine::getItem), pybind11::return_value_policy::reference)
   .def("__setitem__", pybind11::overload_cast<pybind11::object, pybind11::object>(&korali::Engine::setItem), pybind11::return_value_policy::reference)
   .def("loadConfig",     &korali::Engine::loadConfig, pybind11::return_value_policy::reference);
}

