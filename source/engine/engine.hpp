#ifndef _ENGINE_HPP_
#define _ENGINE_HPP_

#include "experiment/experiment.hpp"
#include "auxiliar/py2json.hpp"
#include "conduit/distributed/distributed.hpp"
#include "conduit/conduit.hpp"
#include <chrono>

namespace korali
{
 class Engine : public korali::Module
 {
  public:

  Engine();

  std::string _profilingPath;
  std::string _profilingDetail;
  double _profilingFrequency;
  std::chrono::time_point<std::chrono::high_resolution_clock> _profilingLastSave;

  // State save/load methods
  void saveProfilingInfo(bool forceSave = false);
  void run(std::vector<korali::Experiment>& experiments);
  void run(korali::Experiment& experiment);
  void run();

  nlohmann::json& operator[](const std::string& key);
  nlohmann::json& operator[](const unsigned long int& key);
  pybind11::object getItem(pybind11::object key);
  void setItem(pybind11::object key, pybind11::object val);

  // JSON-based configuration
  korali::KoraliJson  _js;

  // Communicator Methods
  static long int getMPICommPointer();
 };
}

#endif
