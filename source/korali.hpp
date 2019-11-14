#ifndef _KORALI_HPP_
#define _KORALI_HPP_

#include "experiment/experiment.hpp"
#include "auxiliar/py2json.hpp"
#include "conduit/distributed/distributed.hpp"
#include "conduit/conduit.hpp"

namespace korali
{
 class Engine : public korali::Module
 {
  public:

  Engine();

  bool _isFirstRun;

  // State save/load methods
  void saveProfilingInfo();
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
