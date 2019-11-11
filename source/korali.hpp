#ifndef _KORALI_HPP_
#define _KORALI_HPP_

#include "engine/engine.hpp"
#include "auxiliar/py2json.hpp"
#include "conduit/MPI/MPI.hpp"
#include "conduit/conduit.hpp"

namespace korali
{
 class Korali : public korali::Module
 {
  public:

  Korali();

  bool _isConduitInitialized;

  // Main Thread
  size_t _engineCount;

  // State save/load methods
  void saveProfilingInfo();
  void run(std::vector<korali::Engine>& engines);
  void run(korali::Engine&);

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
