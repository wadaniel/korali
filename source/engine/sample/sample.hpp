#ifndef __KORALI_SAMPLE_HPP_
#define __KORALI_SAMPLE_HPP_

#include "auxiliar/logger.hpp"
#include "engine/sample/libco/libco.h"
#include "auxiliar/json.hpp"
#include "auxiliar/koraliJson.hpp"
#include <string>

namespace korali
{

enum class SampleState { uninitialized, initialized, running, waiting, finished };

class Sample {

 public:

 Sample* _self;
 SampleState _state;
 cothread_t _sampleThread;

 // JSON-based configuration
 korali::KoraliJson _js;

 Sample()
 {
  _self = this;
  _state = SampleState::uninitialized;
 }

 // Execution Control Functions
 void start();
 void resume();
 void yield();
 void run(std::uint64_t funcPtr) { (*reinterpret_cast<std::function<void(korali::Sample&)>*>(funcPtr))(*this); }

 bool contains(const std::string& key) { return _self->_js.contains(key); }

 nlohmann::json& operator[](const std::string& key) { return _self->_js[key]; }
 nlohmann::json& operator[](const unsigned long int& key) { return _self->_js[key]; }

 pybind11::object getItem(pybind11::object key) { return _self->_js.getItem(key); }
 void setItem(pybind11::object key, pybind11::object val) { _self->_js.setItem(key, val); }

};

}

#endif // __KORALI_SAMPLE_HPP_
