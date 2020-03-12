#ifndef __KORALI_SAMPLE_HPP_
#define __KORALI_SAMPLE_HPP_

#include "auxiliar/koraliJson.hpp"
#include "auxiliar/logger.hpp"
#include "external/libco/libco.h"
#include <string>

#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

namespace korali
{

enum class SampleState { uninitialized, initialized, running, waiting, finished };

class Sample {

 public:

 Sample* _self;
 SampleState _state;
 cothread_t _sampleThread;
 bool _isAllocated;

 // JSON-based configuration
 korali::KoraliJson _js;

 Sample()
 {
  _self = this;
  _state = SampleState::uninitialized;
  _js["Sample Id"] = 0;
  _isAllocated = false;
 }

 // Execution Control Functions
 void start();
 void resume();
 void yield();
 void run(std::uint64_t funcPtr) { (*reinterpret_cast<std::function<void(korali::Sample&)>*>(funcPtr))(*this); }

 bool contains(const std::string& key) { return _self->_js.contains(key); }

 knlohmann::json& operator[](const std::string& key) { return _self->_js[key]; }
 knlohmann::json& operator[](const unsigned long int& key) { return _self->_js[key]; }

 pybind11::object getItem(pybind11::object key) { return _self->_js.getItem(key); }
 void setItem(pybind11::object key, pybind11::object val) { _self->_js.setItem(key, val); }

};

}

#endif // __KORALI_SAMPLE_HPP_
