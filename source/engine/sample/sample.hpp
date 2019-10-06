#ifndef __KORALI_SAMPLE_HPP_
#define __KORALI_SAMPLE_HPP_

#include "auxiliar/logger.hpp"
#include "engine/sample/libco/libco.h"
#include "auxiliar/json.hpp"
#include <string>

namespace korali
{

enum class SampleState { uninitialized, initialized, running, waiting, finished };

class Sample {

 public:

 Sample* _self;
 SampleState _state;
 cothread_t _sampleThread;

 nlohmann::json _data;

 Sample()
 {
  _self = this;
  _state = SampleState::uninitialized;
 }

 nlohmann::json& operator[](std::string key)
 {
  return _self->_data[key];
 }

 // Execution Control Functions
 void start();
 void resume();
 void yield();
};

}

#endif // __KORALI_SAMPLE_HPP_
