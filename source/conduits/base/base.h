#ifndef _KORALI_BASECONDUIT_H_
#define _KORALI_BASECONDUIT_H_

#include <stdlib.h>
#include "json/json.hpp"

namespace Korali::Conduit
{

class Base {
 public:

 virtual void run() = 0;
 virtual void evaluateSample(double* sampleArray, size_t sampleId) = 0;
 virtual void checkProgress() = 0;
 virtual bool isRoot() = 0;

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
};

class Conduit;

} // namespace Korali

#endif // _KORALI_BASECONDUIT_H_
