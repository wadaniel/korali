#ifndef _KORALI_CONDUIT_BASE_HPP_
#define _KORALI_CONDUIT_BASE_HPP_

namespace Korali { namespace Conduit {

class Base {
 public:

 virtual void initialize() = 0;
 virtual void finalize() = 0;

 virtual void evaluateSample(double* sampleArray, size_t sampleId) = 0;
 virtual void checkProgress() = 0;
 virtual bool isRoot() = 0;
 virtual void abort() = 0;

 size_t _currentSample;

 // Serialization Methods
 virtual void getConfiguration(nlohmann::json& js) = 0;
 virtual void setConfiguration(nlohmann::json& js) = 0;
};

class Conduit;

} } // namespace Korali::Conduit


#endif // _KORALI_CONDUIT_BASE_HPP_

