#ifndef _KORALI_CONDUIT_BASE_HPP_
#define _KORALI_CONDUIT_BASE_HPP_

#include "auxiliars/json.hpp"

namespace Korali { namespace Conduit {

class Base : public Korali::Module {

 public:

 virtual void evaluateSample(double* sampleArray, size_t sampleId) = 0;
 virtual void checkProgress() = 0;
 virtual bool isRoot() = 0;
 virtual void abort() = 0;

};

class Conduit;

} } // namespace Korali::Conduit


#endif // _KORALI_CONDUIT_BASE_HPP_

