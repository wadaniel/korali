#ifndef _KORALI_CONDUIT_SIMPLE_HPP_
#define _KORALI_CONDUIT_SIMPLE_HPP_

#include "conduits/base.hpp"
#include <queue>
#include <vector>
#include <map>

namespace Korali { namespace Conduit {

class Simple : public Base
{
 public:

 void initialize() override;
 void finalize() override;

 void evaluateSample(double* sampleArray, size_t sampleId) override;
 void checkProgress() override;
 bool isRoot() override;

 // Serialization Methods
 void getConfiguration() override;
 void setConfiguration() override;
};

} } // namespace Korali::Conduit

#endif // _KORALI_CONDUIT_SIMPLE_HPP_
