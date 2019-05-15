#ifndef _KORALI_CONDUIT_SEQUENTIAL_H_
#define _KORALI_CONDUIT_SEQUENTIAL_H_

#include "conduits/base/base.h"

namespace Korali::Conduit
{

class Sequential : public Base {
 public:

 void run() override;
 void evaluateSample(double* sampleArray, size_t sampleId) override;
 void checkProgress() override;
 bool isRoot() override;

 // Constructor / Destructor
 Sequential(nlohmann::json& js);
 ~Sequential();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_CONDUIT_SEQUENTIAL_H_
