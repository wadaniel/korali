#ifndef _KORALI_CONDUIT_SINGLE_H_
#define _KORALI_CONDUIT_SINGLE_H_

#include "conduits/base.h"

namespace Korali::Conduit
{

class Single : public Base {
 public:

 void run() override;
 void evaluateSample(double* sampleArray, size_t sampleId) override;
 void checkProgress() override;
 bool isRoot() override;

 // Constructor / Destructor
 Single(nlohmann::json& js);
 ~Single();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_CONDUIT_SINGLE_H_
