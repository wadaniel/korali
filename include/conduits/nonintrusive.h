#ifndef _KORALI_CONDUIT_NONINTRUSIVE_H_
#define _KORALI_CONDUIT_NONINTRUSIVE_H_

#include "conduits/base.h"

namespace Korali::Conduit
{

class Nonintrusive : public Base {
 public:

 std::string _command;

 void run() override;
 void evaluateSample(double* sampleArray, size_t sampleId) override;
 void checkProgress() override;
 bool isRoot() override;

 // Constructor / Destructor
 Nonintrusive(nlohmann::json& js);
 ~Nonintrusive();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_CONDUIT_NONINTRUSIVE_H_
