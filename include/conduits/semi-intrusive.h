#ifndef _KORALI_CONDUIT_SEMI_INTRUSIVE_H_
#define _KORALI_CONDUIT_SEMI_INTRUSIVE_H_

#include "conduits/base.h"

namespace Korali { namespace Conduit {

class SemiIntrusive : public Base {
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

#endif // _KORALI_CONDUIT_SEMI_INTRUSIVE_H_
