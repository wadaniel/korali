#ifdef _KORALI_USE_UPCXX

#ifndef _KORALI_UPCXX_H_
#define _KORALI_UPCXX_H_

#include "conduits/base/base.h"
#include <upcxx/upcxx.hpp>
#include <queue>

namespace Korali::Conduit
{

class UPCXX : public Base
{
 public:
 int _rankId;
 int _rankCount;

 upcxx::global_ptr<double> samplePtr; // Individual Pointer for Sample parameters
 upcxx::global_ptr<double>* samplePtrGlobal; // Common Pointer for Sample parameters
 std::queue<int> _workers;
 bool _continueEvaluations;

 void run() override;
 void evaluateSample(double* sampleArray, size_t sampleId) override;
 void checkProgress() override;
 bool isRoot() override;

 // Constructor / Destructor
 UPCXX(nlohmann::json& js);
 ~UPCXX();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_UPCXX_H_

#endif // _KORALI_USE_UPCXX
