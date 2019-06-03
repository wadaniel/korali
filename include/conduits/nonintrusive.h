#ifndef _KORALI_CONDUIT_NONINTRUSIVE_H_
#define _KORALI_CONDUIT_NONINTRUSIVE_H_

#include "conduits/base.h"
#include <sys/types.h>
#include <unistd.h>
#include <queue>
#include <vector>
#include <map>

namespace Korali::Conduit
{

class Nonintrusive : public Base {
 public:

 int _concurrentJobs;
 int** _pipeDescriptors;
 std::queue<int> _launcherQueue;

 std::map<int, size_t> _launcherIdToSamplerIdMap;
 std::map<int, pid_t> _launcherIdToProcessIdMap;
 std::map<pid_t, int> _processIdMapToLauncherIdMap;

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
