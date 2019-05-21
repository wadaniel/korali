#ifdef _KORALI_USE_MPI

#ifndef _KORALI_CONDUIT_KORALIMPI_H_
#define _KORALI_CONDUIT_KORALIMPI_H_

#include "conduits/base/base.h"
#include <queue>
#include <vector>
#include <map>

namespace Korali::Conduit
{

class KoraliMPI : public Base
{
 public:
 int _rankId;
 int _rankCount;

 // Team Management
 int _rankOffset;
 int _ranksPerTeam;
 int _teamCount;

 int _teamId;
 int _localRankId;
 size_t* _teamSampleId;
 MPI_Comm _teamComm;
 MPI_Request* _teamRequests;
 bool* _teamBusy;
 double* _teamFitness;

 std::queue<int> _teamQueue;
 std::map< int, std::vector<int> > _teamWorkers;

 bool _continueEvaluations;

 void run() override;
 void workerThread();
 void evaluateSample(double* sampleArray, size_t sampleId) override;
 void checkProgress() override;
 bool isRoot() override;
 int getRootRank();

 // Constructor / Destructor
 KoraliMPI(nlohmann::json& js);
 ~KoraliMPI();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_CONDUIT_KORALIMPI_H_

#endif // _KORALI_USE_MPI
