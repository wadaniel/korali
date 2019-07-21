#ifndef _KORALI_CONDUIT_MPI_HPP_
#define _KORALI_CONDUIT_MPI_HPP_

#include "conduits/base.hpp"
#include <queue>
#include <vector>
#include <map>

namespace Korali { namespace Conduit {

class MPI : public Base
{
 public:

 #ifdef _KORALI_USE_MPI
 int _rankId;
 int _rankCount;

 // Team Management
 int _ranksPerTeam;
 int _teamCount;

 int _teamId;
 int _localRankId;

 std::vector<size_t> _teamSampleId;
 std::vector<bool> _teamBusy;
 std::vector<double> _teamFitness;

 MPI_Comm _teamComm;
 std::vector<MPI_Request> _teamRequests;

 std::queue<int> _teamQueue;
 std::map< int, std::vector<int> > _teamWorkers;

 bool _continueEvaluations;
 #endif

 void initialize() override;
 void finalize() override;

 void workerThread();
 void evaluateSample(double* sampleArray, size_t sampleId) override;
 void checkProgress() override;
 bool isRoot() override;
 int getRootRank();

 // Serialization Methods
 void getConfiguration() override;
 void setConfiguration() override;
};

} } // namespace Korali::Conduit

#endif // _KORALI_CONDUIT_MPI_HPP_
