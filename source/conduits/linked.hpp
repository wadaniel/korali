#ifndef _KORALI_CONDUIT_LINKED_HPP_
#define _KORALI_CONDUIT_LINKED_HPP_

#include "conduits/base.hpp"
#include <queue>
#include <vector>
#include <map>

namespace Korali { namespace Conduit {

class Linked : public Base
{
 public:
 int _rankId;
 int _rankCount;

 // Team Management
 int _ranksPerTeam;
 int _teamCount;

 int _teamId;
 int _localRankId;

 bool _useMPI;
 std::vector<size_t> _teamSampleId;
 std::vector<bool> _teamBusy;
 std::vector<double> _teamFitness;

 #ifdef _KORALI_USE_MPI
 MPI_Comm _teamComm;
 std::vector<MPI_Request> _teamRequests;
 #endif

 std::queue<int> _teamQueue;
 std::map< int, std::vector<int> > _teamWorkers;

 bool _continueEvaluations;

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

#endif // _KORALI_CONDUIT_LINKED_HPP_
