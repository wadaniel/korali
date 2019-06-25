#ifdef _KORALI_USE_MPI

#include "korali.h"

using namespace Korali::Conduit;

#define MPI_TAG_CONTINUE 0
#define MPI_TAG_FITNESS 1
#define MPI_TAG_SAMPLE 2

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Distributed::Distributed(nlohmann::json& js)
{
 _name = "Distributed";
 setConfiguration(js);

 _currentSample = 0;
 _continueEvaluations = true;

 int isInitialized;
 MPI_Initialized(&isInitialized);
 if (isInitialized == false)  MPI_Init(nullptr, nullptr);

 MPI_Comm_size(MPI_COMM_WORLD, &_rankCount);
 MPI_Comm_rank(MPI_COMM_WORLD, &_rankId);

 _teamCount = (_rankCount-1) / _ranksPerTeam;
 _teamId = -1;
 _localRankId = -1;

 int currentRank = 0;
 for (int i = 0; i < _teamCount; i++)
 {
   _teamQueue.push(i);
   for (int j = 0; j < _ranksPerTeam; j++)
   {
     if (currentRank == _rankId)
     {
       _teamId = i;
       _localRankId = j;
     }
     _teamWorkers[i].push_back(currentRank++);
   }
 }

 _teamSampleId = (size_t*) calloc (_teamCount, sizeof(size_t));
 _teamFitness = (double*) calloc (_teamCount, sizeof(double));
 _teamRequests = (MPI_Request*) calloc (_teamCount, sizeof(MPI_Request));
 _teamBusy = (bool*) calloc (_teamCount, sizeof(bool));
 MPI_Comm_split(MPI_COMM_WORLD, _teamId, _rankId, &_teamComm);

 int mpiSize = -1;
 MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

 if(isRoot()) if (_rankCount < _ranksPerTeam + 1)
 {
  fprintf(stderr, "[Korali] Error: You are running MPI with %d ranks. \n", _rankCount);
  fprintf(stderr, "[Korali] However, you need at least %d ranks to have at least one worker team. \n", _ranksPerTeam + 1);
  exit(-1);
 }

 MPI_Barrier(MPI_COMM_WORLD);
}

Distributed::~Distributed()
{
}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/


void Distributed::getConfiguration(nlohmann::json& js)
{
 js["Conduit"] = _name;
 js["MPI"]["Ranks Per Team"] = _ranksPerTeam;
}

void Distributed::setConfiguration(nlohmann::json& js)
{
 _ranksPerTeam = consume(js, { "MPI", "Ranks Per Team" }, KORALI_NUMBER, std::to_string(1));
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Distributed::run()
{
 if (isRoot())
 {
   _k->_solver->run();

   int continueFlag = 0;
   for (int i = 0; i < _teamCount; i++)
    for (int j = 0; j < _ranksPerTeam; j++)
     MPI_Send(&continueFlag, 1, MPI_INT, _teamWorkers[i][j], MPI_TAG_CONTINUE, MPI_COMM_WORLD);
 }
 else workerThread();

 MPI_Barrier(MPI_COMM_WORLD);
}

void Distributed::workerThread()
{
 if (_teamId == -1) return;

 int continueFlag = 1;
 while (continueFlag == 1)
 {
  MPI_Recv(&continueFlag, 1, MPI_INT, getRootRank(), MPI_TAG_CONTINUE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (continueFlag == 1)
  {
   double sample[_k->N];
   MPI_Recv(&sample, _k->N, MPI_DOUBLE, getRootRank(), MPI_TAG_SAMPLE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   bool isLeader = (_localRankId == 0);

   Korali::ModelData data;
   data._comm = _teamComm;
   data._hashId = _rankId * 500000 + _currentSample++;

   _k->_problem->packVariables(sample, data);

   _k->_model(data);

   if (_localRankId == 0)
   {
    double fitness = _k->_problem->evaluateFitness(data);
    MPI_Send(&fitness, 1, MPI_DOUBLE, getRootRank(), MPI_TAG_FITNESS, MPI_COMM_WORLD);
   }

   MPI_Barrier(_teamComm);
  }
 }
}

void Distributed::evaluateSample(double* sampleArray, size_t sampleId)
{
 while (_teamQueue.empty()) checkProgress();

 int teamId = _teamQueue.front(); _teamQueue.pop();
 _teamSampleId[teamId] = sampleId;

 MPI_Irecv(&_teamFitness[teamId], 1, MPI_DOUBLE, _teamWorkers[teamId][0], MPI_TAG_FITNESS, MPI_COMM_WORLD, &_teamRequests[teamId]);
 _teamBusy[teamId] = true;

 for (int i = 0; i < _ranksPerTeam; i++)
 {
  int workerId = _teamWorkers[teamId][i];
  int continueFlag = 1;
  MPI_Send(&continueFlag, 1, MPI_INT, workerId, MPI_TAG_CONTINUE, MPI_COMM_WORLD);
  MPI_Send(&sampleArray[sampleId*_k->N],_k->N, MPI_DOUBLE, workerId, MPI_TAG_SAMPLE, MPI_COMM_WORLD);
 }
}

void Distributed::checkProgress()
{
 for (int i = 0; i < _teamCount; i++) if (_teamBusy[i] == true)
 {
  int flag;
  MPI_Test(&_teamRequests[i], &flag, MPI_STATUS_IGNORE);
  if (flag)
  {
   _k->_solver->processSample(_teamSampleId[i], _teamFitness[i]);
   _teamBusy[i] = false;
   _teamQueue.push(i);
  }
 }
}

int Distributed::getRootRank()
{
 return _rankCount-1;
}

bool Distributed::isRoot()
{
 return _rankId == getRootRank();
}

#endif // #ifdef _KORALI_USE_MPI
