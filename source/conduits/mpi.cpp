#include "korali.hpp"

using namespace Korali::Conduit;

#ifdef _KORALI_USE_MPI

#define MPI_TAG_FITNESS 1
#define MPI_TAG_SAMPLE 2
#define MPI_TAG_ID 3

#endif

Korali::Conduit::MPI::MPI()
{
 #ifdef _KORALI_USE_MPI
 _rankCount = 1;
 _rankId = 0;

 int isInitialized;
 MPI_Initialized(&isInitialized);
 if (isInitialized == false)  MPI_Init(nullptr, nullptr);

 MPI_Comm_size(MPI_COMM_WORLD, &_rankCount);
 MPI_Comm_rank(MPI_COMM_WORLD, &_rankId);
 #endif
}

void Korali::Conduit::MPI::initialize()
{
 #ifndef _KORALI_USE_MPI
  koraliError("Running an MPI-based Korali application, but Korali was installed without support for MPI.\n");
 #endif

 #ifdef _KORALI_USE_MPI
 _continueEvaluations = true;

 if (_rankCount == 1) koraliError("Korali MPI applications require at least 2 MPI ranks to run.\n");

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

 _teamSampleId.resize(_teamCount);
 _teamFitness.resize(_teamCount);
 _teamRequests.resize(_teamCount);
 _teamBusy.resize(_teamCount);

 MPI_Comm_split(MPI_COMM_WORLD, _teamId, _rankId, &_teamComm);

 int mpiSize = -1;
 MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

 if(isRoot()) if (_rankCount < _ranksPerTeam + 1)
 koraliError("You are running Korali with %d ranks. However, you need at least %d ranks to have at least one worker team. \n", _rankCount, _ranksPerTeam + 1 );

 MPI_Barrier(MPI_COMM_WORLD);

 if (!isRoot()) workerThread();
 #endif
}


/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/


void Korali::Conduit::MPI::getConfiguration()
{
 #ifdef _KORALI_USE_MPI
 _k->_js["Conduit"]["Type"] = "MPI";
 _k->_js["Conduit"]["Ranks Per Team"] = _ranksPerTeam;
 #endif
}

void Korali::Conduit::MPI::setConfiguration()
{
 #ifdef _KORALI_USE_MPI
 _ranksPerTeam = consume(_k->_js, { "Conduit", "Ranks Per Team" }, KORALI_NUMBER, "1");
 #endif
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Conduit::MPI::finalize()
{
 #ifdef _KORALI_USE_MPI
 if (isRoot())
	{
	 int stopFlag = -1;
		for (int i = 0; i < _teamCount; i++)
			for (int j = 0; j < _ranksPerTeam; j++)
			MPI_Send(&stopFlag, 1, MPI_INT, _teamWorkers[i][j], MPI_TAG_ID, MPI_COMM_WORLD);
 }
	MPI_Barrier(MPI_COMM_WORLD);
 #endif
}

void Korali::Conduit::MPI::workerThread()
{
 #ifdef _KORALI_USE_MPI
 if (_teamId == -1) return;

 int sampleId = 0;
 while (sampleId != -1)
 {
		MPI_Recv(&sampleId, 1, MPI_INT, getRootRank(), MPI_TAG_ID, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (sampleId != -1)
		{
		 double sample[_k->N];
		 MPI_Recv(&sample, _k->N, MPI_DOUBLE, getRootRank(), MPI_TAG_SAMPLE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		 bool isLeader = (_localRankId == 0);

		 Korali::Model data;
		 data._comm = _teamComm;
		 data._sampleId = sampleId;

		 _k->_problem->packVariables(sample, data);

		 _k->_model(data);

		 if (isLeader)
		 {
			 double fitness = _k->_problem->evaluateFitness(data);
			 MPI_Send(&fitness, 1, MPI_DOUBLE, getRootRank(), MPI_TAG_FITNESS, MPI_COMM_WORLD);
		 }

		 MPI_Barrier(_teamComm);
		}
	}
 #endif
}

void Korali::Conduit::MPI::evaluateSample(double* sampleArray, size_t sampleId)
{
 _k->functionEvaluationCount++;

 #ifdef _KORALI_USE_MPI
 while (_teamQueue.empty()) checkProgress();

 int teamId = _teamQueue.front(); _teamQueue.pop();
 _teamSampleId[teamId] = sampleId;

 MPI_Irecv(&_teamFitness[teamId], 1, MPI_DOUBLE, _teamWorkers[teamId][0], MPI_TAG_FITNESS, MPI_COMM_WORLD, &_teamRequests[teamId]);
 _teamBusy[teamId] = true;

 for (int i = 0; i < _ranksPerTeam; i++)
 {
  int workerId = _teamWorkers[teamId][i];
  int continueFlag = 1;
  MPI_Send(&sampleId, 1, MPI_INT, workerId, MPI_TAG_ID, MPI_COMM_WORLD);
  MPI_Send(sampleArray, _k->N, MPI_DOUBLE, workerId, MPI_TAG_SAMPLE, MPI_COMM_WORLD);
 }
 #endif
}

void Korali::Conduit::MPI::checkProgress()
{
 #ifdef _KORALI_USE_MPI
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
 #endif
}

int Korali::Conduit::MPI::getRootRank()
{
 #ifdef _KORALI_USE_MPI
 return _rankCount-1;
 #endif

 return 0;
}

bool Korali::Conduit::MPI::isRoot()
{
 #ifdef _KORALI_USE_MPI
 return _rankId == getRootRank();
 #endif

 return true;
}

void Korali::Conduit::MPI::abort()
{
 #ifdef _KORALI_USE_MPI
 MPI_Abort(MPI_COMM_WORLD, -1);
 #endif
}
