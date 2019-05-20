#ifdef _KORALI_USE_UPCXX

#include "korali.h"

using namespace Korali::Conduit;

Korali::Conduit::UPCXX* _ux;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

UPCXX::UPCXX(nlohmann::json& js) : Base::Base(js)
{
 setConfiguration(js);

 _continueEvaluations = true;

 _ux = this;

 upcxx::init();
 _rankId = upcxx::rank_me();
 _rankCount = upcxx::rank_n();

 int _teamCount = (_rankCount-_rankOffset) / _ranksPerTeam;

 if (_rankId == 0)
 {
  printf("Rank Count: %d\n", _rankCount);
  printf("Ranks per Team: %d\n", _ranksPerTeam);
  printf("Team Count: %d\n", _teamCount);
 }

 _teamId = -1;
 _localRankId = -1;

 int currentRank = 1;
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

 MPI_Comm_split(MPI_COMM_WORLD, _teamId, _rankId, &teamComm);

 int mpiSize = -1;
 MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
 printf("Rank %d - Team: %d - LocalId: %d - MPI_Ranks: %d\n", _rankId, _teamId, _localRankId, mpiSize);

 if (_rankCount < 2)
 {
  fprintf(stderr, "[Korali] Error: Running Korali's UPCxx Conduit with less than 2 ranks is not allowed.\n");
  exit(-1);
 }


 samplePtr = upcxx::new_array<double>(_k->_problem->N);
 upcxx::dist_object<upcxx::global_ptr<double>> p(samplePtr);
 upcxx::barrier();

 if (_rankId == 0)
 {
  samplePtrGlobal = (upcxx::global_ptr<double>*) calloc (_rankCount, sizeof(upcxx::global_ptr<double>));
  for (int i = 0; i < _rankCount; i++) samplePtrGlobal[i] = p.fetch(i).wait();
 }

 upcxx::barrier();
}

UPCXX::~UPCXX()
{
 upcxx::finalize();
}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json UPCXX::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Type"] = "UPC++";
 js["Rank Offset"] = _rankOffset;
 js["Ranks Per Team"] = _ranksPerTeam;

 return js;
}

void UPCXX::setConfiguration(nlohmann::json& js)
{
 _ranksPerTeam = consume(js, { "Ranks Per Team" }, KORALI_NUMBER, std::to_string(1));
 _rankOffset = consume(js, { "Rank Offset" }, KORALI_NUMBER, std::to_string(1));
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void UPCXX::run()
{
 // Supervisor
 if (_rankId == 0)
 {
  _k->_solver->run();
  for (int i = 1; i < _rankCount; i++) upcxx::rpc_ff(i, [](){_ux->_continueEvaluations = false;});
 }
 // Workers
 else
 {
   if (_teamId != -1)
   while(_continueEvaluations) upcxx::progress();
 }

 upcxx::barrier();
}

void UPCXX::evaluateSample(double* sampleArray, size_t sampleId)
{
 while(_teamQueue.empty()) upcxx::progress();
 int teamId = _teamQueue.front(); _teamQueue.pop();

 upcxx::future<> puts = upcxx::make_future();

 for (int i = 0; i < _ranksPerTeam; i++)
 {
  int workerId = _teamWorkers[teamId][i];
  auto put = upcxx::rput(&sampleArray[sampleId*_k->_problem->N], samplePtrGlobal[workerId], _k->_problem->N);
  puts = upcxx::when_all(puts, put);
 }

 puts.then([teamId, sampleId]()
 {
  upcxx::future<> samplingRequests = upcxx::make_future();

  for (int i = 0; i < _ux->_ranksPerTeam; i++)
  {
   auto request = upcxx::rpc(_ux->_teamWorkers[teamId][i], [](size_t sampleId)
    {
    bool isLeader = (_ux->_localRankId == 0);
    double fitness = _k->_problem->evaluateFitness(_ux->samplePtr.local(), isLeader, MPI_COMM_NULL);
    if (_ux->_localRankId == 0)
    upcxx::rpc_ff(0,  [](size_t sampleId, double fitness)
    {
     _k->_solver->processSample(sampleId, fitness);
    },sampleId, fitness);
   }, sampleId);

  samplingRequests = upcxx::when_all(samplingRequests, request);
 }

 samplingRequests.then([teamId]()
 {
   _ux->_teamQueue.push(teamId);
 });

});
}

void UPCXX::checkProgress()
{
 upcxx::progress();
}

bool UPCXX::isRoot()
{
 return _rankId == 0;
}

#endif // #ifdef _KORALI_USE_UPCXX
