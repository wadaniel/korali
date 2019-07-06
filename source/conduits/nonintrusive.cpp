#include "korali.h"
#include <sys/wait.h>

using namespace Korali::Conduit;


/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Nonintrusive::getConfiguration()
{
 _k->_js["Conduit"] = "Nonintrusive";
 _k->_js["Nonintrusive"]["Concurrent Jobs"] = _concurrentJobs;
}

void Nonintrusive::setConfiguration()
{
 _concurrentJobs = consume(_k->_js, { "Nonintrusive", "Concurrent Jobs" }, KORALI_NUMBER, std::to_string(1));
 if (_concurrentJobs < 1)
 {
  fprintf(stderr, "[Korali] Error: You need to define at least 1 concurrent job(s) for non-intrusive models \n");
  exit(-1);
 }
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Nonintrusive::initialize()
{
 _currentSample = 0;

 _pipeDescriptors = (int**) calloc(_concurrentJobs, sizeof(int*));
 for (int i = 0; i < _concurrentJobs; i++) _pipeDescriptors[i] = (int*) calloc(2, sizeof(int));
 for (int i = 0; i < _concurrentJobs; i++) _launcherQueue.push(i);
}

void Nonintrusive::finalize()
{
}

void Nonintrusive::evaluateSample(double* sampleArray, size_t sampleId)
{
 Korali::ModelData data;

 _k->_problem->packVariables(&sampleArray[_k->N*sampleId], data);

 data._hashId = _currentSample++;

 while (_launcherQueue.empty()) checkProgress();

 int launcherId = _launcherQueue.front(); _launcherQueue.pop();

 // Opening Inter-process communicator pipes
 if (pipe(_pipeDescriptors[launcherId]) == -1)
 {
  fprintf(stderr, "[Korali] Error: Unable to create inter-process pipe. \n");
  exit(-1);
 }

 pid_t processId = fork();

 _launcherIdToSamplerIdMap[launcherId] = sampleId;
 _launcherIdToProcessIdMap[launcherId] = processId;
 _processIdMapToLauncherIdMap[processId] = launcherId;

 if (processId == 0)
 {
  _k->_model(data);
  double fitness = _k->_problem->evaluateFitness(data);
  write(_pipeDescriptors[launcherId][1], &fitness, sizeof(double));
  exit(0);
 }

}

void Nonintrusive::checkProgress()
{
 int status;
 pid_t processId;

 processId = wait(&status);
 if (processId > 0)
 {
  int launcherId = _processIdMapToLauncherIdMap[processId];
  double fitness = 0.0;
  size_t sampleId = _launcherIdToSamplerIdMap[launcherId];
  read(_pipeDescriptors[launcherId][0], &fitness, sizeof(double));
  _k->_solver->processSample(sampleId, fitness);
  close(_pipeDescriptors[launcherId][1]); // Closing pipes
  close(_pipeDescriptors[launcherId][0]); // Closing pipes
  _launcherQueue.push(launcherId);
 }

}

bool Nonintrusive::isRoot()
{
 return true;
}
