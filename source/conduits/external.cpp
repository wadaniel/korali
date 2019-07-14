#include "korali.h"
#include <sys/wait.h>

using namespace Korali::Conduit;


/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void External::getConfiguration()
{
 _k->_js["Conduit"]["Type"] = "External";
 _k->_js["Conduit"]["Concurrent Jobs"] = _concurrentJobs;
}

void External::setConfiguration()
{
 _concurrentJobs = consume(_k->_js, { "Conduit", "Concurrent Jobs" }, KORALI_NUMBER, std::to_string(1));
 if (_concurrentJobs < 1)
 {
  fprintf(stderr, "[Korali] Error: You need to define at least 1 concurrent job(s) for external models \n");
  exit(-1);
 }
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void External::initialize()
{
 for (int i = 0; i < _concurrentJobs; i++) _pipeDescriptors.push_back(std::vector<int>(2));
 for (int i = 0; i < _concurrentJobs; i++) _launcherQueue.push(i);
}

void External::finalize()
{
}

void External::evaluateSample(double* sampleArray, size_t sampleId)
{
 Korali::Model data;

 _k->_problem->packVariables(sampleArray, data);

 data._sampleId = sampleId;

 while (_launcherQueue.empty()) checkProgress();

 int launcherId = _launcherQueue.front(); _launcherQueue.pop();

 // Opening Inter-process communicator pipes
 if (pipe(_pipeDescriptors[launcherId].data()) == -1)
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

 _k->functionEvaluationCount++;

}

void External::checkProgress()
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

bool External::isRoot()
{
 return true;
}
