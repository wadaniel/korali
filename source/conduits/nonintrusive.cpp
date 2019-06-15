#include "korali.h"
#include <sys/wait.h>

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Nonintrusive::Nonintrusive(nlohmann::json& js) : Base::Base(js)
{
 _name = "Nonintrusive";
 setConfiguration(js);
}

Nonintrusive::~Nonintrusive()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Nonintrusive::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Type"] = _name;

 return js;
}

void Nonintrusive::setConfiguration(nlohmann::json& js)
{
 this->Base::setConfiguration(js);

 _concurrentJobs = consume(js, { "Concurrent Jobs" }, KORALI_NUMBER, std::to_string(1));
 if (_concurrentJobs < 1)
 {
  fprintf(stderr, "[Korali] Error: You need to define at least 1 concurrent job(s) for non-intrusive models \n");
  exit(-1);
 }
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Nonintrusive::run()
{

 _pipeDescriptors = (int**) calloc(_concurrentJobs, sizeof(int*));
 for (int i = 0; i < _concurrentJobs; i++) _pipeDescriptors[i] = (int*) calloc(2, sizeof(int));

 for (int i = 0; i < _concurrentJobs; i++) _launcherQueue.push(i);

 _k->_solver->run();
}

void Nonintrusive::evaluateSample(double* sampleArray, size_t sampleId)
{
 Korali::ModelData data;

 int curVar = 0;
 for (int i = 0; i < _k->_problem->_computationalVariableCount; i++) data._computationalVariables.push_back(sampleArray[_k->_problem->N*sampleId + curVar++]);
 for (int i = 0; i < _k->_problem->_statisticalVariableCount;   i++) data._statisticalVariables.push_back(  sampleArray[_k->_problem->N*sampleId + curVar++]);

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
