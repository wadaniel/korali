#include "korali.h"
#include <sys/wait.h>

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Nonintrusive::Nonintrusive(nlohmann::json& js) : Base::Base(js)
{
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

 js["Type"] = "Nonintrusive";

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
	for (int i = 0; i < _concurrentJobs; i++)	_launcherQueue.push(i);
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

 pid_t processId = fork();

 _launcherIdToProcessIdMap[launcherId] = processId;
 _processIdMapToLauncherIdMap[processId] = launcherId;

 if (processId == 0)
 {
  _k->_model(data);
  //double fitness = _k->_problem->evaluateFitness(data);
  //_k->_solver->processSample(sampleId, fitness);
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
		printf("Subprocess %d finished.\n", launcherId );
		_launcherQueue.push(launcherId);
	 }

}

bool Nonintrusive::isRoot()
{
 return true;
}
