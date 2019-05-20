#ifdef _KORALI_USE_MULTITHREAD

#include "korali.h"
#include <unistd.h>

Korali::Conduit::Multithread* _pt;

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Multithread::Multithread(nlohmann::json& js) : Base::Base(js)
{
 setConfiguration(js);

 _pt = this;

 if (_workerCount < 1) { fprintf(stderr, "[Korali] Error: Running Korali's Multithread Conduit with less than 1 worker is not allowed.\n"); exit(-1); }

 _workers = (pthread_t*) calloc (sizeof(pthread_t), _workerCount);

 _continueEvaluations = true;
}

Multithread::~Multithread()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Multithread::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Type"]    = "Multithread";
 js["Workers"] = _workerCount;

 return js;
}

void Multithread::setConfiguration(nlohmann::json& js)
{
 this->Base::setConfiguration(js);

 size_t coreCount = sysconf(_SC_NPROCESSORS_ONLN);

 _workerCount = consume(js, { "Workers" }, KORALI_NUMBER, std::to_string(coreCount));
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Multithread::run()
{
 #ifdef _KORALI_USE_PYTHON
  printf("[Korali] Warning: Python's global lock prevents concurrent execution. Performance will be degraded.\n");
 #endif

 for(size_t i = 0; i < _workerCount; i++)
 {
 	int ret = pthread_create(&_workers[i], 0, (void*(*)(void*)) workerThread, 0);
  if (ret != 0) { printf("[Korali] SMP Conduit: Error creating worker %lu.\n", i); exit(-1); }
 }

 _k->_solver->run();

 _continueEvaluations = false;
 for(size_t i = 0; i < _workerCount; i++) pthread_join(_workers[i], 0);
}

void Multithread::workerThread()
{
 while(_pt->_continueEvaluations)
 {
  bool foundSample = false;
  size_t sampleId = 0;

  if (_pt->_sampleLock.trylock())
  {
   if (!_pt->_sampleQueue.empty())
   {
    sampleId = _pt->_sampleQueue.front();
    _pt->_sampleQueue.pop();
    foundSample = true;
   }
   _pt->_sampleLock.unlock();
  }

  if (foundSample)
  {
   double candidateFitness = _k->_problem->evaluateFitness(&_pt->_sampleArrayPointer[_k->_problem->N*sampleId], true);
   _pt->_sampleLock.lock();
    _k->_solver->processSample(sampleId, candidateFitness);
   _pt->_sampleLock.unlock();
  }
 }
}

void Multithread::evaluateSample(double* sampleArray, size_t sampleId)
{
 _sampleArrayPointer = sampleArray;
 _sampleLock.lock();
 _sampleQueue.push(sampleId);
 _sampleLock.unlock();
}

void Multithread::checkProgress()
{

}

bool Multithread::isRoot()
{
 return true;
}

#endif //_KORALI_USE_MULTITHREAD
