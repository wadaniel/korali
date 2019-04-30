#ifdef _KORALI_USE_MULTITHREAD

#ifndef _KORALI_MULTITHREAD_H_
#define _KORALI_MULTITHREAD_H_

#include "conduits/base/base.h"
#include "pthread.h"
#include <queue>

namespace Korali::Conduit
{

class Lock
{
 pthread_mutex_t _lock;
 public:
 Lock() { pthread_mutex_init(&_lock, 0); }
 ~Lock() { pthread_mutex_destroy(&_lock); }
 void lock() { pthread_mutex_lock(&_lock); }
 void unlock() { pthread_mutex_unlock(&_lock); }
 bool trylock() { return pthread_mutex_trylock(&_lock) == 0; }
};

class Multithread : public Base {
 public:

 Lock _sampleLock;
 size_t _workerCount;
 pthread_t* _workers;
 std::queue<size_t> _sampleQueue;
 bool _continueEvaluations;
 double* _sampleArrayPointer;

 static void workerThread();
 void run() override;
 void evaluateSample(double* sampleArray, size_t sampleId) override;
 void checkProgress() override;
 bool isRoot() override;

 // Constructor / Destructor
 Multithread(nlohmann::json& js);
 ~Multithread();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_MULTITHREAD_H_

#endif // _KORALI_USE_MULTITHREAD
