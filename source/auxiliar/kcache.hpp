#ifndef _KORALI_CACHE_HPP_
#define _KORALI_CACHE_HPP_

/** \file
* @brief Implements an LRU cache that returns a pre-calculated value if it is not
*        too old. Age is determined by an external timer. Mutual exclusion mechanisms
*        have been added for thread-safe access.
*        by Sergio Martin (2020)
******************************************************************************/

#include <map>
#include <functional>
#include <omp.h>

/**
* \namespace korali
* @brief The Korali namespace includes all Korali-specific functions, variables, and modules.
*/
namespace korali
{

template <typename valType, typename timerType>
struct cacheElement_t
{
 valType value;
 timerType time;
};

/**
* \class cBuffer
* @brief This class defines a circular buffer with overwrite policy on add
*/
template <typename keyType, typename valType, typename timerType>
class kCache
{
  private:

  /**
  * @brief Container for cache elements
  */
  std::map<keyType, cacheElement_t<valType, timerType>> _data;

  timerType* _timer;
  timerType _maxAge;
  omp_lock_t _lock;

  public:

  kCache()
  {
   _timer = &_maxAge;
   _maxAge = 0;
   omp_init_lock(&_lock);
  }

  void setMaxAge(const valType& maxAge)
  {
   _maxAge = maxAge;
  }

  void setTimer(timerType* timer)
  {
   _timer = timer;
  }

  void set(const keyType& key, const valType& val)
  {
   omp_set_lock(&_lock);
   _data[key].value = val;
   _data[key].time = *_timer;
   omp_unset_lock(&_lock);
  }

  bool contains(const keyType& key)
  {

   omp_set_lock(&_lock);
   if (_data.count(key) == 0)
   {
    omp_unset_lock(&_lock);
    return false;
   }

   timerType age = *_timer - _data[key].time;
   omp_unset_lock(&_lock);

   if (age >= _maxAge) return false;

   return true;
  }

  valType get(const keyType& key)
  {
   omp_set_lock(&_lock);
   valType val = _data[key].value;
   omp_unset_lock(&_lock);
   return val;
  }

  valType access(const keyType& key, std::function<valType(void)> func)
  {
   valType val;

   if (contains(key))
   {
    val = get(key);
   }
   else
   {
    val = func();
    set(key, val);
   }

   return val;
  }

};

} // namespace korali

#endif
