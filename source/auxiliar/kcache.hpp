#ifndef _KORALI_CACHE_HPP_
#define _KORALI_CACHE_HPP_

/** \file
* @brief Implements an LRU cache that returns a pre-calculated value if it is not
*        too old. Age is determined by an external timer.
*        by Sergio Martin (2020)
******************************************************************************/

#include <map>

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

  public:

  kCache()
  {
   _timer = &_maxAge;
   _maxAge = 0;
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
   _data[key].value = val;
   _data[key].time = *_timer;
  }

  bool contains(const keyType& key)
  {
   if (_data.count(key) == 0) return false;

   timerType age = *_timer - _data[key].time;
   if (age >= _maxAge) return false;

   return true;
  }

  valType get(const keyType& key)
  {
   return _data[key].value;
  }

};

} // namespace korali

#endif
