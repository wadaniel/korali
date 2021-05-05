#ifndef _KORALI_CACHE_HPP_
#define _KORALI_CACHE_HPP_

/** \file
* @brief Implements an LRU cache that returns a pre-calculated value if it is not
*        too old. Age is determined by an external timer. Mutual exclusion mechanisms
*        have been added for thread-safe access.
*        by Sergio Martin (2020)
******************************************************************************/

#include <functional>
#include <map>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
* \namespace korali
* @brief The Korali namespace includes all Korali-specific functions, variables, and modules.
*/
namespace korali
{
/**
* @brief Struct that defines an element present in Korali's cache structure
*/
template <typename valType, typename timerType>
struct cacheElement_t
{
  /**
   * @brief Value of the element
   */
  valType value;

  /**
  * @brief Time when the element was last updated
  */
  timerType time;
};

/**
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

  /**
  * @brief Pointer to the external timer
  */
  timerType *_timer;

  /**
   * @brief Maximum age threshold of data before expiration
   */
  timerType _maxAge;

  /**
   * @brief Lock for thread-safe operation
   */
#ifdef _OPENMP
  omp_lock_t _lock;
#endif

  public:
  /**
   * @brief Default constructor
   */
  kCache()
  {
    _timer = &_maxAge;
    _maxAge = 0;
#ifdef _OPENMP
    omp_init_lock(&_lock);
#endif
  }

  /**
   * @brief Re-sets the maximum age threshold
   * @param maxAge the maximum age threshold
   */
  void setMaxAge(const timerType &maxAge)
  {
    _maxAge = maxAge;
  }

  /**
   * @brief Sets the pointer to an external timer
   * @param timer The external timer
   */
  void setTimer(timerType *timer)
  {
    _timer = timer;
  }

  /**
   * @brief Updates the value of a data element in the cache
   * @param key Key of the data element to update
   * @param val Value of the data element to update
   */
  void set(const keyType &key, const valType &val)
  {
#ifdef _OPENMP
    omp_set_lock(&_lock);
#endif
    _data[key].value = val;
    _data[key].time = *_timer;
#ifdef _OPENMP
    omp_unset_lock(&_lock);
#endif
  }

  /**
   * @brief Updates the value of a data element in the cache, forcing a specific time for it
   * @param key Key of the data element to update
   * @param val Value of the data element to update
   * @param time Time assigned to the data element
   */
  void set(const keyType &key, const valType &val, const timerType &time)
  {
#ifdef _OPENMP
    omp_set_lock(&_lock);
#endif
    _data[key].value = val;
    _data[key].time = time;
#ifdef _OPENMP
    omp_unset_lock(&_lock);
#endif
  }

  /**
   * @brief Checks whether a given data element is present in cache
   * @param key Key of the data element to check
   * @return Whether or not the data element is present
   */
  bool contains(const keyType &key)
  {
#ifdef _OPENMP
    omp_set_lock(&_lock);
#endif
    if (_data.count(key) == 0)
    {
#ifdef _OPENMP
      omp_unset_lock(&_lock);
#endif
      return false;
    }

    timerType age = *_timer - _data[key].time;
#ifdef _OPENMP
    omp_unset_lock(&_lock);
#endif

    if (age >= _maxAge) return false;

    return true;
  }

  /**
   * @brief Reads the value of a data element from the cache. The data element should be present, or an error will occur.
   * @param key Key of the data element to access
   * @return The value of the element stored in cache
   */
  valType get(const keyType &key)
  {
#ifdef _OPENMP
    omp_set_lock(&_lock);
#endif
    valType val = _data[key].value;
#ifdef _OPENMP
    omp_unset_lock(&_lock);
#endif
    return val;
  }

  /**
  * @brief Reads the value of a data element from the cache. If the element is not present, it calls the provided function to generate it.
  * @param key Key of the data element to access
  * @param func Function (lambda or regular) that, upon calling, will return the new value for the element.
  * @return The value of the element stored in cache or generated by the function
  */
  valType access(const keyType &key, std::function<valType(void)> func)
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

  /**
  * @brief Returns the stored entry keys as an ordered vector
  * @return A vector containing all ordered keys
  */
  std::vector<keyType> getKeys()
  {
    std::vector<keyType> v;
    for (auto it = _data.begin(); it != _data.end(); ++it) v.push_back(it->first);
    return v;
  }

  /**
   * @brief Returns the stored entry values in the cache
   * @return A vector containing all stored entry values, ordered by key
   */
  std::vector<valType> getValues()
  {
    std::vector<valType> v;
    for (auto it = _data.begin(); it != _data.end(); ++it) v.push_back(it->second.value);
    return v;
  }

  /**
   * @brief Returns the stored entry times in the cache
   * @return A vector containing all stored entry times, ordered by key
   */
  std::vector<timerType> getTimes()
  {
    std::vector<timerType> v;
    for (auto it = _data.begin(); it != _data.end(); ++it) v.push_back(it->second.time);
    return v;
  }
};

} // namespace korali

#endif
