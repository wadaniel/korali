#ifndef _KORALI_CIRCULAR_BUFFER_HPP_
#define _KORALI_CIRCULAR_BUFFER_HPP_

/** \file
* @brief Implements a circular buffer with automatic overwrite on full
*        by Sergio Martin (2020), partially based on the implementation
*        by Jose Herrera https://gist.github.com/xstherrera1987/3196485
******************************************************************************/

#include <memory>

/**
* \namespace korali
* @brief The Korali namespace includes all Korali-specific functions, variables, and modules.
*/
namespace korali
{
/**
* \class cBuffer
* @brief This class defines a circular buffer with overwrite policy on add
*/
template <typename T>
class cBuffer
{
  private:
  /**
  * @brief Size of buffer container
  */
  size_t _max_size;

  /**
  * @brief Number of elements already added
  */
  size_t _size;

  /**
  * @brief Container for data
  */
  std::unique_ptr<T[]> _data;

  /**
   * @brief Position of the start of the buffer
   */
  size_t _start;

  /**
   * @brief Position of the end of the buffer
   */
  size_t _end;

  public:
  /**
   * @brief Default constructor
   */
  cBuffer()
  {
    _max_size = 0;
    _size = 0;
    _start = 0;
    _end = 0;
  };

  /**
   * @brief Constructor with a specific size
   * @param size The buffer size
   */
  cBuffer(size_t size)
  {
    resize(size);
  };

  /**
  * @brief Returns the current number of elements in the buffer
  * @return The number of elements
  */
  size_t size() { return _size; };

  /**
  * @brief Returns the current number of elements in the buffer
  * @param size The buffer size
  */
  void resize(size_t size)
  {
    _data = std::make_unique<T[]>(size);

    _size = 0;
    _max_size = size;
    _start = 0;
    _end = 0;
  }

  /**
  * @brief Adds an element to the buffer
  * @param v The element to add
  */
  void add(const T &v)
  {
    // Storing value
    _data[_end] = v;

    // Increasing size until we reach the max size
    if (_size < _max_size) _size++;

    // Increasing end pointer, and continuing from beginning if exceeding size
    _end++;
    if (_end == _max_size) _end = 0;

    // If end pointer met start pointer, then push it one position to replace oldest entry
    if (_end == _start) _start++;

    // If start pointer reached the end, send it back to the beginning
    if (_start == _max_size) _start = 0;
  }

  /**
  * @brief Accesses an element at the required position
  * @param pos The access position
  * @return The element corresponding to the position
  */
  T &operator[](size_t pos)
  {
    return _data[(_start + pos) % _max_size];
  }
};

} // namespace korali

#endif
