#ifndef _KORALI_CIRCULAR_BUFFER_HPP_
#define _KORALI_CIRCULAR_BUFFER_HPP_

/** \file
* @brief Implements a circular buffer with automatic overwrite on full
*        by Sergio Martin (2020), partially based on the implementation
*        by Jose Herrera https://gist.github.com/xstherrera1987/3196485
******************************************************************************/

#include <memory>

namespace korali
{
template <typename T>
class cBuffer
{
  size_t _max_size;
  size_t _size;
  std::unique_ptr<T[]> _data;
  size_t _start;
  size_t _end;

  public:
  cBuffer()
  {
    _max_size = 0;
    _size = 0;
    _start = 0;
    _end = 0;
  };

  cBuffer(size_t size)
  {
    resize(size);
  };

  size_t size() { return _size; };

  void resize(size_t size)
  {
    _data = std::make_unique<T[]>(size);

    _size = 0;
    _max_size = size;
    _start = 0;
    _end = 0;
  }

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

  T &operator[](size_t pos)
  {
    return _data[(_start + pos) % _max_size];
  }
};

} // namespace korali

#endif
