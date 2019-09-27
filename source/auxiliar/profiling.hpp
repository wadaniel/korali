#ifndef _KORALI_PROFILING_HPP_
#define _KORALI_PROFILING_HPP_

#include <chrono>
#include <vector>
#include <map>

namespace korali {

class ProfileInfo
{
 public:

  // Timeline segments
  std::vector<int> _segmentId; // -1 for idle time
  std::vector<std::chrono::time_point<std::chrono::system_clock>> _segmentStart;
  std::vector<std::chrono::time_point<std::chrono::system_clock>> _segmentEnd;

  double _runTime;
  double _workTime;
  double _idleTime;

  ProfileInfo();
  void startWorkSegment(int segmentId = 0);
  void endWorkSegment();
};

extern std::map<std::string, ProfileInfo> _profileInfo;

} // namespace korali


#endif // _KORALI_PROFILING_HPP_
