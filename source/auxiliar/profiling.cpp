#include "auxiliar/profiling.hpp"
#include "stdio.h"

std::map<std::string, korali::ProfileInfo> korali::_profileInfo;

korali::ProfileInfo::ProfileInfo()
{
 _runTime = 0.0;
 _workTime = 0.0;
 _idleTime = 0.0;

 _segmentId.push_back(-1);
 _segmentStart.push_back(std::chrono::high_resolution_clock::now());
}

void korali::ProfileInfo::startWorkSegment(int segmentId)
{
 size_t currentSegment = _segmentStart.size() - 1;
 auto currentTime = std::chrono::high_resolution_clock::now();
 std::chrono::duration<double> elapsed = currentTime - _segmentStart[currentSegment];
 double segmentTime = elapsed.count();

 _segmentId.push_back(segmentId);
 _segmentStart.push_back(currentTime);

 _runTime += segmentTime;
 if (segmentId == -1) _workTime += segmentTime;
 if (segmentId != -1) _idleTime += segmentTime;
}

void korali::ProfileInfo::endWorkSegment()
{
 startWorkSegment(-1);
}
