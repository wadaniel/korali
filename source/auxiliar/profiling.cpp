#include "auxiliar/profiling.hpp"
#include "stdio.h"

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
 _segmentId.push_back(segmentId);
 _segmentStart.push_back(std::chrono::high_resolution_clock::now());

 size_t currentSegment = _segmentStart.size() - 1;
 size_t previousSegment = _segmentStart.size() - 2;
 std::chrono::duration<double> elapsed = _segmentStart[currentSegment] - _segmentStart[previousSegment];
 double segmentTime = elapsed.count();

 _runTime += segmentTime;
 if (segmentId == -1) _workTime += segmentTime;
 if (segmentId != -1) _idleTime += segmentTime;
}

void korali::ProfileInfo::endWorkSegment()
{
 startWorkSegment(-1);
}
