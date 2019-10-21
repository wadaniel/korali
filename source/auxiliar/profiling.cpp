#include "auxiliar/profiling.hpp"
#include "stdio.h"

std::map<std::string, korali::ProfileInfo> korali::_profileInfo;

korali::ProfileInfo::ProfileInfo()
{
 _runTime = 0.0;
 _workTime = 0.0;
 _idleTime = 0.0;

 _segmentId.push_back(-1);
 _segmentTimes.push_back(std::chrono::high_resolution_clock::now());
}

void korali::ProfileInfo::startSegment(int segmentId)
{
 size_t currentSegment = _segmentTimes.size() - 1;
 auto currentTime = std::chrono::high_resolution_clock::now();
 std::chrono::duration<double> elapsed = currentTime - _segmentTimes[currentSegment];
 double segmentTime = elapsed.count();

 _segmentId.push_back(segmentId);
 _segmentTimes.push_back(currentTime);

 _runTime += segmentTime;
 if (segmentId == -1) _workTime += segmentTime;
 if (segmentId != -1) _idleTime += segmentTime;
}

void korali::ProfileInfo::commitSegment()
{
 startSegment(-1);
}

void korali::ProfileInfo::discardSegment()
{
 size_t currentSegment = _segmentTimes.size() - 1;
 _segmentId.erase(_segmentId.begin() + currentSegment);
 _segmentTimes.erase(_segmentTimes.begin() + currentSegment);
}

nlohmann::json korali::ProfileInfo::dumpJson() const
{
 auto js = nlohmann::json();

 size_t segmentCount = 0;
 for (size_t i = 1; i < _segmentTimes.size(); i += 2)
 {
  js["Segments"][segmentCount]["Start Time"] =  std::chrono::duration<double>(_segmentTimes[i]-_segmentTimes[0]).count();

  auto endTimestamp =  std::chrono::high_resolution_clock::now();
  if (i+1 < _segmentTimes.size()) endTimestamp = _segmentTimes[i+1];

  js["Segments"][segmentCount]["End Time"]   =  std::chrono::duration<double>(endTimestamp-_segmentTimes[0]).count();
  js["Segments"][segmentCount]["Segment Id"] =  _segmentId[i];
  segmentCount++;
 }

 js["Idle Time"] = _idleTime;
 js["Work Time"] = _workTime;

 return js;
}
