#ifndef _KORALI_SOLVER_BASE_H_
#define _KORALI_SOLVER_BASE_H_

#include <chrono>
#include <stdlib.h>
#include "problems/base/base.h"
#include "auxiliar/json/json.hpp"

namespace Korali::Solver
{

class Base {
 public:

 // Reporting Settings
 std::chrono::time_point<std::chrono::system_clock> startTime, endTime;
 std::chrono::time_point<std::chrono::system_clock> t0, t1, t2, t3;
 std::string _plottingScript;

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 // Running Methods
 virtual void run() = 0;
 virtual void processSample(size_t sampleId, double fitness) = 0;

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
 virtual void setState(nlohmann::json& js);
};

} // namespace Korali

#endif // _KORALI_SOLVER_BASE_H_
