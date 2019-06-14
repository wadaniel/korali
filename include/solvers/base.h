#ifndef _KORALI_SOLVER_BASE_H_
#define _KORALI_SOLVER_BASE_H_

#include <chrono>
#include <stdlib.h>
#include "problems/base.h"
#include "json/json.hpp"

namespace Korali { namespace Solver {

class Base {
 public:

 // Reporting Settings
 std::chrono::time_point<std::chrono::system_clock> startTime, endTime;
 std::chrono::time_point<std::chrono::system_clock> t0, t1, t2, t3;

 // Constructor / Destructor
 Base(nlohmann::json& js);
 ~Base();

 std::string _name;

 // Running Methods
 virtual void run() = 0;
 virtual void processSample(size_t sampleId, double fitness) = 0;

 // Serialization Methods
 virtual nlohmann::json getConfiguration();
 virtual void setConfiguration(nlohmann::json& js);
 virtual void setState(nlohmann::json& js);
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_BASE_H_
