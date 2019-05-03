#ifndef _BASESOLVER_H_
#define _BASESOLVER_H_

#include <chrono>
#include <stdlib.h>
#include "problems/base/base.h"
#include "json/json.hpp"

namespace Korali::Solver
{

class Base {
 public:

 // Reporting Settings
 std::chrono::time_point<std::chrono::system_clock> startTime, endTime;
 std::chrono::time_point<std::chrono::system_clock> t0, t1, t2, t3;

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

 // Print Methods
 virtual void printGeneration() const = 0;
 virtual void printFinal() const = 0;
 virtual void startPlot() const = 0;
};

} // namespace Korali

#endif // _BASESOLVER_H_
