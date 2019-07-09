#ifndef _KORALI_SOLVER_BASE_H_
#define _KORALI_SOLVER_BASE_H_

#include <stdlib.h>
#include <chrono>
#include <vector>

#include "problems/base.h"
#include "json/json.hpp"

namespace Korali { namespace Solver {

class Base {
 public:

 virtual void initialize() = 0;
 virtual void finalize() = 0;

 virtual void printGeneration() = 0;
 virtual void runGeneration() = 0;
 virtual bool checkTermination() = 0;

 virtual void setConfiguration() = 0;
 virtual void getConfiguration() = 0;

 virtual void processSample(size_t sampleId, double fitness) = 0;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_BASE_H_
