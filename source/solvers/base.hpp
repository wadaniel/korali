#ifndef _KORALI_SOLVER_BASE_HPP_
#define _KORALI_SOLVER_BASE_HPP_

#include <stdlib.h>
#include <chrono>
#include <vector>

#include "problems/base.hpp"

namespace Korali { namespace Solver {

class Base {
 public:

 size_t N; // Number of variables in the problem

 virtual void initialize() = 0;
 virtual void finalize() = 0;

 virtual void printGeneration() = 0;
 virtual void runGeneration() = 0;
 virtual bool checkTermination() = 0;

 virtual void setConfiguration(nlohmann::json& js) = 0;
 virtual void getConfiguration(nlohmann::json& js) = 0;

 virtual void processSample(size_t sampleId, double fitness) = 0;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_BASE_HPP_
