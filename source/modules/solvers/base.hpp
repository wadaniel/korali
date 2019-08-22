#ifndef _KORALI_SOLVER_BASE_HPP_
#define _KORALI_SOLVER_BASE_HPP_

#include "modules/module.hpp"

namespace Korali { namespace Solver {

class Base : public Korali::Module {

 public:

 size_t N; // Sample Size

 virtual void processSample(size_t sampleId, double fitness) = 0;
 virtual void printGeneration() = 0;
 virtual void runGeneration() = 0;

};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_BASE_HPP_
