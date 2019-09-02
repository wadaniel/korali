#ifndef _KORALI_SOLVER_BASE_HPP_
#define _KORALI_SOLVER_BASE_HPP_

#include <vector>
#include "modules/base.hpp"
#include "modules/engine/engine.hpp"

namespace Korali { namespace Solver {

class Base : public Korali::Base
{
 public:

 std::vector<Korali::Variable*> _variables;
 size_t N; // Sample Size

 virtual void processEvaluation(const double evaluation, const size_t sampleId) = 0;
 virtual void printGeneration() = 0;
 virtual void runGeneration() = 0;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_BASE_HPP_
