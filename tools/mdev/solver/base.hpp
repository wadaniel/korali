#ifndef _KORALI_SOLVER_[MODULETAG] _HPP_
#define _KORALI_SOLVER_ [MODULETAG] _HPP_

#include "modules/solver/solver.hpp"

namespace korali
{
namespace solver
{
class[CLASSNAME] : public korali::Solver
{
  public:
  void setInitialConfiguration() override;
  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;
};

} // namespace solver
} // namespace korali

#endif // _KORALI_SOLVER_[MODULETAG]_HPP_
