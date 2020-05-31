#ifndef _KORALI_PROBLEM_[MODULETAG] _HPP_
#define _KORALI_PROBLEM_ [MODULETAG] _HPP_

#include "modules/problem/problem.hpp"

namespace korali
{
namespace problem
{
class[CLASSNAME] : public korali::Problem
{
  public:
  void initialize() override;
  void evaluate(korali::Sample & sample);
};

} // namespace problem
} // namespace korali

#endif // _KORALI_PROBLEM_[MODULETAG]_HPP_
