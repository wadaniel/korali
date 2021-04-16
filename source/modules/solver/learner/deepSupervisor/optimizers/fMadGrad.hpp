/**************************************************************
 * A single-precision fast version of MADGRAD for Learning
 **************************************************************/

#ifndef _KORALI_FAST_MADGRAD_HPP_
#define _KORALI_FAST_MADGRAD_HPP_

#include "fGradientBasedOptimizer.hpp"

namespace korali
{
/**
* @brief Class declaration for module: MADGRAD.
*/
class fMadGrad : public fGradientBasedOptimizer
{
  public:
  /**
 * @brief Default constructor for the optimizer
 * @param nVars Variable-space dimensionality
 */
  fMadGrad(size_t nVars);

  std::vector<float> _s;
  std::vector<float> _v;
  std::vector<float> _z;
  float _epsilon;

  virtual bool checkTermination() override;
  virtual void processResult(float evaluation, std::vector<float> &gradient) override;
  virtual void reset() override;
  virtual void printInfo() override;
};

} // namespace korali

#endif // _KORALI_FAST_MADGRAD_HPP_
