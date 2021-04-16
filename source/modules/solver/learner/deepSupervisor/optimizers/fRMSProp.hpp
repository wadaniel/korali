/**************************************************************
 * A single-precision fast version of MADGRAD for Learning
 **************************************************************/

#ifndef _KORALI_FAST_RMSPROP_HPP_
#define _KORALI_FAST_RMSPROP_HPP_

#include "fGradientBasedOptimizer.hpp"

namespace korali
{
/**
* @brief Class declaration for module: MADGRAD.
*/
class fRMSProp : public fGradientBasedOptimizer
{
  public:
  /**
 * @brief Default constructor for the optimizer
 * @param nVars Variable-space dimensionality
 */
  fRMSProp(size_t nVars);

  std::vector<float> _r;
  std::vector<float> _v;
  float _epsilon;
  float _decay;

  virtual bool checkTermination() override;
  virtual void processResult(float evaluation, std::vector<float> &gradient) override;
  virtual void reset() override;
  virtual void printInfo() override;
};

} // namespace korali

#endif // _KORALI_FAST_RMSPROP_HPP_
