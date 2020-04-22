//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Optimizer_h
#define smarties_Optimizer_h

#include "Layers/Parameters.h"
#include <random>

namespace smarties
{

class Optimizer
{
protected:
  const std::shared_ptr<Parameters> weights;
  const std::vector<std::shared_ptr<Parameters>> sampled_weights = allocManyParams(weights);
  const std::shared_ptr<Parameters> target_weights = weights->allocateEmptyAlike();

  Uint cntUpdateDelay = 0;

  using NetSaveF_t = std::function<void(const Parameters*const, const std::string, const bool)>;
  using NetLoadF_t = std::function<int(const Parameters*const, const std::string)>;

public:

  bool bAnnealLearnRate = true;
  const Real eta_init = 1e-4;
  nnReal eta = eta_init;
  const Uint batchSize = 256;
  Real lambda = std::numeric_limits<float>::epsilon();
  const Real epsAnneal = 0;
  const Real tgtUpdateAlpha = 0;
  long unsigned nStep = 0;

  Optimizer(const std::shared_ptr<Parameters>& W);
  virtual ~Optimizer();
  virtual void save(const NetSaveF_t& F,
                    const std::string fname,
                    const bool bBackup) = 0;

  virtual void prepare_update(const Rvec&L) = 0;
  virtual void apply_update() = 0;

  virtual void getMetrics(std::ostringstream& buff) = 0;
  virtual void getHeaders(std::ostringstream&buff,const std::string nnName) = 0;
  virtual bool ready2UpdateWeights() = 0;

  const Parameters * getWeights(const Sint weightsIndex)
  {
    if(weightsIndex == 0) return weights.get();
    if(weightsIndex <  0) return target_weights.get();
    assert((Uint) weightsIndex < sampled_weights.size());
      return sampled_weights[weightsIndex].get();

    return sampled_weights[weightsIndex].get();
  }
};

class AdamOptimizer : public Optimizer
{
protected:
  const Real beta_1, beta_2;
  Real beta_t_1 = beta_1, beta_t_2 = beta_2;
  const std::vector<std::shared_ptr<Parameters>> gradients;
  const std::shared_ptr<Parameters> gradSum = weights->allocateEmptyAlike();
  const std::shared_ptr<Parameters> _1stMom = weights->allocateEmptyAlike();
  const std::shared_ptr<Parameters> _2ndMom = weights->allocateEmptyAlike();

  // Instantiating generator
  std::mt19937 generator;

public:

  AdamOptimizer(const std::shared_ptr<Parameters>& W,
                const std::vector<std::shared_ptr<Parameters>> & G,
                const Real B1=.9, const Real B2=.999);

  void prepare_update(const Rvec& L) override;
  bool ready2UpdateWeights() override
  {
    return true;
  }
  void apply_update() override;

  void save(const NetSaveF_t& F,
            const std::string fname,
            const bool bBackup) override;
  void getMetrics(std::ostringstream& buff) override;
  void getHeaders(std::ostringstream& buff, const std::string nnName) override;
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
