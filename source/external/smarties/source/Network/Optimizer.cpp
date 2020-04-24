//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Optimizer.h"
#include "../Utils/SstreamUtilities.h"
#include "../extern/saruprng.h"
#include <unistd.h>

namespace smarties
{

struct AdaMax
{
  const nnReal eta, B1, B2, lambda, fac;
  AdaMax(nnReal _eta, nnReal beta1, nnReal beta2, nnReal betat1,
    nnReal betat2, nnReal _lambda, nnReal _fac, Saru& _gen) :
    eta(_eta), B1(beta1), B2(beta2), lambda(_lambda), fac(_fac)
  {}

  #ifdef NDEBUG
  #pragma omp declare simd notinbranch //simdlen(VEC_WIDTH)
  #endif
  inline nnReal step(const nnReal grad, nnReal&M1, nnReal&M2, nnReal&M3, const nnReal W) const
  {
    const nnReal DW = grad * fac;
    M1 = B1 * M1 + (1-B1) * DW;
    M2 = std::max(B2 * M2, std::fabs(DW));
    #ifdef SMARTIES_NESTEROV_ADAM
      const nnReal numer = B1*M1 + (1-B1)*DW;
    #else
      const nnReal numer = M1;
    #endif
    #ifdef NET_L1_PENAL
      const nnReal penal = W>0 ? - lambda : lambda;
    #else
      const nnReal penal = - W * lambda;
    #endif
    return eta * (numer/std::max(M2, nnEPS) + penal);
  }
};

template <class T>
struct Entropy : public T
{
  Saru& gen;
  const nnReal eps;
  Entropy(nnReal _eta, nnReal beta1, nnReal beta2, nnReal bett1, nnReal bett2,
    nnReal _lambda, nnReal _fac, Saru& _gen) : T(_eta, beta1, beta2, bett1,
    bett2, _lambda, _fac, _gen), gen(_gen), eps(_eta*_lambda) {}

  inline nnReal step(const nnReal grad, nnReal&M1, nnReal&M2, const nnReal W) {
    return T::step(grad, M1, M2, W) + eps * gen.d_mean0_var1();
  }
};

struct Adam
{
  const nnReal eta, B1, B2, lambda, fac;
  Adam(nnReal _eta, nnReal beta1, nnReal beta2, nnReal betat1,
    nnReal betat2, nnReal _lambda, nnReal _fac, Saru& _gen) :
    eta(_eta*std::sqrt(1-betat2)/(1-betat1)), B1(beta1), B2(beta2),
    lambda(_lambda), fac(_fac)
  {}

  #ifdef NDEBUG
  #pragma omp declare simd notinbranch //simdlen(VEC_WIDTH)
  #endif
  inline nnReal step(const nnReal grad, nnReal&M1, nnReal&M2, nnReal&M3, const nnReal W) const
  {
    #ifdef NET_L1_PENAL
      const nnReal penal = W>0 ? - lambda : lambda;
    #else
      const nnReal penal = - W * lambda;
    #endif
    #ifndef SMARTIES_ADAMW
    const nnReal DW = fac * grad + penal;
    #else
    const nnReal DW = fac * grad;
    #endif
    M1 = B1 * M1 + (1-B1) * DW;
    M2 = B2 * M2 + (1-B2) * DW*DW;
    #ifdef SMARTIES_NESTEROV_ADAM // No significant effect
      const nnReal numer = B1*M1 + (1-B1)*DW;
    #else
      const nnReal numer = M1;
    #endif
    #ifdef SMARTIES_SAFE_ADAM //numerical safety, assumes that 1-beta2 = (1-beta1)^2/10
      //assert( std::fabs( (1-B2) - 0.1*std::pow(1-B1,2) ) < nnEPS );
      M2 = M2 < M1*M1 ? M1*M1 : M2;
    #endif
    const nnReal ret = numer / ( nnEPS + std::sqrt(M2) );
    assert(not std::isnan(ret) && not std::isinf(ret));
    #ifdef SMARTIES_ADAMW
      return eta * ( ret + penal );
    #else
      return eta * ret;
    #endif
  }
};

void AdamOptimizer::prepare_update(const Rvec& esLosses)
{
  gradSum->reduceThreadsGrad(gradients);
  nStep++;
}

void AdamOptimizer::apply_update()
{
  if(nStep == 0) die("nStep == 0");

  using Algorithm = Adam;
  //update is deterministic: can be handled independently by each node
  //communication overhead is probably greater than a parallelised sum

  const Real factor = 1.0 / batchSize;
  nnReal* const paramAry = weights->params;
  const nnReal _eta = bAnnealLearnRate? Utilities::annealRate(eta,nStep,epsAnneal) : eta;

  {
    const Uint thrID = 0;
    Saru gen(nStep, thrID, generator()); //needs 3 seeds
    Algorithm algo(_eta,beta_1,beta_2,beta_t_1,beta_t_2,lambda,factor,gen);
    nnReal* const M1 = _1stMom->params;
    nnReal* const M2 = _2ndMom->params;
    #ifdef AMSGRAD
      nnReal* const M3 = _2ndMax->params; // holds max of second moment
    #else
      nnReal* const M3 = _2ndMom->params; // unused
    #endif
    nnReal* const G  = gradSum->params;

    for (Uint i=0; i<weights->nParams; ++i)
    paramAry[i] += algo.step(G[i], M1[i], M2[i], M3[i], paramAry[i]);
  }

  gradSum->clear();
  // Needed by Adam optimization algorithm:
  beta_t_1 *= beta_1;
  if (beta_t_1<nnEPS) beta_t_1 = 0;
  beta_t_2 *= beta_2;
  if (beta_t_2<nnEPS) beta_t_2 = 0;

  // update frozen weights:
  if(tgtUpdateAlpha > 0 && target_weights) {
    if (cntUpdateDelay == 0) {
      // the targetDelay setting param can be either >1 or <1.
      // if >1 then it means "every how many steps copy weight to tgt weights"
      cntUpdateDelay = tgtUpdateAlpha;
      if(tgtUpdateAlpha>=1) target_weights->copy(weights);
      else { // else is the learning rate of an exponential averaging
        nnReal* const targetAry = target_weights->params;
        for(Uint j=0; j<weights->nParams; ++j)
          targetAry[j] += tgtUpdateAlpha*(paramAry[j] - targetAry[j]);
      }
    }
    if(cntUpdateDelay>0) cntUpdateDelay--;
  }
}

void AdamOptimizer::save(const NetSaveF_t& saveFunc,
                         const std::string fname,
                         const bool backup)
{
  saveFunc(weights.get(), fname+"_weights", backup);
  saveFunc(_1stMom.get(), fname+"_1stMom", backup);
  saveFunc(_2ndMom.get(), fname+"_2ndMom", backup);

  if(target_weights)
    saveFunc(target_weights.get(), fname+"_tgt_weights", backup);

  if(backup) {
    std::ostringstream ss; ss << std::setw(9) << std::setfill('0') << nStep;
    saveFunc(weights.get(), fname+"_"+ss.str()+"_weights", false);
    saveFunc(_1stMom.get(), fname+"_"+ss.str()+"_1stMom", false);
    saveFunc(_2ndMom.get(), fname+"_"+ss.str()+"_2ndMom", false);
  }
}

void AdamOptimizer::getMetrics(std::ostringstream& buff)
{
  Utilities::real2SS(buff, weights->compute_weight_norm(), 7, 1);
  if(tgtUpdateAlpha > 0)
   Utilities::real2SS(buff, weights->compute_weight_dist(target_weights), 6, 1);
}

void AdamOptimizer::getHeaders(std::ostringstream&buff,const std::string nnName)
{
  buff << std::left << std::setfill(' ') <<"| " << std::setw(6) << nnName;
  if(tgtUpdateAlpha > 0) buff << "| dTgt ";
}

Optimizer::Optimizer(const std::shared_ptr<Parameters>& W) : weights(W) {
  target_weights->copy(weights);
}

AdamOptimizer::AdamOptimizer(const std::shared_ptr<Parameters>& W,
                             const std::vector<std::shared_ptr<Parameters>> & G,
                             const Real beta1, const Real beta2) :
Optimizer(W), beta_1(beta1), beta_2(beta2), gradients(G) {
    printf("Optimizer: Parameter updates using Adam SGD algorithm.\n");
}

Optimizer::~Optimizer()
{
}

} // end namespace smarties
