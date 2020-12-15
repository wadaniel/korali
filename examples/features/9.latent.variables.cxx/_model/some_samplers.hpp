#ifndef _SAMPLER_HPP_
#define _SAMPLER_HPP_

#include "korali.hpp"
#include "model.hpp"
#include "utils.hpp"

#include <random>
#include <vector>

void dummySampler(korali::Sample &k);

class MCMCLatentSampler
{
  //  define any parameters
  std::vector<double> previousSampleMeans;
  std::vector<double> currentHyperparameters;
  std::vector<std::vector<double>> currentSamples;
  std::vector<std::vector<double>> previousSamples;
  int numberLatent;
  int numberHyperparameters;

  bool sample_discrete = false;
  int min_if_discrete = 0;
  int max_if_discrete = 0;

  std::function<void(korali::Sample &)> zeta_func;
  std::function<void(korali::Sample &)> S_func;
  std::function<void(korali::Sample &)> phi_func;

  std::mt19937 gen;
  std::uniform_int_distribution<int> random_int;

  public:
  MCMCLatentSampler(int numberLatentVars, int numberHyperparams, std::vector<double> initialLatentValues, std::vector<double> initialHyperparams, std::function<void(korali::Sample &)> zeta_, std::function<void(korali::Sample &)> S_, std::function<void(korali::Sample &)> phi_, bool sample_discrete_ = false, int min_if_discrete_ = 0, int max_if_discrete_ = 0);
  void initialize();
  void sampleLatent(korali::Sample &k);
};

class MultimodalGaussianSampler
{
  private:
  int nDimensions;
  int nClusters;
  int nPoints;
  std::vector<std::vector<double>> points;

  // Initialize random number generation
  // (https://stackoverflow.com/questions/38244877/how-to-use-stdnormal-distribution)
  // Not sure if it's possible to put part of this into .cpp
  //std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<> uniform_distrib;

  public:
  //MultimodalGaussianSampler  : gen{std::random_device{}()}, uniform_distrib(0.0, 1.0) (std::vector<std::vector<double>> points_, int nDimensions_, int nClusters_);
  MultimodalGaussianSampler(std::vector<std::vector<double>> points_, int nDimensions_, int nClusters_);
  void sampleLatent(korali::Sample &k);
};

#endif
