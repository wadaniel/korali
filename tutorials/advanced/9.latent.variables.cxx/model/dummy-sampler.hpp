#ifndef _SAMPLER_HPP_
#define _SAMPLER_HPP_

#include "korali.hpp"

#include <vector>
/*
void direct(korali::Sample& k)
{
  float x = k["Parameters"][0];
  k["Evaluation"] = -0.5*x*x;
}*/

void dummySampler(korali::Sample& k) //, int numberLatentVars)
{

// Always samples 0

    // Test that all input parameters are set:
  // Edit: We can't have access to those: std::vector<double> latentVars = k["Latent Variables"];

  std::vector<double> hyperparams = k["Hyperparameters"];
  size_t numberSamples = k["Number Samples"];
  size_t numberLatentVars = k["Number Of Latent Variables"];

  std::vector<std::vector<double>> samples;
  for(size_t i; i < numberSamples; i++){
    std::vector<double> v; 
       for(size_t j; j < numberLatentVars; j++){
          v.push_back(0);
    }
    samples.push_back(v);
  }

  k["Samples"] = samples;  //-0.5*x*x;
}




#endif
