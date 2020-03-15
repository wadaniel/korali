#ifndef _SAMPLER_CPP_
#define _SAMPLER_CPP_

#include "some_samplers.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>
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
  for(size_t i = 0; i < numberSamples; i++){
       std::vector<double> v;
       for(size_t j=0; j < numberLatentVars; j++){
          v.push_back(0); // @suppress("Method cannot be resolved")
    }
    samples.push_back(v);
  }

  k["Samples"] = samples;
}



MCMCLatentSampler::MCMCLatentSampler(int numberLatentVars, int numberHyperparams,
                    std::vector<double> initialLatentValues, std::vector<double> initialHyperparams,
                    std::function<void(korali::Sample&)> zeta_, std::function<void(korali::Sample&)> S_,
                            std::function<void(korali::Sample&)> phi_ )
       {

  /*  zeta_func = reinterpret_cast<uint64_t>(&zeta_);
    S_func = reinterpret_cast<uint64_t>(&S_);
    phi_func = reinterpret_cast<uint64_t>(&phi_);
*/
    zeta_func = zeta_;
    S_func = S_;
    phi_func = phi_;

    if (initialLatentValues.size() != numberLatentVars ) throw std::invalid_argument("number of latent variables should match the dimension of their initial values");
    if (initialHyperparams.size() != numberHyperparams ) throw std::invalid_argument("number of hyperparameters should match the dimension of their initial values");

    previousSampleMeans = initialLatentValues;
    currentHyperparameters = initialHyperparams;

    numberLatent = numberLatentVars;
    numberHyperparameters = numberHyperparams;



    }


  /*  void MCMCLatentSampler::initialize(int numberSamples){

      _currentSamples.resize( _numberMarkovChainSamples);
      for (size_t j = 0; j < _numberMarkovChainSamples; j++){
          _currentSamples[j].resize( numberSamples);
          for (size_t i = 0; i < _numberLatent; i++){
              size_t idx = _latentProblem->_latentVariableIndices[i];
              _currentSamples[j][i] = _k->_variables[idx]->_initialValue;    // _k->_variables: vector of korali-variables* (std::vector<korali::Variable*>).
          }
      }
    }
*/
    void MCMCLatentSampler::sampleLatent(korali::Sample& kSample)
    {
        /*
         * probability to sample from:
         * p(d, z | theta) * p(theta) -- that's the (log-)posterior
         * - use a "Custom" bayesian problem, with our log-posterior as "Likelihood Model" , but with the current values for theta inserted.
        */

      std::vector<double> hyperparameters = kSample["Hyperparameters"];
      size_t numberSamples = kSample["Number Samples"];
      size_t numberLatentVars = kSample["Number Of Latent Variables"];


    /* Create one sampling experiment to sample all latent variables. After all, the latent vars are correlated /
        have a joint distrib.
        Todo: does it make sense to re-create these experiments at every E-M step? Otherwise, how
            to automatically update the initial mean and the distribution function of the sampling experiments?*/

         auto k = korali::Engine();
         auto e = korali::Experiment();

         // Based on tutorial a2-sampling
         e["Problem"]["Type"] = "Sampling";
         e["Problem"]["Probability Function"] = [hparams=hyperparameters, this](korali::Sample& s) -> void {
                        if (! s.contains("Parameters")){
                            korali::logError("Something is trying to evaluate the likelihood without passing values for the latent variables (= parameters, here) to the sample.\n");
                        }
                        s["Latent Variables"] = s["Parameters"];
                        s["Hyperparameters"] = hparams; // _currentHyperparameters;
                        // Ugly? & Probably doesnt work

                        S_func(s);
                        zeta_func(s);
                        phi_func(s);
                        //s.run(S_func);
                        // s.run(zeta_func);
                        // s.run(phi_func); // @suppress("Invalid arguments")
                          // -> Assume these set: sample["S"], sample["zeta"] and sample["phi"]
                          if (! s.contains("S")) korali::logError("The specified likelihood model did not assign the value: 'S' to the sample.\n"); // @suppress("Invalid arguments")
                          if (! s.contains("zeta")) korali::logError("The specified likelihood model did not assign the value: 'zeta' to the sample.\n");
                          if (! s.contains("phi")) korali::logError("The specified likelihood model did not assign the value: 'phi' to the sample.\n");

                          double _zetaValue = s["zeta"].get<double>();
                          std::vector<double> _sValues = s["S"].get<std::vector<double>>();
                          std::vector<double> _phiValues = s["phi"].get<std::vector<double>>();

                          s["P(x)"] = - _zetaValue + std::inner_product(std::begin(_sValues), std::end(_sValues), // @suppress("Function cannot be resolved")
                                                                                                            std::begin(_phiValues), 0.0);

                    };

        for (size_t i=0; i < numberLatent; i++){

            // Defining problem's variables
            std::string varName = "latent_" + std::to_string(i);

            e["Variables"][i]["Name"] = varName;
            e["Variables"][i]["Initial Mean"] = previousSampleMeans[i];
            e["Variables"][i]["Initial Standard Deviation"] = 1.0;
        }

        // Configuring the MCMC sampler parameters
        e["Solver"]["Type"]  = "MCMC";
        e["Solver"]["Burn In"] = 500;
        e["Solver"]["Termination Criteria"]["Max Samples"] = 5000;

        // Configuring output settings
        e["File Output"]["Frequency"] = 500;
        e["Console Output"]["Frequency"] = 500;
        e["Console Output"]["Verbosity"] = "Detailed";

        // Todo: I don't think a result path is needed (and it'd need a step id in the pathname as well)
        //e["Results"]["Path"] = "setup/results_phase_1/" + "0"*(3 - str(i).length()) +  std:to_string(i);
        k.run(e);

        std::vector<std::vector<double>> db = e["Solver"]["Sample Database"].get<std::vector<std::vector<double>>>();
        printf("Database size: %lu\n", db.size());
        /*for (size_t i = 0; i < db.size(); i++)
        {
        printf("[ ");
        for (size_t j = 0; j < db[i].size(); j++)
        printf("%f, ", db[i][j]);
        printf("]\n");
        }*/
        // TODO: modify this
        std::vector<std::vector<double>>::const_iterator first = db.end() - numberSamples;
        std::vector<std::vector<double>>::const_iterator last = db.end();
        std::vector<std::vector<double>> samples(first, last);

        kSample["Samples"] = samples;

        // set new "previous sample means"
        for(size_t i= 0; i< numberLatent; i++){ // @suppress("Type cannot be resolved")
            double sum = 0;
            for(size_t j = 0; j < numberSamples; j++) { // @suppress("Type cannot be resolved")
                sum += samples[j][i];
                }
            previousSampleMeans[i] = sum / static_cast<double>(numberSamples);
        }

      //

    }


#endif
